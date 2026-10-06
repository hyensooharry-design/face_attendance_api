from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.embedding import get_embedding_from_image_bytes
from api.supabase_client import get_supabase

router = APIRouter(tags=["recognize"])

VALID_EVENT_TYPES = {"CHECK_IN", "CHECK_OUT"}
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "0").strip() == "1"


def _raise_if_error(resp: Any, msg: str) -> None:
    err = getattr(resp, "error", None)
    if err:
        raise HTTPException(status_code=500, detail={"msg": msg, "error": repr(err)})


def _normalize_event_type(value: str) -> str:
    raw = (value or "").strip()
    upper = raw.upper()
    aliases = {
        "CHECKIN": "CHECK_IN",
        "CHECK-IN": "CHECK_IN",
        "CHECK_IN": "CHECK_IN",
        "CHECKOUT": "CHECK_OUT",
        "CHECK-OUT": "CHECK_OUT",
        "CHECK_OUT": "CHECK_OUT",
    }
    normalized = aliases.get(upper)
    if normalized not in VALID_EVENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type: {raw!r}. Use one of {sorted(VALID_EVENT_TYPES)}",
        )
    return normalized


def _parse_pgvector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, list):
        try:
            return np.asarray(value, dtype=np.float32)
        except Exception:
            return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not text:
            return None
        try:
            arr = np.fromstring(text, sep=",", dtype=np.float32)
            return arr if arr.size > 0 else None
        except Exception:
            return None
    return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        return -1.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def _ensure_camera_exists(camera_id: str) -> None:
    sb = get_supabase()
    resp = sb.table("cameras").upsert({"camera_id": camera_id}, on_conflict="camera_id").execute()
    _raise_if_error(resp, "Failed to ensure camera exists")


def _fetch_all_embeddings(limit: int = 2000) -> List[Dict[str, Any]]:
    sb = get_supabase()
    resp = (
        sb.table("face_embeddings")
        .select("person_id, embedding, persons(employee_id)")
        .limit(limit)
        .execute()
    )
    _raise_if_error(resp, "Failed to fetch face embeddings")
    return resp.data or []


def _extract_employee_id_from_row(row: Dict[str, Any]) -> Optional[int]:
    rel = row.get("persons")
    if isinstance(rel, dict):
        value = rel.get("employee_id")
        return int(value) if value is not None else None
    if isinstance(rel, list) and rel:
        value = (rel[0] or {}).get("employee_id")
        return int(value) if value is not None else None
    return None


def _fetch_employee_brief(employee_id: int) -> Dict[str, Any]:
    sb = get_supabase()
    resp = (
        sb.table("employees")
        .select("employee_id, name, employee_code, is_active")
        .eq("employee_id", employee_id)
        .single()
        .execute()
    )
    _raise_if_error(resp, "Failed to fetch employee")
    return resp.data or {}


def _insert_attendance_log(
    *,
    event_type: str,
    camera_id: str,
    recognized: bool,
    similarity: Optional[float],
    employee_id: Optional[int],
) -> Dict[str, Any]:
    sb = get_supabase()
    now = datetime.now(timezone.utc).isoformat()

    payload: Dict[str, Any] = {
        "event_time": now,
        "event_type": event_type,
        "camera_id": camera_id,
        "recognized": recognized,
        "similarity": similarity,
        "employee_id": employee_id,
        "created_at": now,
    }

    try:
        resp = sb.table("attendance_logs").insert(payload).execute()
    except Exception as exc:
        detail: Dict[str, Any] = {"msg": "Failed to insert attendance log"}
        if DEBUG_ERRORS:
            detail["error"] = repr(exc)
        raise HTTPException(status_code=500, detail=detail) from exc

    _raise_if_error(resp, "Failed to insert attendance log")
    return (resp.data or [{}])[0]


@router.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    event_type: str = Form(...),
    camera_id: str = Form(...),
    threshold: float = Form(0.35),
) -> Dict[str, Any]:
    event_type = _normalize_event_type(event_type)

    camera_id = (camera_id or "").strip()
    if not camera_id:
        raise HTTPException(status_code=400, detail="camera_id is required")

    if not 0.0 <= float(threshold) <= 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    _ensure_camera_exists(camera_id)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        query_embedding = get_embedding_from_image_bytes(image_bytes).astype(np.float32)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        detail: Dict[str, Any] = {"msg": "Face embedding failed"}
        if DEBUG_ERRORS:
            detail["error"] = repr(exc)
        raise HTTPException(status_code=500, detail=detail) from exc

    rows = _fetch_all_embeddings(limit=2000)
    if not rows:
        log_row = _insert_attendance_log(
            event_type=event_type,
            camera_id=camera_id,
            recognized=False,
            similarity=None,
            employee_id=None,
        )
        return {
            "recognized": False,
            "similarity": None,
            "employee_id": None,
            "name": None,
            "employee_code": None,
            "camera_id": camera_id,
            "event_type": event_type,
            "log_id": log_row.get("log_id"),
            "event_time": log_row.get("event_time"),
            "created_at": log_row.get("created_at"),
            "message": "No enrolled faces found in DB.",
        }

    best_employee_id: Optional[int] = None
    best_similarity = -1.0

    for row in rows:
        stored_embedding = _parse_pgvector(row.get("embedding"))
        employee_id = _extract_employee_id_from_row(row)
        if stored_embedding is None or employee_id is None:
            continue

        similarity = _cosine_similarity(query_embedding, stored_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_employee_id = employee_id

    recognized = bool(
        best_employee_id is not None and best_similarity >= float(threshold)
    )

    employee: Dict[str, Any] = {}
    if recognized and best_employee_id is not None:
        employee = _fetch_employee_brief(best_employee_id)
        if employee.get("is_active") is False:
            recognized = False

    log_row = _insert_attendance_log(
        event_type=event_type,
        camera_id=camera_id,
        recognized=recognized,
        similarity=float(best_similarity) if best_similarity >= -0.5 else None,
        employee_id=best_employee_id if recognized else None,
    )

    return {
        "recognized": recognized,
        "similarity": float(best_similarity) if best_similarity >= -0.5 else None,
        "employee_id": employee.get("employee_id") if recognized else None,
        "name": employee.get("name") if recognized else None,
        "employee_code": employee.get("employee_code") if recognized else None,
        "camera_id": camera_id,
        "event_type": event_type,
        "log_id": log_row.get("log_id"),
        "event_time": log_row.get("event_time"),
        "created_at": log_row.get("created_at"),
    }
