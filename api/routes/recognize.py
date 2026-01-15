# api/routes/recognize.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import traceback
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from api.supabase_client import get_supabase
from api.embedding import get_embedding_from_image_bytes

router = APIRouter(tags=["recognize"])

# ✅ DB enum(event_type)에 맞춰 강제 통일
VALID_EVENT_TYPES = {"CHECK_IN", "CHECK_OUT"}


# -----------------------------
# Utilities
# -----------------------------
def _raise_if_error(resp: Any, msg: str) -> None:
    """
    supabase-py 응답에서 error가 있으면 FastAPI 예외로 변환.
    (Render에서도 원인 파악 가능하도록 detail 풍부하게)
    """
    err = getattr(resp, "error", None)
    if err:
        raise HTTPException(status_code=500, detail={"msg": msg, "error": repr(err)})


def _normalize_event_type(v: str) -> str:
    raw = (v or "").strip()
    up = raw.upper()

    # 흔한 변형 흡수
    if up in {"CHECKIN", "CHECK-IN", "CHECK_IN"}:
        up = "CHECK_IN"
    elif up in {"CHECKOUT", "CHECK-OUT", "CHECK_OUT"}:
        up = "CHECK_OUT"

    if up not in VALID_EVENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type: {raw!r}. Use one of {sorted(VALID_EVENT_TYPES)}",
        )
    return up


def _parse_pgvector(v: Any) -> Optional[np.ndarray]:
    """
    face_embeddings.embedding 이 pgvector일 가능성이 높음.
    supabase python client에서 문자열/리스트 등으로 올 수 있어 방어적으로 파싱.
    """
    if v is None:
        return None

    if isinstance(v, list):
        try:
            return np.asarray(v, dtype=np.float32)
        except Exception:
            return None

    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        if not s:
            return None
        try:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
            return arr if arr.size > 0 else None
        except Exception:
            return None

    return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def _ensure_camera_exists(camera_id: str) -> None:
    """
    attendance_logs.camera_id는 cameras.camera_id FK라서
    camera가 없으면 log insert가 실패.
    -> recognize에서 미리 upsert로 보장.
    """
    sb = get_supabase()
    resp = (
        sb.table("cameras")
        .upsert({"camera_id": camera_id, "is_active": True}, on_conflict="camera_id")
        .execute()
    )
    _raise_if_error(resp, "Failed to ensure camera exists")


def _fetch_all_embeddings(limit: int = 2000) -> List[Dict[str, Any]]:
    sb = get_supabase()
    resp = (
        sb.table("face_embeddings")
        .select("employee_id, embedding_dim, embedding")
        .limit(limit)
        .execute()
    )
    _raise_if_error(resp, "Failed to fetch face embeddings")
    return resp.data or []


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

    # ✅ postgrest.exceptions.APIError 같은 건 execute에서 "예외"로 터질 수 있음
    try:
        resp = sb.table("attendance_logs").insert(payload).execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"msg": "Failed to insert attendance log (exception)", "error": repr(e)},
        )

    _raise_if_error(resp, "Failed to insert attendance log")
    return (resp.data or [{}])[0]


# -----------------------------
# Route
# -----------------------------
@router.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    event_type: str = Form(...),
    camera_id: str = Form(...),
    threshold: float = Form(0.35),
) -> Dict[str, Any]:
    try:
        # ✅ 0) event_type normalize (DB enum 불일치 방지)
        event_type = _normalize_event_type(event_type)

        # 1) camera FK 보장
        camera_id = (camera_id or "").strip()
        if not camera_id:
            raise HTTPException(status_code=400, detail="camera_id is required")
        _ensure_camera_exists(camera_id)

        # 2) 이미지 -> 임베딩
        img_bytes = await file.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="empty file")

        try:
            query_emb = get_embedding_from_image_bytes(img_bytes).astype(np.float32)
        except Exception as e:
            raise HTTPException(status_code=500, detail={"msg": "embedding failed", "error": repr(e)})

        # 3) DB 임베딩 fetch -> best match
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

        best_emp_id: Optional[int] = None
        best_sim: float = -1.0

        for r in rows:
            emb = _parse_pgvector(r.get("embedding"))
            if emb is None:
                continue
            if emb.shape[0] != query_emb.shape[0]:
                continue
            sim = _cosine_similarity(query_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_emp_id = r.get("employee_id")

        recognized = bool(best_emp_id is not None and best_sim >= float(threshold))

        # 4) 직원 정보 + (원하면 비활성 제외)
        emp_brief: Dict[str, Any] = {}
        if recognized and best_emp_id is not None:
            emp_brief = _fetch_employee_brief(int(best_emp_id))
            if emp_brief.get("is_active") is False:
                recognized = False

        # 5) 로그 저장
        log_row = _insert_attendance_log(
            event_type=event_type,
            camera_id=camera_id,
            recognized=recognized,
            similarity=float(best_sim) if best_sim >= -0.5 else None,
            employee_id=int(best_emp_id) if recognized and best_emp_id is not None else None,
        )

        return {
            "recognized": recognized,
            "similarity": float(best_sim) if best_sim >= -0.5 else None,
            "employee_id": emp_brief.get("employee_id") if recognized else None,
            "name": emp_brief.get("name") if recognized else None,
            "employee_code": emp_brief.get("employee_code") if recognized else None,
            "camera_id": camera_id,
            "event_type": event_type,
            "log_id": log_row.get("log_id"),
            "event_time": log_row.get("event_time"),
            "created_at": log_row.get("created_at"),
        }

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "msg": "recognize crashed (unhandled exception)",
                "error": repr(e),
                "trace": tb[-2500:],
            },
        )
