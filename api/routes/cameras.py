# api/routes/cameras.py
from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.common import execute_or_500, get_data, get_one_or_404
from api.supabase_client import get_supabase
from api.schemas import CameraCreateRequest, CameraUpdateRequest, CameraResponse

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=List[CameraResponse])
def list_cameras(
    is_active: Optional[bool] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
) -> Any:
    sb = get_supabase()

    def _run():
        q = sb.table("cameras").select("*").limit(limit)
        if is_active is not None:
            q = q.eq("in_active", is_active)
        return q.execute()

    resp = execute_or_500(_run, "list cameras")
    data = get_data(resp)
    # Map in_active -> is_active for frontend
    for row in data:
        if "in_active" in row:
            row["is_active"] = row.pop("in_active")
    return data


@router.post("", response_model=CameraResponse)
def create_camera(body: CameraCreateRequest) -> Any:
    sb = get_supabase()
    payload = body.model_dump(exclude_none=True)
    # Map is_active -> in_active for DB
    if "is_active" in payload:
        payload["in_active"] = payload.pop("is_active")

    resp = execute_or_500(lambda: sb.table("cameras").insert(payload).execute(), "create camera")
    res = get_one_or_404(resp, "Insert failed (no row returned)")
    if "in_active" in res:
        res["is_active"] = res.pop("in_active")
    return res


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: str) -> Any:
    sb = get_supabase()
    resp = execute_or_500(
        lambda: sb.table("cameras").select("*").eq("camera_id", camera_id).maybe_single().execute(),
        "get camera",
    )
    res = get_one_or_404(resp, "Camera not found")
    if "in_active" in res:
        res["is_active"] = res.pop("in_active")
    return res


@router.patch("/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: str, body: CameraUpdateRequest) -> Any:
    sb = get_supabase()
    payload = body.model_dump(exclude_none=True)
    if "is_active" in payload:
        payload["in_active"] = payload.pop("is_active")
        
    if not payload:
        raise HTTPException(status_code=400, detail="No fields to update")

    resp = execute_or_500(
        lambda: sb.table("cameras").update(payload).eq("camera_id", camera_id).execute(),
        "update camera",
    )
    res = get_one_or_404(resp, "Camera not found")
    if "in_active" in res:
        res["is_active"] = res.pop("in_active")
    return res


@router.delete("/{camera_id}")
def delete_camera(camera_id: str) -> Any:
    sb = get_supabase()
    resp = execute_or_500(
        lambda: sb.table("cameras").delete().eq("camera_id", camera_id).execute(),
        "delete camera",
    )
    if not get_data(resp):
        raise HTTPException(status_code=404, detail="Camera not found or already deleted")
    return {"ok": True}
