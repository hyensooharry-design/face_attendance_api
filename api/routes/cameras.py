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
    limit: int = Query(default=200, ge=1, le=2000),
) -> Any:
    sb = get_supabase()

    def _run():
        # Render schema: cameras(camera_id, name, location, created_at)
        return sb.table("cameras").select("*").limit(limit).execute()

    resp = execute_or_500(_run, "list cameras")
    return get_data(resp)


@router.post("", response_model=CameraResponse)
def create_camera(body: CameraCreateRequest) -> Any:
    sb = get_supabase()
    payload = body.model_dump(exclude_none=True)

    resp = execute_or_500(lambda: sb.table("cameras").insert(payload).execute(), "create camera")
    return get_one_or_404(resp, "Insert failed (no row returned)")


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: str) -> Any:
    sb = get_supabase()
    resp = execute_or_500(
        lambda: sb.table("cameras").select("*").eq("camera_id", camera_id).maybe_single().execute(),
        "get camera",
    )
    return get_one_or_404(resp, "Camera not found")


@router.patch("/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: str, body: CameraUpdateRequest) -> Any:
    sb = get_supabase()
    payload = body.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No fields to update")

    resp = execute_or_500(
        lambda: sb.table("cameras").update(payload).eq("camera_id", camera_id).execute(),
        "update camera",
    )
    return get_one_or_404(resp, "Camera not found")


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
