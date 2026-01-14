from __future__ import annotations

from typing import Any, Optional, List

from fastapi import APIRouter, HTTPException, Query

from api.supabase_client import get_supabase
from api.schemas import CameraCreateRequest, CameraUpdateRequest, CameraResponse

router = APIRouter(prefix="/cameras", tags=["cameras"])


def _raise_if_error(resp: Any, msg: str) -> None:
    err = getattr(resp, "error", None)
    if err:
        raise HTTPException(status_code=500, detail=f"{msg}: {err}")


@router.get("", response_model=List[CameraResponse])
def list_cameras(
    is_active: Optional[bool] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
) -> Any:
    sb = get_supabase()
    query = sb.table("cameras").select("*").limit(limit)
    if is_active is not None:
        query = query.eq("is_active", is_active)

    resp = query.execute()
    _raise_if_error(resp, "Failed to list cameras")
    return resp.data or []


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: str) -> Any:
    sb = get_supabase()
    resp = sb.table("cameras").select("*").eq("camera_id", camera_id).maybe_single().execute()
    _raise_if_error(resp, "Failed to get camera")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Camera not found")
    return resp.data


@router.post("", response_model=CameraResponse)
def create_camera(payload: CameraCreateRequest) -> Any:
    sb = get_supabase()
    resp = sb.table("cameras").insert(payload.model_dump(exclude_none=True)).execute()
    _raise_if_error(resp, "Failed to create camera")
    if not resp.data:
        raise HTTPException(status_code=500, detail="Insert succeeded but returned no data")
    return resp.data[0]


@router.patch("/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: str, payload: CameraUpdateRequest) -> Any:
    sb = get_supabase()
    data = payload.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(status_code=400, detail="No fields to update")

    resp = sb.table("cameras").update(data).eq("camera_id", camera_id).execute()
    _raise_if_error(resp, "Failed to update camera")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Camera not found")
    return resp.data[0]


@router.delete("/{camera_id}")
def delete_camera(camera_id: str) -> Any:
    sb = get_supabase()
    resp = sb.table("cameras").delete().eq("camera_id", camera_id).execute()
    _raise_if_error(resp, "Failed to delete camera")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Camera not found or already deleted")
    return {"ok": True, "deleted": resp.data[0]}
