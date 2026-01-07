from __future__ import annotations

from typing import List, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.supabase_client import get_supabase

router = APIRouter(prefix="/cameras", tags=["cameras"])


# -----------------------------
# Schemas
# -----------------------------
class CameraCreateRequest(BaseModel):
    camera_id: str = Field(..., min_length=1)
    label: Optional[str] = None
    location: Optional[str] = None
    is_active: bool = True


class CameraResponse(BaseModel):
    camera_id: str
    label: Optional[str] = None
    location: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


def _raise_if_error(resp: Any, msg: str) -> None:
    err = getattr(resp, "error", None)
    if err:
        raise HTTPException(status_code=500, detail=f"{msg}: {err}")


# -----------------------------
# Endpoints
# -----------------------------
@router.post("", response_model=CameraResponse)
def upsert_camera(payload: CameraCreateRequest) -> CameraResponse:
    sb = get_supabase()

    resp = sb.table("cameras").upsert(payload.dict(), on_conflict="camera_id").execute()
    _raise_if_error(resp, "Failed to upsert camera")

    if not resp.data:
        raise HTTPException(status_code=500, detail="Camera upsert returned empty result")

    return CameraResponse(**resp.data[0])


@router.get("", response_model=List[CameraResponse])
def list_cameras(active_only: bool = True) -> List[CameraResponse]:
    sb = get_supabase()

    q = sb.table("cameras").select("*").order("camera_id")
    if active_only:
        q = q.eq("is_active", True)

    resp = q.execute()
    _raise_if_error(resp, "Failed to list cameras")

    return [CameraResponse(**r) for r in (resp.data or [])]


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: str) -> CameraResponse:
    sb = get_supabase()

    resp = (
        sb.table("cameras")
        .select("*")
        .eq("camera_id", camera_id)
        .limit(1)
        .execute()
    )
    _raise_if_error(resp, "Failed to fetch camera")

    if not resp.data:
        raise HTTPException(status_code=404, detail="Camera not found")

    return CameraResponse(**resp.data[0])
