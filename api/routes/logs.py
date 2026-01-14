from __future__ import annotations

from typing import Any, Optional, List

from fastapi import APIRouter, HTTPException, Query

from api.supabase_client import get_supabase
from api.schemas import (
    AttendanceLogCreateRequest,
    AttendanceLogUpdateRequest,
    AttendanceLogResponse,
)

router = APIRouter(prefix="/logs", tags=["logs"])


def _raise_if_error(resp: Any, msg: str) -> None:
    err = getattr(resp, "error", None)
    if err:
        raise HTTPException(status_code=500, detail=f"{msg}: {err}")


@router.get("", response_model=List[AttendanceLogResponse])
def list_logs(
    employee_id: Optional[int] = Query(default=None),
    camera_id: Optional[str] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    recognized: Optional[bool] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    order_desc: bool = Query(default=True),
) -> Any:
    sb = get_supabase()
    query = sb.table("attendance_logs").select("*").limit(limit)

    if employee_id is not None:
        query = query.eq("employee_id", employee_id)
    if camera_id is not None:
        query = query.eq("camera_id", camera_id)
    if event_type is not None:
        query = query.eq("event_type", event_type)
    if recognized is not None:
        query = query.eq("recognized", recognized)

    query = query.order("event_time", desc=order_desc)

    resp = query.execute()
    _raise_if_error(resp, "Failed to list logs")
    return resp.data or []


@router.get("/{log_id}", response_model=AttendanceLogResponse)
def get_log(log_id: int) -> Any:
    sb = get_supabase()
    resp = sb.table("attendance_logs").select("*").eq("log_id", log_id).maybe_single().execute()
    _raise_if_error(resp, "Failed to get log")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Log not found")
    return resp.data


@router.post("", response_model=AttendanceLogResponse)
def create_log(payload: AttendanceLogCreateRequest) -> Any:
    sb = get_supabase()
    data = payload.model_dump(exclude_none=True)

    resp = sb.table("attendance_logs").insert(data).execute()
    _raise_if_error(resp, "Failed to create log")
    if not resp.data:
        raise HTTPException(status_code=500, detail="Insert succeeded but returned no data")
    return resp.data[0]


@router.patch("/{log_id}", response_model=AttendanceLogResponse)
def update_log(log_id: int, payload: AttendanceLogUpdateRequest) -> Any:
    sb = get_supabase()
    data = payload.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(status_code=400, detail="No fields to update")

    resp = sb.table("attendance_logs").update(data).eq("log_id", log_id).execute()
    _raise_if_error(resp, "Failed to update log")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Log not found")
    return resp.data[0]


@router.delete("/{log_id}")
def delete_log(log_id: int) -> Any:
    sb = get_supabase()
    resp = sb.table("attendance_logs").delete().eq("log_id", log_id).execute()
    _raise_if_error(resp, "Failed to delete log")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Log not found or already deleted")
    return {"ok": True, "deleted": resp.data[0]}
