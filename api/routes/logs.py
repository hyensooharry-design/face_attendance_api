# api/routes/logs.py
from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.common import execute_or_500, get_data, get_one_or_404
from api.supabase_client import get_supabase
from api.schemas import (
    AttendanceLogCreateRequest,
    AttendanceLogUpdateRequest,
    AttendanceLogResponse,
)

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("", response_model=List[AttendanceLogResponse])
def list_logs(
    limit: int = Query(default=200, ge=1, le=2000),
    employee_id: Optional[int] = Query(default=None),
    camera_id: Optional[str] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    recognized: Optional[bool] = Query(default=None),
    order_desc: bool = Query(default=True),
) -> Any:
    sb = get_supabase()

    def _run():
        q = sb.table("attendance_logs").select("*").limit(limit)
        if employee_id is not None:
            q = q.eq("employee_id", employee_id)
        if camera_id is not None:
            q = q.eq("camera_id", camera_id)
        if event_type is not None:
            q = q.eq("event_type", event_type)
        if recognized is not None:
            q = q.eq("recognized", recognized)
        q = q.order("event_time", desc=order_desc)
        return q.execute()

    resp = execute_or_500(_run, "list logs")
    return get_data(resp)


@router.get("/{log_id}", response_model=AttendanceLogResponse)
def get_log(log_id: int) -> Any:
    sb = get_supabase()
    resp = execute_or_500(
        lambda: sb.table("attendance_logs").select("*").eq("log_id", log_id).maybe_single().execute(),
        "get log",
    )
    return get_one_or_404(resp, "Log not found")


@router.post("", response_model=AttendanceLogResponse)
def create_log(body: AttendanceLogCreateRequest) -> Any:
    sb = get_supabase()
    payload = body.model_dump(exclude_none=True)

    resp = execute_or_500(lambda: sb.table("attendance_logs").insert(payload).execute(), "create log")
    return get_one_or_404(resp, "Insert failed (no row returned)")


@router.patch("/{log_id}", response_model=AttendanceLogResponse)
def update_log(log_id: int, body: AttendanceLogUpdateRequest) -> Any:
    sb = get_supabase()
    payload = body.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No fields to update")

    resp = execute_or_500(
        lambda: sb.table("attendance_logs").update(payload).eq("log_id", log_id).execute(),
        "update log",
    )
    return get_one_or_404(resp, "Log not found")


@router.delete("/{log_id}")
def delete_log(log_id: int) -> Any:
    sb = get_supabase()
    resp = execute_or_500(
        lambda: sb.table("attendance_logs").delete().eq("log_id", log_id).execute(),
        "delete log",
    )
    if not get_data(resp):
        raise HTTPException(status_code=404, detail="Log not found or already deleted")
    return {"ok": True}
