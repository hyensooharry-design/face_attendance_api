from __future__ import annotations

from typing import Optional, List, Any, Dict
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.supabase_client import get_supabase

router = APIRouter(prefix="/logs", tags=["logs"])


class AttendanceLogResponse(BaseModel):
    log_id: Optional[int] = None
    event_time: Optional[str] = None
    event_type: Optional[str] = None
    camera_id: Optional[str] = None
    recognized: Optional[bool] = None
    similarity: Optional[float] = None
    employee_id: Optional[int] = None
    name: Optional[str] = None  # joined from employees


def _raise_if_error(resp: Any, msg: str) -> None:
    err = getattr(resp, "error", None)
    if err:
        raise HTTPException(status_code=500, detail=f"{msg}: {err}")


@router.get("", response_model=List[AttendanceLogResponse])
def list_logs(
    limit: int = Query(default=200, ge=1, le=1000),
    employee_id: Optional[int] = None,
    camera_id: Optional[str] = None,
    recognized: Optional[bool] = None,
    from_time: Optional[str] = Query(default=None, description="ISO8601 timestamptz lower bound"),
    to_time: Optional[str] = Query(default=None, description="ISO8601 timestamptz upper bound"),
) -> List[AttendanceLogResponse]:
    sb = get_supabase()

    # Try join select (requires relationship between attendance_logs.employee_id -> employees.employee_id)
    q = sb.table("attendance_logs").select(
        "log_id, event_time, event_type, camera_id, recognized, similarity, employee_id, employees(name)"
    ).order("event_time", desc=True).limit(limit)

    if employee_id is not None:
        q = q.eq("employee_id", employee_id)
    if camera_id is not None:
        q = q.eq("camera_id", camera_id)
    if recognized is not None:
        q = q.eq("recognized", recognized)
    if from_time:
        q = q.gte("event_time", from_time)
    if to_time:
        q = q.lte("event_time", to_time)

    resp = q.execute()

    # If join fails, fallback without join
    err = getattr(resp, "error", None)
    if err:
        q2 = sb.table("attendance_logs").select(
            "log_id, event_time, event_type, camera_id, recognized, similarity, employee_id"
        ).order("event_time", desc=True).limit(limit)

        if employee_id is not None:
            q2 = q2.eq("employee_id", employee_id)
        if camera_id is not None:
            q2 = q2.eq("camera_id", camera_id)
        if recognized is not None:
            q2 = q2.eq("recognized", recognized)
        if from_time:
            q2 = q2.gte("event_time", from_time)
        if to_time:
            q2 = q2.lte("event_time", to_time)

        resp2 = q2.execute()
        _raise_if_error(resp2, "Failed to list logs")
        return [AttendanceLogResponse(**r) for r in (resp2.data or [])]

    rows = resp.data or []
    out: List[AttendanceLogResponse] = []
    for r in rows:
        name = None
        emp = r.get("employees")
        if isinstance(emp, dict):
            name = emp.get("name")
        out.append(
            AttendanceLogResponse(
                log_id=r.get("log_id"),
                event_time=r.get("event_time"),
                event_type=r.get("event_type"),
                camera_id=r.get("camera_id"),
                recognized=r.get("recognized"),
                similarity=r.get("similarity"),
                employee_id=r.get("employee_id"),
                name=name,
            )
        )
    return out
