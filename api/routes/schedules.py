# api/routes/schedules.py
from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.common import execute_or_500, get_data, get_one_or_404
from api.supabase_client import get_supabase
from api.schemas import ScheduleCreateRequest, ScheduleUpdateRequest, ScheduleResponse

router = APIRouter(prefix="/schedules", tags=["schedules"])


@router.get("", response_model=List[ScheduleResponse])
def list_schedules(
    employee_id: Optional[int] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    order_desc: bool = Query(default=True),
) -> Any:
    sb = get_supabase()

    def _run():
        q = sb.table("schedules").select("*").limit(limit)
        if employee_id is not None:
            q = q.eq("employee_id", employee_id)
        q = q.order("start_time", desc=order_desc)
        return q.execute()

    resp = execute_or_500(_run, "list schedules")
    return get_data(resp)


@router.get("/{schedule_id}", response_model=ScheduleResponse)
def get_schedule(schedule_id: int) -> Any:
    sb = get_supabase()
    resp = execute_or_500(
        lambda: sb.table("schedules").select("*").eq("schedule_id", schedule_id).maybe_single().execute(),
        "get schedule",
    )
    return get_one_or_404(resp, "Schedule not found")


@router.post("", response_model=ScheduleResponse)
def create_schedule(body: ScheduleCreateRequest) -> Any:
    sb = get_supabase()
    payload = body.model_dump(exclude_none=True)

    resp = execute_or_500(lambda: sb.table("schedules").insert(payload).execute(), "create schedule")
    return get_one_or_404(resp, "Insert failed (no row returned)")


@router.patch("/{schedule_id}", response_model=ScheduleResponse)
def update_schedule(schedule_id: int, body: ScheduleUpdateRequest) -> Any:
    sb = get_supabase()
    payload = body.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No fields to update")

    resp = execute_or_500(
        lambda: sb.table("schedules").update(payload).eq("schedule_id", schedule_id).execute(),
        "update schedule",
    )
    return get_one_or_404(resp, "Schedule not found")


@router.delete("/{schedule_id}")
def delete_schedule(schedule_id: int) -> Any:
    sb = get_supabase()
    resp = execute_or_500(
        lambda: sb.table("schedules").delete().eq("schedule_id", schedule_id).execute(),
        "delete schedule",
    )
    if not get_data(resp):
        raise HTTPException(status_code=404, detail="Schedule not found or already deleted")
    return {"ok": True}
