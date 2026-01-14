from __future__ import annotations

from typing import Any, Optional, List

from fastapi import APIRouter, HTTPException, Query

from api.supabase_client import get_supabase
from api.schemas import ScheduleCreateRequest, ScheduleUpdateRequest, ScheduleResponse

router = APIRouter(prefix="/schedules", tags=["schedules"])


def _raise_if_error(resp: Any, msg: str) -> None:
    err = getattr(resp, "error", None)
    if err:
        raise HTTPException(status_code=500, detail=f"{msg}: {err}")


@router.get("", response_model=List[ScheduleResponse])
def list_schedules(
    employee_id: Optional[int] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    order_desc: bool = Query(default=True),
) -> Any:
    sb = get_supabase()
    query = sb.table("schedules").select("*").limit(limit)

    if employee_id is not None:
        query = query.eq("employee_id", employee_id)

    query = query.order("start_time", desc=order_desc)

    resp = query.execute()
    _raise_if_error(resp, "Failed to list schedules")
    return resp.data or []


@router.get("/{schedule_id}", response_model=ScheduleResponse)
def get_schedule(schedule_id: int) -> Any:
    sb = get_supabase()
    resp = sb.table("schedules").select("*").eq("schedule_id", schedule_id).maybe_single().execute()
    _raise_if_error(resp, "Failed to get schedule")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return resp.data


@router.post("", response_model=ScheduleResponse)
def create_schedule(payload: ScheduleCreateRequest) -> Any:
    sb = get_supabase()
    resp = sb.table("schedules").insert(payload.model_dump(exclude_none=True)).execute()
    _raise_if_error(resp, "Failed to create schedule")
    if not resp.data:
        raise HTTPException(status_code=500, detail="Insert succeeded but returned no data")
    return resp.data[0]


@router.patch("/{schedule_id}", response_model=ScheduleResponse)
def update_schedule(schedule_id: int, payload: ScheduleUpdateRequest) -> Any:
    sb = get_supabase()
    data = payload.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(status_code=400, detail="No fields to update")

    resp = sb.table("schedules").update(data).eq("schedule_id", schedule_id).execute()
    _raise_if_error(resp, "Failed to update schedule")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return resp.data[0]


@router.delete("/{schedule_id}")
def delete_schedule(schedule_id: int) -> Any:
    sb = get_supabase()
    resp = sb.table("schedules").delete().eq("schedule_id", schedule_id).execute()
    _raise_if_error(resp, "Failed to delete schedule")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Schedule not found or already deleted")
    return {"ok": True, "deleted": resp.data[0]}
