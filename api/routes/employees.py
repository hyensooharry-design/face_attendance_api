from __future__ import annotations

from typing import Any, Optional, List

from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from api.supabase_client import get_supabase
from api.schemas import EmployeeCreateRequest, EmployeeUpdateRequest, EmployeeResponse

router = APIRouter(prefix="/employees", tags=["employees"])


def _raise_if_error(resp: Any, msg: str) -> None:
    err = getattr(resp, "error", None)
    if err:
        raise HTTPException(status_code=500, detail=f"{msg}: {err}")


@router.get("", response_model=List[EmployeeResponse])
def list_employees(
    q: Optional[str] = Query(default=None, description="Search by name or employee_code (contains)"),
    is_active: Optional[bool] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
) -> Any:
    sb = get_supabase()
    query = sb.table("employees").select("*").limit(limit)

    if is_active is not None:
        query = query.eq("is_active", is_active)

    # NOTE: supabase-py doesn't have OR contains in a super clean way for all backends.
    # We'll do a simple "ilike" filter if q exists (name or employee_code).
    if q:
        # PostgREST "or" syntax
        query = query.or_(f"name.ilike.%{q}%,employee_code.ilike.%{q}%")

    resp = query.execute()
    _raise_if_error(resp, "Failed to list employees")
    return resp.data or []


@router.get("/{employee_id}", response_model=EmployeeResponse)
def get_employee(employee_id: int) -> Any:
    sb = get_supabase()
    resp = sb.table("employees").select("*").eq("employee_id", employee_id).maybe_single().execute()
    _raise_if_error(resp, "Failed to get employee")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Employee not found")
    return resp.data


@router.post("", response_model=EmployeeResponse)
def create_employee(payload: EmployeeCreateRequest) -> Any:
    sb = get_supabase()
    resp = sb.table("employees").insert(payload.model_dump(exclude_none=True)).execute()
    _raise_if_error(resp, "Failed to create employee")
    if not resp.data:
        raise HTTPException(status_code=500, detail="Insert succeeded but returned no data")
    return resp.data[0]


@router.patch("/{employee_id}", response_model=EmployeeResponse)
def update_employee(employee_id: int, payload: EmployeeUpdateRequest) -> Any:
    sb = get_supabase()
    data = payload.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(status_code=400, detail="No fields to update")

    resp = sb.table("employees").update(data).eq("employee_id", employee_id).execute()
    _raise_if_error(resp, "Failed to update employee")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Employee not found")
    return resp.data[0]


@router.delete("/{employee_id}")
def delete_employee(employee_id: int) -> Any:
    """
    Hard delete.
    If you prefer safer approach: use PATCH is_active=false in UI instead.
    """
    sb = get_supabase()
    resp = sb.table("employees").delete().eq("employee_id", employee_id).execute()
    _raise_if_error(resp, "Failed to delete employee")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Employee not found or already deleted")
    return {"ok": True, "deleted": resp.data[0]}


# -----------------------------
# Compatibility endpoint (UI)
# -----------------------------
@router.post("/{employee_id}/enroll-face")
async def enroll_face_compat(employee_id: int, file: UploadFile = File(...)) -> Any:
    """
    UI compatibility:
    /employees/{id}/enroll-face  -> forwards to /faces/enroll/{id}
    (avoid circular import by local import)
    """
    from api.routes.faces import enroll_face  # local import
    return await enroll_face(employee_id, file)
