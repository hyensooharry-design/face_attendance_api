from __future__ import annotations

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.supabase_client import get_supabase

router = APIRouter(prefix="/employees", tags=["employees"])


# -----------------------------
# Schemas
# -----------------------------
class EmployeeCreateRequest(BaseModel):
    employee_code: Optional[str] = Field(
        default=None, description="Student/Employee visible code (e.g., 20261234)"
    )
    name: str = Field(..., min_length=1)
    is_active: bool = True


class EmployeeResponse(BaseModel):
    employee_id: int
    employee_code: Optional[str] = None
    name: str
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    has_face: Optional[bool] = None  # convenience


# -----------------------------
# Helpers
# -----------------------------
def _raise_if_error(resp: Any, msg: str) -> None:
    # supabase-py typically returns object with .data and .error
    err = getattr(resp, "error", None)
    if err:
        raise HTTPException(status_code=500, detail=f"{msg}: {err}")


# -----------------------------
# Endpoints
# -----------------------------
@router.post("", response_model=EmployeeResponse)
def create_employee(payload: EmployeeCreateRequest) -> EmployeeResponse:
    sb = get_supabase()
    data = {
        "employee_code": payload.employee_code,
        "name": payload.name,
        "is_active": payload.is_active,
    }

    resp = sb.table("employees").insert(data).execute()
    _raise_if_error(resp, "Failed to create employee")
    if not resp.data:
        raise HTTPException(status_code=500, detail="Failed to create employee: empty response")

    row = resp.data[0]
    return EmployeeResponse(**row, has_face=None)


@router.get("", response_model=List[EmployeeResponse])
def list_employees(query: Optional[str] = None, limit: int = 100) -> List[EmployeeResponse]:
    """
    Minimal list:
    - filter by name or employee_code (ilike)
    - best-effort include has_face if FK relationship exists in Supabase
    """
    sb = get_supabase()

    q = sb.table("employees").select(
        "employee_id, employee_code, name, is_active, created_at, updated_at, face_embeddings(employee_id)"
    ).limit(limit)

    # filter
    if query:
        # Supabase "or" syntax: or=(name.ilike.*q*,employee_code.ilike.*q*)
        like = f"%{query}%"
        q = q.or_(f"name.ilike.{like},employee_code.ilike.{like}")

    resp = q.execute()

    # If relationship select fails (common when FK not configured), fallback to plain select
    err = getattr(resp, "error", None)
    if err:
        resp2 = sb.table("employees").select(
            "employee_id, employee_code, name, is_active, created_at, updated_at"
        ).limit(limit).execute()
        _raise_if_error(resp2, "Failed to list employees")
        rows = resp2.data or []
        return [EmployeeResponse(**r, has_face=None) for r in rows]

    rows = resp.data or []
    out: List[EmployeeResponse] = []
    for r in rows:
        has_face = False
        fe = r.get("face_embeddings")
        # Depending on Supabase response shape, fe can be dict or list or None
        if isinstance(fe, dict) and fe.get("employee_id") is not None:
            has_face = True
        elif isinstance(fe, list) and len(fe) > 0:
            has_face = True

        out.append(
            EmployeeResponse(
                employee_id=int(r["employee_id"]),
                employee_code=r.get("employee_code"),
                name=r.get("name", ""),
                is_active=bool(r.get("is_active", True)),
                created_at=r.get("created_at"),
                updated_at=r.get("updated_at"),
                has_face=has_face,
            )
        )
    return out


@router.get("/{employee_id}", response_model=EmployeeResponse)
def get_employee(employee_id: int) -> EmployeeResponse:
    sb = get_supabase()

    resp = (
        sb.table("employees")
        .select("employee_id, employee_code, name, is_active, created_at, updated_at, face_embeddings(employee_id)")
        .eq("employee_id", employee_id)
        .limit(1)
        .execute()
    )

    err = getattr(resp, "error", None)
    if err:
        # Fallback to plain employees row
        resp2 = (
            sb.table("employees")
            .select("employee_id, employee_code, name, is_active, created_at, updated_at")
            .eq("employee_id", employee_id)
            .limit(1)
            .execute()
        )
        _raise_if_error(resp2, "Failed to fetch employee")
        if not resp2.data:
            raise HTTPException(status_code=404, detail="Employee not found")
        return EmployeeResponse(**resp2.data[0], has_face=None)

    if not resp.data:
        raise HTTPException(status_code=404, detail="Employee not found")

    r = resp.data[0]
    fe = r.get("face_embeddings")
    has_face = False
    if isinstance(fe, dict) and fe.get("employee_id") is not None:
        has_face = True
    elif isinstance(fe, list) and len(fe) > 0:
        has_face = True

    return EmployeeResponse(
        employee_id=int(r["employee_id"]),
        employee_code=r.get("employee_code"),
        name=r.get("name", ""),
        is_active=bool(r.get("is_active", True)),
        created_at=r.get("created_at"),
        updated_at=r.get("updated_at"),
        has_face=has_face,
    )
