# api/routes/employees.py
from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from api.supabase_client import get_supabase
from api.common import execute_or_500, get_data, get_one_or_404
from api.schemas import EmployeeCreateRequest, EmployeeUpdateRequest, EmployeeResponse

router = APIRouter(prefix="/employees", tags=["employees"])


@router.get("", response_model=List[EmployeeResponse])
def list_employees(
    # UI/api_client.py가 query= 로 보냄 :contentReference[oaicite:2]{index=2}
    query: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    is_active: Optional[bool] = Query(default=None),
) -> Any:
    sb = get_supabase()

    def _run():
        q = sb.table("employees").select("*").limit(limit)
        if is_active is not None:
            q = q.eq("is_active", is_active)
        if query:
            q = q.or_(f"name.ilike.%{query}%,employee_code.ilike.%{query}%")
        return q.execute()

    resp = execute_or_500(_run, "list employees")
    rows = get_data(resp)

    # has_face 붙이기
    # Render schema: employees(employee_id) -> persons(employee_id) -> face_embeddings(person_id)
    emp_ids = [r.get("employee_id") for r in rows if r.get("employee_id") is not None]
    face_emp_set = set()

    if emp_ids:
        # 1) map employees -> person ids
        persons_resp = execute_or_500(
            lambda: sb.table("persons").select("id, employee_id").in_("employee_id", emp_ids).execute(),
            "list persons for has_face",
        )
        person_rows = get_data(persons_resp)
        emp_by_person = {pr["id"]: pr.get("employee_id") for pr in person_rows if pr.get("id")}
        person_ids = list(emp_by_person.keys())

        # 2) check which persons have embeddings
        if person_ids:
            emb_resp = execute_or_500(
                lambda: sb.table("face_embeddings").select("person_id").in_("person_id", person_ids).execute(),
                "list face_embeddings for has_face",
            )
            for er in get_data(emb_resp):
                pid = er.get("person_id")
                if pid in emp_by_person and emp_by_person[pid] is not None:
                    face_emp_set.add(emp_by_person[pid])

    for r in rows:
        r["has_face"] = r.get("employee_id") in face_emp_set

    return rows


@router.get("/{employee_id}", response_model=EmployeeResponse)
def get_employee(employee_id: int) -> Any:
    sb = get_supabase()
    resp = execute_or_500(
        lambda: sb.table("employees").select("*").eq("employee_id", employee_id).maybe_single().execute(),
        "get employee",
    )
    row = get_one_or_404(resp, "Employee not found")

    # has_face: persons -> face_embeddings
    p = execute_or_500(
        lambda: sb.table("persons").select("id").eq("employee_id", employee_id).maybe_single().execute(),
        "get person for has_face",
    )
    p_rows = get_data(p)
    if not p_rows:
        row["has_face"] = False
        return row

    person_id = p_rows[0].get("id")
    face_resp = execute_or_500(
        lambda: sb.table("face_embeddings").select("id").eq("person_id", person_id).limit(1).execute(),
        "get face_embeddings",
    )
    row["has_face"] = bool(get_data(face_resp))
    return row


@router.post("", response_model=EmployeeResponse)
def create_employee(body: EmployeeCreateRequest) -> Any:
    sb = get_supabase()
    payload = body.model_dump(exclude_none=True)

    resp = execute_or_500(lambda: sb.table("employees").insert(payload).execute(), "create employee")
    row = get_one_or_404(resp, "Insert failed (no row returned)")

    # 새로 만든 직원은 기본적으로 face 없음
    row["has_face"] = False
    return row


@router.patch("/{employee_id}", response_model=EmployeeResponse)
def update_employee(employee_id: int, body: EmployeeUpdateRequest) -> Any:
    sb = get_supabase()
    payload = body.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No fields to update")

    resp = execute_or_500(
        lambda: sb.table("employees").update(payload).eq("employee_id", employee_id).execute(),
        "update employee",
    )
    row = get_one_or_404(resp, "Employee not found")

    p = execute_or_500(
        lambda: sb.table("persons").select("id").eq("employee_id", employee_id).maybe_single().execute(),
        "get person for has_face",
    )
    p_rows = get_data(p)
    if not p_rows:
        row["has_face"] = False
        return row

    person_id = p_rows[0].get("id")
    face_resp = execute_or_500(
        lambda: sb.table("face_embeddings").select("id").eq("person_id", person_id).limit(1).execute(),
        "get face_embeddings",
    )
    row["has_face"] = bool(get_data(face_resp))
    return row


@router.delete("/{employee_id}")
def delete_employee(employee_id: int) -> Any:
    sb = get_supabase()
    resp = execute_or_500(
        lambda: sb.table("employees").delete().eq("employee_id", employee_id).execute(),
        "delete employee",
    )
    deleted = get_data(resp)
    if not deleted:
        raise HTTPException(status_code=404, detail="Employee not found or already deleted")
    return {"ok": True}


# UI 호환 엔드포인트 (api_client.py가 대체 경로로도 시도함)
@router.post("/{employee_id}/enroll-face")
async def enroll_face_compat(employee_id: int, file: UploadFile = File(...)) -> Any:
    from api.routes.faces import enroll_face
    return await enroll_face(employee_id, file)
