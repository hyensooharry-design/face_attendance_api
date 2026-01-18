# api/routes/faces.py
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from api.common import execute_or_500, get_data
from api.embedding import get_embedding_from_image_bytes
from api.supabase_client import get_supabase

router = APIRouter(prefix="/faces", tags=["faces"])


def vec_to_pgvector_str(v: np.ndarray) -> str:
    v = v.astype(np.float32).tolist()
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def _ensure_person(employee_id: int) -> str:
    """Ensure persons row exists for employee_id and return persons.id (UUID string)."""
    sb = get_supabase()

    # 1) try existing person
    resp = execute_or_500(
        lambda: sb.table("persons").select("id, employee_id").eq("employee_id", employee_id).maybe_single().execute(),
        "get person",
    )
    rows = get_data(resp)
    if rows:
        return str(rows[0]["id"])

    # 2) create person (optionally attach name from employees)
    emp_name: Optional[str] = None
    try:
        emp = execute_or_500(
            lambda: sb.table("employees").select("name").eq("employee_id", employee_id).maybe_single().execute(),
            "fetch employee name",
        )
        emp_rows = get_data(emp)
        if emp_rows:
            emp_name = emp_rows[0].get("name")
    except Exception:
        emp_name = None

    insert_payload: dict = {"employee_id": employee_id}
    if emp_name:
        insert_payload["name"] = emp_name

    created = execute_or_500(
        lambda: sb.table("persons").insert(insert_payload).execute(),
        "create person",
    )
    created_rows = get_data(created)
    if not created_rows:
        raise HTTPException(status_code=500, detail="Failed to create person")
    return str(created_rows[0]["id"])


@router.get("")
def list_faces(limit: int = 200) -> List[Any]:
    sb = get_supabase()
    resp = execute_or_500(
        # Render schema: face_embeddings references persons(person_id)
        lambda: sb.table("face_embeddings")
        .select("id, person_id, model_name, model_version, created_at, persons(employee_id,name)")
        .limit(limit)
        .execute(),
        "list faces",
    )
    return get_data(resp)


@router.get("/{employee_id}")
def get_face(employee_id: int) -> Any:
    sb = get_supabase()
    # Resolve to person
    p = execute_or_500(
        lambda: sb.table("persons").select("id, employee_id, name").eq("employee_id", employee_id).maybe_single().execute(),
        "get person",
    )
    p_rows = get_data(p)
    if not p_rows:
        raise HTTPException(status_code=404, detail="Person not found for employee")
    person_id = p_rows[0]["id"]

    resp = execute_or_500(
        lambda: sb.table("face_embeddings")
        .select("id, person_id, model_name, model_version, created_at")
        .eq("person_id", person_id)
        .limit(200)
        .execute(),
        "get face_embeddings by person",
    )
    emb_rows = get_data(resp)
    if not emb_rows:
        raise HTTPException(status_code=404, detail="No face embeddings found for employee")
    return {"person": p_rows[0], "embeddings": emb_rows}


@router.post("/enroll/{employee_id}")
async def enroll_face(
    employee_id: int,
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(default="arcface"),
    model_version: Optional[str] = Form(default="onnx"),
) -> Any:
    img = await file.read()
    if not img:
        raise HTTPException(status_code=400, detail="Empty image")

    emb = get_embedding_from_image_bytes(img)
    emb_str = vec_to_pgvector_str(emb)

    sb = get_supabase()
    person_id = _ensure_person(employee_id)

    # Keep 1 active embedding per person by default (delete-then-insert)
    execute_or_500(
        lambda: sb.table("face_embeddings").delete().eq("person_id", person_id).execute(),
        "delete old embeddings",
    )

    payload = {
        "person_id": person_id,
        "model_name": model_name,
        "model_version": model_version,
        "embedding": emb_str,
    }

    execute_or_500(
        lambda: sb.table("face_embeddings").insert(payload).execute(),
        "enroll face (insert face_embeddings)",
    )
    return {"ok": True, "employee_id": employee_id, "person_id": person_id}


@router.delete("/{employee_id}")
def delete_face(employee_id: int) -> Any:
    sb = get_supabase()
    p = execute_or_500(
        lambda: sb.table("persons").select("id").eq("employee_id", employee_id).maybe_single().execute(),
        "get person",
    )
    p_rows = get_data(p)
    if not p_rows:
        raise HTTPException(status_code=404, detail="Person not found")

    person_id = p_rows[0]["id"]
    resp = execute_or_500(
        lambda: sb.table("face_embeddings").delete().eq("person_id", person_id).execute(),
        "delete face embeddings",
    )
    if not get_data(resp):
        raise HTTPException(status_code=404, detail="No embeddings found or already deleted")
    return {"ok": True, "employee_id": employee_id}
