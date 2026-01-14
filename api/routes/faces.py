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


@router.get("")
def list_faces(limit: int = 200) -> List[Any]:
    sb = get_supabase()
    resp = execute_or_500(
        lambda: sb.table("face_embeddings").select("employee_id,embedding_dim,model_name,model_version").limit(limit).execute(),
        "list faces",
    )
    return get_data(resp)


@router.get("/{employee_id}")
def get_face(employee_id: int) -> Any:
    sb = get_supabase()
    resp = execute_or_500(
        lambda: sb.table("face_embeddings").select("*").eq("employee_id", employee_id).maybe_single().execute(),
        "get face",
    )
    rows = get_data(resp)
    if not rows:
        raise HTTPException(status_code=404, detail="Face not found")
    return rows[0]


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
    payload = {
        "employee_id": employee_id,
        "embedding_dim": int(getattr(emb, "shape", [512])[0]) if hasattr(emb, "shape") else 512,
        "model_name": model_name,
        "model_version": model_version,
        "embedding": emb_str,
    }

    execute_or_500(
        lambda: sb.table("face_embeddings").upsert(payload, on_conflict="employee_id").execute(),
        "enroll face (upsert face_embeddings)",
    )
    return {"ok": True, "employee_id": employee_id}


@router.delete("/{employee_id}")
def delete_face(employee_id: int) -> Any:
    sb = get_supabase()
    resp = execute_or_500(
        lambda: sb.table("face_embeddings").delete().eq("employee_id", employee_id).execute(),
        "delete face",
    )
    if not get_data(resp):
        raise HTTPException(status_code=404, detail="Face not found or already deleted")
    return {"ok": True}
