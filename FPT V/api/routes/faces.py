from __future__ import annotations

import os
import numpy as np
from typing import Optional, Any

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from api.supabase_client import get_supabase
from api.embedding import get_embedding_from_image_bytes


router = APIRouter(prefix="/employees", tags=["faces"])


# -----------------------------
# Config
# -----------------------------
DEFAULT_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "facenet-pytorch")
DEFAULT_MODEL_VERSION = os.getenv("FACE_MODEL_VERSION", "")


def vec_to_pgvector_str(v: np.ndarray) -> str:
    """
    Supabase pgvector insert typically accepts:
    - a string like "[0.1,0.2,...]"
    - OR sometimes a python list (depends on client/DB adapter)

    We’ll use string for maximum compatibility.
    """
    v = v.astype(np.float32).tolist()
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def _raise_if_error(resp: Any, msg: str) -> None:
    err = getattr(resp, "error", None)
    if err:
        raise HTTPException(status_code=500, detail=f"{msg}: {err}")


class FaceEnrollResponse(BaseModel):
    ok: bool
    employee_id: int
    embedding_dim: int
    model_name: str
    model_version: str


@router.post("/{employee_id}/faces", response_model=FaceEnrollResponse)
async def enroll_face(employee_id: int, file: UploadFile = File(...)) -> FaceEnrollResponse:
    """
    Enroll a face for an employee:
    - upload image
    - create 512-d embedding
    - upsert into face_embeddings (PK = employee_id)
    """
    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    emb = get_embedding_from_image_bytes(img_bytes)  # expected np.ndarray shape (512,)
    if emb is None:
        raise HTTPException(status_code=400, detail="Failed to extract embedding (no face?)")

    emb = np.asarray(emb).reshape(-1).astype(np.float32)
    if emb.ndim != 1:
        raise HTTPException(status_code=500, detail="Embedding must be 1D vector")
    if emb.shape[0] != 512:
        raise HTTPException(status_code=500, detail=f"Embedding dim mismatch: {emb.shape[0]} != 512")

    sb = get_supabase()

    # Optional: ensure employee exists
    emp_resp = sb.table("employees").select("employee_id").eq("employee_id", employee_id).limit(1).execute()
    _raise_if_error(emp_resp, "Failed to check employee")
    if not emp_resp.data:
        raise HTTPException(status_code=404, detail="Employee not found")

    payload = {
        "employee_id": employee_id,
        "embedding_dim": 512,
        "model_name": DEFAULT_MODEL_NAME,
        "model_version": DEFAULT_MODEL_VERSION,
        # IMPORTANT: your Supabase UI shows embedding type is vector
        "embedding": vec_to_pgvector_str(emb),
    }

    # PK is employee_id => upsert
    resp = sb.table("face_embeddings").upsert(payload, on_conflict="employee_id").execute()
    _raise_if_error(resp, "Failed to upsert face embedding")

    return FaceEnrollResponse(
        ok=True,
        employee_id=employee_id,
        embedding_dim=512,
        model_name=DEFAULT_MODEL_NAME,
        model_version=DEFAULT_MODEL_VERSION,
    )
