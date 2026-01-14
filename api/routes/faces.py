from __future__ import annotations

from typing import Any, Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from api.supabase_client import get_supabase
from api.schemas import FaceEmbeddingUpsertRequest, FaceEmbeddingResponse
from api.embedding import get_embedding_from_image_bytes

router = APIRouter(prefix="/faces", tags=["faces"])


def _raise_if_error(resp: Any, msg: str) -> None:
    err = getattr(resp, "error", None)
    if err:
        raise HTTPException(status_code=500, detail=f"{msg}: {err}")


@router.get("", response_model=List[FaceEmbeddingResponse])
def list_faces(limit: int = Query(default=200, ge=1, le=2000)) -> Any:
    sb = get_supabase()
    resp = sb.table("face_embeddings").select("*").limit(limit).execute()
    _raise_if_error(resp, "Failed to list face embeddings")
    return resp.data or []


@router.get("/{employee_id}", response_model=FaceEmbeddingResponse)
def get_face(employee_id: int) -> Any:
    sb = get_supabase()
    resp = sb.table("face_embeddings").select("*").eq("employee_id", employee_id).maybe_single().execute()
    _raise_if_error(resp, "Failed to get face embedding")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Face embedding not found")
    return resp.data


@router.put("/{employee_id}", response_model=FaceEmbeddingResponse)
def upsert_face(employee_id: int, payload: FaceEmbeddingUpsertRequest) -> Any:
    """
    Upsert face embedding row (employee_id is PK).
    """
    sb = get_supabase()
    body = payload.model_dump(exclude_none=True)
    body["employee_id"] = employee_id

    resp = sb.table("face_embeddings").upsert(body).execute()
    _raise_if_error(resp, "Failed to upsert face embedding")
    if not resp.data:
        raise HTTPException(status_code=500, detail="Upsert succeeded but returned no data")
    return resp.data[0]


@router.delete("/{employee_id}")
def delete_face(employee_id: int) -> Any:
    sb = get_supabase()
    resp = sb.table("face_embeddings").delete().eq("employee_id", employee_id).execute()
    _raise_if_error(resp, "Failed to delete face embedding")
    if not resp.data:
        raise HTTPException(status_code=404, detail="Face embedding not found or already deleted")
    return {"ok": True, "deleted": resp.data[0]}


@router.post("/enroll/{employee_id}")
async def enroll_face(employee_id: int, file: UploadFile = File(...)) -> Any:
    """
    Compute embedding from uploaded image, then upsert into face_embeddings.
    """
    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    emb = get_embedding_from_image_bytes(img_bytes)  # should return list[float] or numpy array

    sb = get_supabase()
    body = {
        "employee_id": employee_id,
        "embedding_dim": 512,
        "model_name": "arcface",
        "model_version": "onnx",
        "embedding": emb,
    }
    resp = sb.table("face_embeddings").upsert(body).execute()
    _raise_if_error(resp, "Failed to enroll face embedding")
    if not resp.data:
        raise HTTPException(status_code=500, detail="Enroll succeeded but returned no data")

    return {"ok": True, "employee_id": employee_id, "face_embedding": resp.data[0]}
