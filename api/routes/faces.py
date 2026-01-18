from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2

from api.supabase_client import get_supabase
from api.models.face_models import detect_faces, get_embedding, safe_crop

router = APIRouter(prefix="/faces", tags=["faces"])


def vec_to_pg(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v.tolist()) + "]"


@router.post("/enroll/{employee_id}")
async def enroll_face(employee_id: int, file: UploadFile = File(...)):
    img_bytes = await file.read()
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Invalid image")

    faces = detect_faces(frame)
    if not faces:
        raise HTTPException(400, "No face detected")

    faces.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    face = safe_crop(frame, faces[0])

    emb = get_embedding(face)
    emb = emb / np.linalg.norm(emb)

    sb = get_supabase()

    # 🔑 ENSURE PERSON EXISTS (employee_id is text in DB)
    emp_id_str = str(employee_id)
    p = sb.table("persons").select("id").eq("employee_id", emp_id_str).execute().data
    if p:
        person_id = p[0]["id"]
    else:
        # Fetch name from employees table
        e = sb.table("employees").select("name").eq("employee_id", employee_id).execute().data
        emp_name = e[0]["name"] if e else "Unknown"
        
        person = sb.table("persons").insert({
            "employee_id": emp_id_str,
            "name": emp_name
        }).execute().data
        person_id = person[0]["id"]

    sb.table("face_embeddings").insert({
        "person_id": person_id,
        "embedding": vec_to_pg(emb),
    }).execute()

    # refresh cache
    try:
        from api.routes.recognize import refresh_embeddings
        refresh_embeddings()
    except Exception:
        pass

    return {"ok": True, "person_id": person_id}

@router.post("/check-duplicate")
async def check_duplicate(image: UploadFile = File(...)):
    # Import KNOWN cache
    from api.routes.recognize import KNOWN, refresh_embeddings
    if not KNOWN:
        refresh_embeddings()

    img_bytes = await image.read()
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"duplicate": False}

    faces = detect_faces(frame)
    if not faces:
        return {"duplicate": False}

    faces.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    face = safe_crop(frame, faces[0])
    emb = get_embedding(face)
    emb = emb / np.linalg.norm(emb)

    for eid, data in KNOWN.items():
        ref = data["vec"]
        s = float(np.dot(emb, ref))
        if s > 0.65: # High threshold for duplicates
            return {
                "duplicate": True, 
                "employee_id": eid,
                "name": data.get("name"),
                "employee_code": data.get("code")
            }

    return {"duplicate": False}


@router.delete("/{employee_id}")
async def delete_face(employee_id: int):
    sb = get_supabase()
    
    # We delete from 'persons' (employee_id is text)
    sb.table("persons").delete().eq("employee_id", str(employee_id)).execute()
    
    # refresh cache
    try:
        from api.routes.recognize import refresh_embeddings
        refresh_embeddings()
    except Exception:
        pass
        
    return {"ok": True}
