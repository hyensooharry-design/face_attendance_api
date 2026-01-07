import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from datetime import datetime, timezone

from api.supabase_client import get_supabase
from api.schemas import RecognizeResponse
from api.embedding import get_embedding_from_image_bytes

load_dotenv()

app = FastAPI(title="Face Attendance API")

def vec_to_pgvector_str(v: np.ndarray) -> str:
    v = v.astype(np.float32).tolist()
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(file: UploadFile = File(...)):
    img_bytes = await file.read()

    # 1) 임베딩 생성 (512,)
    emb = get_embedding_from_image_bytes(img_bytes)
    if emb.ndim != 1:
        raise HTTPException(status_code=500, detail="Embedding must be 1D vector")
    if emb.shape[0] != 512:
        raise HTTPException(status_code=500, detail=f"Embedding dim mismatch: got {emb.shape[0]}, expected 512")

    supabase = get_supabase()

    # 2) DB에서 Top-1 매칭 (RPC 호출)
    min_sim = float(os.getenv("MIN_SIMILARITY", "0.45"))
    query_vec = vec_to_pgvector_str(emb)

    match = (
        supabase.rpc("match_face", {"query_embedding": query_vec, "min_similarity": min_sim})
        .execute()
    )

    if not match.data:
        # 등록자 0명 같은 경우
        return RecognizeResponse(employee_id=None, similarity=None, recognized=False)

    row = match.data[0]
    employee_id = row["employee_id"]
    similarity = float(row["similarity"])
    recognized = bool(row["recognized"])

    # 3) 출석 로그 기록 (recognized 여부/유사도 포함)
    log_payload = {
        "employee_id": employee_id,
        "camera_id": os.environ["CAMERA_ID"],
        "event_type": "CHECK_IN",
        "similarity": similarity,
        "recognized": recognized,
        "event_time": datetime.now(timezone.utc).isoformat(),  # ✅ 추가
    }   
    supabase.table("attendance_logs").insert(log_payload).execute()

    return RecognizeResponse(employee_id=employee_id if recognized else None,
                             similarity=similarity,
                             recognized=recognized)
