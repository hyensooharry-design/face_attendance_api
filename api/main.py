import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime, timezone

from api.supabase_client import get_supabase
from api.schemas import RecognizeResponse
from api.embedding import get_embedding_from_image_bytes
from api.routes import employees, faces, logs
from api.routes.cameras import router as cameras_router

app.include_router(cameras_router)
app.include_router(employees.router)
app.include_router(faces.router)
app.include_router(logs.router)

load_dotenv()

app = FastAPI(
    title="Face Attendance API",
    version="1.0.0",
)

# 필요 시 프론트/테스트에서 호출하려면 CORS 허용 (원치 않으면 삭제 가능)
# 기본은 * 로 열어둠. 운영 시에는 특정 도메인으로 좁히는 걸 권장.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def vec_to_pgvector_str(v: np.ndarray) -> str:
    v = v.astype(np.float32).tolist()
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"

# ✅ 루트 추가: Render 주소를 눌러도 Not Found 안 뜨게
@app.get("/")
def root():
    # 1) 간단 상태 반환
    return {
        "service": "Face Attendance API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "recognize": "/recognize",
    }
    # 2) 혹시 루트로 들어오면 docs로 보내고 싶으면 위 return 대신 아래 사용
    # return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(file: UploadFile = File(...)):
    # ---- 파일 검증 (선택) ----
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid content_type: {file.content_type}. Must be image/*")

    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # 1) 임베딩 생성 (512,)
    try:
        emb = get_embedding_from_image_bytes(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    if emb.ndim != 1:
        raise HTTPException(status_code=500, detail="Embedding must be 1D vector")
    if emb.shape[0] != 512:
        raise HTTPException(status_code=500, detail=f"Embedding dim mismatch: got {emb.shape[0]}, expected 512")

    # 환경변수 안전 체크
    camera_id = os.getenv("CAMERA_ID")
    if not camera_id:
        raise HTTPException(status_code=500, detail="Missing env var: CAMERA_ID")

    min_sim = float(os.getenv("MIN_SIMILARITY", "0.45"))

    supabase = get_supabase()

    # 2) DB에서 Top-1 매칭 (RPC 호출)
    query_vec = vec_to_pgvector_str(emb)

    try:
        match = supabase.rpc(
            "match_face",
            {"query_embedding": query_vec, "min_similarity": min_sim}
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase RPC failed: {str(e)}")

    if not match.data:
        # 등록자 0명 같은 경우
        return RecognizeResponse(employee_id=None, similarity=None, recognized=False)

    row = match.data[0]
    employee_id = row.get("employee_id")
    similarity = float(row.get("similarity", 0.0))
    recognized = bool(row.get("recognized", False))

    # 3) 출석 로그 기록 (recognized 여부/유사도 포함)
    log_payload = {
        "employee_id": employee_id,              # recognized False여도 DB 기록을 남기려면 그대로 둠
        "camera_id": camera_id,
        "event_type": "CHECK_IN",
        "similarity": similarity,
        "recognized": recognized,
        "event_time": datetime.now(timezone.utc).isoformat(),
    }

    try:
        supabase.table("attendance_logs").insert(log_payload).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert attendance_logs failed: {str(e)}")

    return RecognizeResponse(
        employee_id=employee_id if recognized else None,
        similarity=similarity,
        recognized=recognized
    )
