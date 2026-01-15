# api/main.py
from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone
from typing import Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import traceback

from api.model_assets import ensure_models
from api.routes import employees, faces, logs, cameras, recognize, schedules

load_dotenv()

app = FastAPI(title="Face Attendance API")

# ---- CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prod에서는 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Routers
app.include_router(employees.router)
app.include_router(faces.router)
app.include_router(logs.router)
app.include_router(cameras.router)
app.include_router(recognize.router)
app.include_router(schedules.router)

# ---- Debug flag
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "0").strip() == "1"


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # 운영에서는 trace 노출 위험. DEBUG_ERRORS=1 일 때만 반환.
    if DEBUG_ERRORS:
        return JSONResponse(
            status_code=500,
            content={
                "msg": "unhandled exception",
                "error": repr(exc),
                "path": str(request.url),
                "trace": traceback.format_exc()[-2500:],
            },
        )
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "face-attendance-api"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


@app.get("/__version")
def __version() -> Dict[str, Any]:
    # Render가 어떤 커밋을 돌리는지 확인용
    return {
        "render_git_commit": os.getenv("RENDER_GIT_COMMIT"),
        "render_service_id": os.getenv("RENDER_SERVICE_ID"),
    }


@app.on_event("startup")
def _startup():
    # 모델 파일 확보 (다운로드/캐시)
    ensure_models()

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "msg": "unhandled server error",
            "error": repr(exc),
            "trace": traceback.format_exc()[-2500:],
            "path": str(request.url),
        },
    )