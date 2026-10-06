from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.model_assets import ensure_models
from api.routes import cameras, employees, faces, logs, recognize, schedules

load_dotenv()

app = FastAPI(
    title="Face Attendance API",
    version="1.0.0",
    description="Face-recognition attendance API backed by Supabase.",
)

DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "0").strip() == "1"
DUMMY_MODE = os.getenv("DUMMY_MODE", "0").strip() == "1"
AUTO_DOWNLOAD_MODELS = os.getenv("AUTO_DOWNLOAD_MODELS", "1").strip() == "1"

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(employees.router)
app.include_router(faces.router)
app.include_router(logs.router)
app.include_router(cameras.router)
app.include_router(recognize.router)
app.include_router(schedules.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    content: Dict[str, Any] = {
        "detail": "Internal Server Error",
        "path": str(request.url.path),
    }
    if DEBUG_ERRORS:
        content["error"] = repr(exc)
        content["trace"] = traceback.format_exc()[-2500:]
    return JSONResponse(status_code=500, content=content)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "face-attendance-api",
        "version": app.version,
        "dummy_mode": DUMMY_MODE,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "ts": datetime.now(timezone.utc).isoformat(),
        "dummy_mode": DUMMY_MODE,
    }


@app.get("/__version")
def version_info() -> Dict[str, Any]:
    return {
        "app_version": app.version,
        "render_git_commit": os.getenv("RENDER_GIT_COMMIT"),
        "render_service_id": os.getenv("RENDER_SERVICE_ID"),
    }


@app.on_event("startup")
def startup() -> None:
    if DUMMY_MODE:
        print("[startup] DUMMY_MODE=1: skipping model download.")
        return

    if not AUTO_DOWNLOAD_MODELS:
        print("[startup] AUTO_DOWNLOAD_MODELS=0: model download disabled.")
        return

    ensure_models()
