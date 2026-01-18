from __future__ import annotations

import os

from datetime import datetime, timezone
from typing import Dict, Any

from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.model_assets import ensure_models
from api.routes import employees, faces, logs, cameras, recognize, schedules  # ✅ add schedules
from api.routes.recognize import refresh_embeddings


load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

app = FastAPI(title="Attendance Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(employees.router)
app.include_router(faces.router)
app.include_router(logs.router)
app.include_router(cameras.router)
app.include_router(recognize.router)
app.include_router(schedules.router)  # ✅ mount schedules


@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "face-attendance-api"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

@app.on_event("startup")
def _startup():
    print("ARC_MODEL_URL:", os.getenv("ARC_MODEL_URL"))
    print("RETINA_MODEL_URL:", os.getenv("RETINA_MODEL_URL"))
    print("MODELS_DIR:", os.getenv("MODELS_DIR"))
    ensure_models()
    try:
        refresh_embeddings()
    except Exception as e:
        print(f"❌ Failed to load embeddings on startup: {e}")
