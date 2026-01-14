from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import employees, faces, logs, cameras, recognize, schedules  # ✅ add schedules

load_dotenv()

app = FastAPI(title="Face Attendance API")

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
