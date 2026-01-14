from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Any, Literal

from pydantic import BaseModel, Field


# -----------------------------
# Employees
# -----------------------------
class EmployeeCreateRequest(BaseModel):
    employee_code: Optional[str] = Field(default=None, description="Employee visible code (e.g., EMP001)")
    name: str = Field(..., min_length=1)
    is_active: bool = True
    role: Optional[Literal["Team Leader", "Manager", "Worker"]] = None


class EmployeeUpdateRequest(BaseModel):
    employee_code: Optional[str] = None
    name: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[Literal["Team Leader", "Manager", "Worker"]] = None


class EmployeeResponse(BaseModel):
    employee_id: int
    employee_code: Optional[str] = None
    name: str
    is_active: bool = True
    role: Optional[str] = None


# -----------------------------
# Face Embeddings
# -----------------------------
class FaceEmbeddingUpsertRequest(BaseModel):
    embedding_dim: int = 512
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    embedding: Any = None  # pgvector / user-defined type: store as list or string depending on your DB setting


class FaceEmbeddingResponse(BaseModel):
    employee_id: int
    embedding_dim: int = 512
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    embedding: Any = None


# -----------------------------
# Cameras
# -----------------------------
class CameraCreateRequest(BaseModel):
    camera_id: str = Field(..., min_length=1)
    is_active: bool = True


class CameraUpdateRequest(BaseModel):
    is_active: Optional[bool] = None


class CameraResponse(BaseModel):
    camera_id: str
    is_active: bool = True


# -----------------------------
# Logs
# -----------------------------
class AttendanceLogCreateRequest(BaseModel):
    event_type: str = Field(..., description="CHECK_IN / CHECK_OUT etc (your enum)")
    camera_id: str
    recognized: bool
    similarity: Optional[float] = None
    employee_id: Optional[int] = None
    event_time: Optional[datetime] = None


class AttendanceLogUpdateRequest(BaseModel):
    event_type: Optional[str] = None
    camera_id: Optional[str] = None
    recognized: Optional[bool] = None
    similarity: Optional[float] = None
    employee_id: Optional[int] = None
    event_time: Optional[datetime] = None


class AttendanceLogResponse(BaseModel):
    log_id: int
    event_time: datetime
    event_type: str
    camera_id: str
    recognized: bool
    similarity: Optional[float] = None
    employee_id: Optional[int] = None
    created_at: datetime


# -----------------------------
# Schedules
# -----------------------------
class ScheduleCreateRequest(BaseModel):
    employee_id: int
    schedule: str = Field(..., min_length=1)
    start_time: datetime
    end_time: datetime


class ScheduleUpdateRequest(BaseModel):
    schedule: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class ScheduleResponse(BaseModel):
    schedule_id: int
    employee_id: int
    schedule: str
    start_time: datetime
    end_time: datetime


# -----------------------------
# Recognize API Response (keep)
# -----------------------------
class RecognizeResponse(BaseModel):
    ok: bool = True
    recognized: bool
    employee_id: Optional[int] = None
    employee_name: Optional[str] = None
    similarity: Optional[float] = None
    message: Optional[str] = None
