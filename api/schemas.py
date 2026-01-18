# api/schemas.py
from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field


# -----------------------------
# employees
# -----------------------------
RoleType = Literal["Team Leader", "Manager", "Worker"]


class EmployeeCreateRequest(BaseModel):
    employee_code: Optional[str] = Field(default=None)
    name: str = Field(min_length=1)
    is_active: bool = True
    role: Optional[RoleType] = None


class EmployeeUpdateRequest(BaseModel):
    employee_code: Optional[str] = None
    name: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[RoleType] = None


class EmployeeResponse(BaseModel):
    employee_id: int
    employee_code: Optional[str] = None
    name: str
    is_active: bool = True
    role: Optional[RoleType] = None
    has_face: bool = False


# -----------------------------
# face_embeddings
# -----------------------------
class FaceResponse(BaseModel):
    employee_id: int
    embedding_dim: int = 512
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    # Embedding column (pgvector) is too long, excluded from default response (query separately if needed)


# -----------------------------
# cameras
# -----------------------------
class CameraCreateRequest(BaseModel):
    camera_id: str = Field(min_length=1)
    is_active: bool = True


class CameraUpdateRequest(BaseModel):
    is_active: Optional[bool] = None


class CameraResponse(BaseModel):
    camera_id: str
    is_active: bool = True
    created_at: Optional[str] = None


# -----------------------------
# attendance_logs
# event_type is USER-DEFINED (enum) in DB, so we accept it as str here
# -----------------------------
class AttendanceLogCreateRequest(BaseModel):
    event_time: Optional[str] = None  # timestamptz ISO string (optional)
    event_type: str
    camera_id: str
    recognized: bool
    similarity: Optional[float] = None
    employee_id: Optional[int] = None


class AttendanceLogUpdateRequest(BaseModel):
    event_time: Optional[str] = None
    event_type: Optional[str] = None
    camera_id: Optional[str] = None
    recognized: Optional[bool] = None
    similarity: Optional[float] = None
    employee_id: Optional[int] = None


class AttendanceLogResponse(BaseModel):
    log_id: int
    event_time: str
    event_type: str
    camera_id: str
    recognized: bool
    similarity: Optional[float] = None
    employee_id: Optional[int] = None
    name: Optional[str] = None
    employee_code: Optional[str] = None
    created_at: str


# -----------------------------
# schedules
# -----------------------------
class ScheduleCreateRequest(BaseModel):
    employee_id: int
    schedule: str
    start_time: str  # timestamptz ISO string
    end_time: str    # timestamptz ISO string


class ScheduleUpdateRequest(BaseModel):
    employee_id: Optional[int] = None
    schedule: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class ScheduleResponse(BaseModel):
    schedule_id: int
    employee_id: int
    schedule: str
    start_time: str
    end_time: str
