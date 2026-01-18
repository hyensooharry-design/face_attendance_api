# api/schemas.py
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# -----------------------------
# employees
# -----------------------------
# DB: role TEXT (자유 문자열)
# API: Literal 제거 → Optional[str]로 완화
RoleType = str


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
    # Render schema: persons(employee_id) -> face_embeddings(person_id)
    employee_id: int
    person_id: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    # embedding(pgvector)은 응답에서 제외


# -----------------------------
# cameras
# -----------------------------
class CameraCreateRequest(BaseModel):
    camera_id: str = Field(min_length=1)
    name: Optional[str] = None
    location: Optional[str] = None


class CameraUpdateRequest(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None


class CameraResponse(BaseModel):
    camera_id: str
    name: Optional[str] = None
    location: Optional[str] = None
    created_at: Optional[str] = None


# -----------------------------
# attendance_logs
# event_type: DB에서는 TEXT / enum 혼용 가능 → str 유지
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
    created_at: str


# -----------------------------
# schedules
# -----------------------------
# DB: start_time / end_time NULL 허용
# API: 필수 → Optional 로 완화
class ScheduleCreateRequest(BaseModel):
    employee_id: int
    schedule: str
    start_time: Optional[str] = None  # timestamptz ISO string
    end_time: Optional[str] = None    # timestamptz ISO string


class ScheduleUpdateRequest(BaseModel):
    employee_id: Optional[int] = None
    schedule: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class ScheduleResponse(BaseModel):
    schedule_id: int
    employee_id: int
    schedule: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
