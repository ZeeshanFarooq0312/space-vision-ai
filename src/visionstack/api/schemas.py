from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EmployeeCreate(BaseModel):
    employee_code: str
    full_name: str
    role: str
    department: str | None = None


class EmployeeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    employee_code: str
    full_name: str
    role: str
    department: str | None
    active: bool


class CameraRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    camera_id: str
    name: str
    role: str
    location: str | None
    active: bool


class ZoneCreate(BaseModel):
    zone_id: str | None = None  # auto-generated if omitted
    camera_id: str
    name: str
    zone_type: Literal["allowed", "restricted", "exit"] = "allowed"
    polygon: list[tuple[float, float]] = Field(min_length=3)
    triggers_login: bool = False
    allowed_roles: list[str] = []


class ZoneRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    zone_id: str
    camera_id: str
    name: str
    zone_type: str
    polygon: list
    triggers_login: bool


class AttendanceEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    employee_id: uuid.UUID
    event_type: str
    camera_id: str
    confidence: float
    occurred_at: datetime


class AlertRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    alert_type: str
    severity: str
    message: str
    status: str
    created_at: datetime


class FaceEnrollmentResult(BaseModel):
    employee_id: uuid.UUID
    face_id: str
    embedding_dim: int


class FaceEnrollmentStatus(BaseModel):
    enrolled: bool
    face_id: str | None
    enrolled_at: datetime | None


class WebcamDevice(BaseModel):
    device_index: int
    label: str


class LiveSessionStart(BaseModel):
    device_index: int
    sample_fps: float = 8.0


class LiveSessionStatus(BaseModel):
    camera_id: str
    running: bool
    frame_count: int
    detection_count: int
    error: str | None = None
    recognized_names: list[str] = []
    # Pixel dimensions of the annotated stream, known only once the first frame has been
    # captured (None until then) -- the frontend needs these to map a zone polygon drawn on the
    # (CSS-scaled) <img> preview back to real frame-pixel coordinates before saving it.
    frame_width: int | None = None
    frame_height: int | None = None


class ZoneVisitRecord(BaseModel):
    track_id: str
    employee_id: str | None
    employee_name: str | None
    occurred_at: datetime
    crop_url: str | None


class ZoneVisitSummary(BaseModel):
    zone_id: str
    zone_name: str
    person_count: int
    visits: list[ZoneVisitRecord] = []


class ProcessedVideo(BaseModel):
    video_id: str
    camera_id: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    frame_count: int
    detection_count: int
    max_people_in_frame: int
    filename: str
    browser_playable: bool
    recognized_names: list[str] = []
    # Only populated for uploaded videos (see video_upload.py's zone-visit tracking) -- empty for
    # live-session recordings, which don't run this yet.
    zone_results: list[ZoneVisitSummary] = []


class VideoUploadResponse(BaseModel):
    video_id: str
    status: str


class UploadJobStatus(BaseModel):
    video_id: str
    status: str
    frame_count: int
    detection_count: int
    error: str | None = None
    frame_width: int | None = None
    frame_height: int | None = None
