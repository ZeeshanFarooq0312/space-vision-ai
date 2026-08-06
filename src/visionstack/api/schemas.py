from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


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


class ZoneRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    zone_id: str
    camera_id: str
    name: str
    zone_type: str
    polygon: list


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
