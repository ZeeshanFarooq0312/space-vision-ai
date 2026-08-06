"""SQLAlchemy ORM models. Mirrors the schema outline in the project plan.

Biometric embedding tables (EmployeeFaceEmbedding, EmployeeBodyEmbedding) are kept separate from
Employee so encryption-at-rest / row-level security / audit logging can be layered on later
without touching the rest of the schema. That protection isn't implemented yet (see plan's
"Known Gaps") — these tables are marked sensitive in comments only for now.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from visionstack.db.base import Base
from visionstack.db.vector import body_embedding_column, face_embedding_column


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[uuid.UUID] = _uuid_pk()
    employee_code: Mapped[str] = mapped_column(String, unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)
    department: Mapped[str | None] = mapped_column(String, nullable=True)
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class EmployeeFaceEmbedding(Base):
    """Biometric data — treat as sensitive. See module docstring."""

    __tablename__ = "employee_face_embeddings"

    id: Mapped[uuid.UUID] = _uuid_pk()
    employee_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("employees.id"), index=True)
    embedding = mapped_column(face_embedding_column())
    enrolled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    is_active: Mapped[bool] = mapped_column(default=True)
    source_ref: Mapped[str | None] = mapped_column(String, nullable=True)


class EmployeeBodyEmbedding(Base):
    """Biometric data — treat as sensitive. See module docstring."""

    __tablename__ = "employee_body_embeddings"

    id: Mapped[uuid.UUID] = _uuid_pk()
    employee_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("employees.id"), index=True)
    embedding = mapped_column(body_embedding_column())
    enrolled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    is_active: Mapped[bool] = mapped_column(default=True)
    source_ref: Mapped[str | None] = mapped_column(String, nullable=True)


class Camera(Base):
    __tablename__ = "cameras"

    camera_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String)  # entry | zone | exit
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String, nullable=True)
    active: Mapped[bool] = mapped_column(default=True)


class Zone(Base):
    __tablename__ = "zones"

    zone_id: Mapped[str] = mapped_column(String, primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.camera_id"), index=True)
    name: Mapped[str] = mapped_column(String)
    zone_type: Mapped[str] = mapped_column(String)  # allowed | restricted | exit
    polygon: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    role_access: Mapped[list["ZoneRoleAccess"]] = relationship(back_populates="zone")


class ZoneRoleAccess(Base):
    __tablename__ = "zone_role_access"

    id: Mapped[uuid.UUID] = _uuid_pk()
    zone_id: Mapped[str] = mapped_column(ForeignKey("zones.zone_id"), index=True)
    role: Mapped[str] = mapped_column(String)

    zone: Mapped[Zone] = relationship(back_populates="role_access")


class Track(Base):
    """Ephemeral per-camera tracklet (Phase 6 local tracking)."""

    __tablename__ = "tracks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.camera_id"), index=True)
    global_identity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("global_identities.id"), nullable=True, index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class GlobalIdentity(Base):
    """Persistent cross-camera identity (Phase 6 fusion output). Kept separate from Track so
    Phase 6's real implementation doesn't require a schema rewrite later."""

    __tablename__ = "global_identities"

    id: Mapped[uuid.UUID] = _uuid_pk()
    employee_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("employees.id"), nullable=True, index=True
    )
    is_unknown: Mapped[bool] = mapped_column(default=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AttendanceEvent(Base):
    __tablename__ = "attendance_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    employee_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("employees.id"), index=True)
    event_type: Mapped[str] = mapped_column(String)  # login | logout | auto_logout
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.camera_id"))
    track_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tracks.id"), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ZoneEvent(Base):
    __tablename__ = "zone_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    zone_id: Mapped[str] = mapped_column(ForeignKey("zones.zone_id"), index=True)
    track_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tracks.id"))
    global_identity_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("global_identities.id"), nullable=True
    )
    employee_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("employees.id"), nullable=True)
    event_type: Mapped[str] = mapped_column(String)  # enter | exit | violation
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = _uuid_pk()
    alert_type: Mapped[str] = mapped_column(String)
    zone_event_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("zone_events.id"), nullable=True)
    attendance_event_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("attendance_events.id"), nullable=True
    )
    severity: Mapped[str] = mapped_column(String)  # low | medium | high
    message: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="open")  # open | acknowledged | resolved
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Trajectory(Base):
    """Feeds Phase 7 forensic replay. High-volume; a future partitioning candidate at scale."""

    __tablename__ = "trajectories"

    id: Mapped[uuid.UUID] = _uuid_pk()
    global_identity_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("global_identities.id"), index=True
    )
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.camera_id"))
    point: Mapped[dict] = mapped_column(JSON)  # {"x": float, "y": float}
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
