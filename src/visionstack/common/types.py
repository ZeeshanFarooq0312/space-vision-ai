"""Shared domain dataclasses used across pipeline phases.

This module is the seam between phases: modules should only exchange data
through these types (plus each module's own Protocol interfaces), never by
reaching into another module's internals. That's what keeps the modular
monolith splittable into services later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


@dataclass
class Frame:
    camera_id: str
    frame_id: int
    timestamp: datetime
    image: np.ndarray


@dataclass
class Detection:
    camera_id: str
    frame_id: int
    timestamp: datetime
    bbox: BBox
    confidence: float
    class_name: str = "person"
    track_id: str | None = None  # set by PersonDetector.track(); None from plain .detect()
    # Appearance embedding for this specific detection (not the track), set by the orchestrator
    # before local tracking so TrackTrackLocalTracker can use it for ReID-based association. None
    # until then, or when no body embedder is wired in.
    embedding: np.ndarray | None = None


@dataclass
class FaceBox:
    bbox: BBox
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Track:
    track_id: str
    camera_id: str
    detections: list[Detection] = field(default_factory=list)
    started_at: datetime | None = None
    ended_at: datetime | None = None


@dataclass
class GlobalIdentity:
    global_id: str
    employee_id: str | None
    is_unknown: bool
    confidence: float = 0.0


@dataclass
class MatchResult:
    employee_id: str | None
    confidence: float
    is_unknown: bool
    modality: Literal["face", "body", "none"]


@dataclass
class AttendanceEvent:
    employee_id: str
    camera_id: str
    event_type: Literal["login", "logout", "auto_logout"]
    occurred_at: datetime
    confidence: float = 0.0
    track_id: str | None = None


@dataclass
class ZoneEvent:
    zone_id: str
    camera_id: str
    track_id: str
    event_type: Literal["enter", "exit", "violation"]
    occurred_at: datetime
    employee_id: str | None = None
    global_identity_id: str | None = None


@dataclass
class Alert:
    alert_type: str
    severity: Literal["low", "medium", "high"]
    message: str
    occurred_at: datetime
    zone_event: ZoneEvent | None = None
    attendance_event: AttendanceEvent | None = None
