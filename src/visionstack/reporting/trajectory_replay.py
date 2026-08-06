"""Phase 7 — forensic trajectory replay, e.g. "who was in Zone B between 14:00-15:30".
TODO: query the trajectories table."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class TrajectoryPoint:
    global_identity_id: str
    camera_id: str
    x: float
    y: float
    occurred_at: datetime


class TrajectoryReplay(Protocol):
    def query(self, zone_id: str, start: datetime, end: datetime) -> list[TrajectoryPoint]:
        """Return trajectory points recorded within a zone over [start, end]."""
        ...


class NoOpTrajectoryReplay:
    def query(self, zone_id: str, start: datetime, end: datetime) -> list[TrajectoryPoint]:
        return []
