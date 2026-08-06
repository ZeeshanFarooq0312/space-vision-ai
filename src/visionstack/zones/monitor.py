"""Phase 5 — per-frame zone entry/exit/violation detection. TODO: implement point-in-polygon
tracking with per-track zone-membership state, driving zone_events + restricted-zone alerts."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from visionstack.common.geometry import Point
from visionstack.common.types import ZoneEvent


class ZoneMonitor(Protocol):
    def check(
        self,
        track_id: str,
        camera_id: str,
        point: Point,
        employee_role: str | None,
        ts: datetime,
    ) -> ZoneEvent | None:
        """Evaluate a track's current floor position against configured zones for this camera,
        returning a ZoneEvent on entry/exit/violation, or None if nothing changed."""
        ...


class NoOpZoneMonitor:
    def check(
        self,
        track_id: str,
        camera_id: str,
        point: Point,
        employee_role: str | None,
        ts: datetime,
    ) -> ZoneEvent | None:
        return None
