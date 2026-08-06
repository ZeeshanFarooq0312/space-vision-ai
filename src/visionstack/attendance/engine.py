"""Phase 4 — attendance event recording. TODO: back with attendance_events table + working-hours calc."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from visionstack.common.types import AttendanceEvent


class AttendanceEngine(Protocol):
    def on_entry_match(self, employee_id: str, camera_id: str, ts: datetime) -> AttendanceEvent | None:
        """Record a login event on a confirmed entry-camera identity match."""
        ...

    def on_exit_match(self, employee_id: str, camera_id: str, ts: datetime) -> AttendanceEvent | None:
        """Record a logout event on a confirmed exit-camera identity match and compute working hours."""
        ...

    def run_end_of_day_auto_logout(self, as_of: datetime) -> list[AttendanceEvent]:
        """Fallback: auto-logout any employee still checked in with no exit-camera event today."""
        ...


class NoOpAttendanceEngine:
    """TODO: implement login/logout writes against the attendance_events table."""

    def on_entry_match(self, employee_id: str, camera_id: str, ts: datetime) -> AttendanceEvent | None:
        return None

    def on_exit_match(self, employee_id: str, camera_id: str, ts: datetime) -> AttendanceEvent | None:
        return None

    def run_end_of_day_auto_logout(self, as_of: datetime) -> list[AttendanceEvent]:
        return []
