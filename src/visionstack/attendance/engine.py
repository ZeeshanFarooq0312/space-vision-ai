"""Phase 4 — attendance event recording.

`DbAttendanceEngine` implements the login half only (`on_entry_match`), driving the zone-based
login feature (see zones/monitor.DbZoneMonitor + api/live_stream.py): a recognized employee's
foot-point entering a zone with `triggers_login=True` calls this, which writes an
AttendanceEvent(login) -- once per employee per calendar day, not once per zone-entry, so walking
in and out of the zone repeatedly in one day doesn't spam duplicate logins. `on_exit_match` /
`run_end_of_day_auto_logout` (logout, working-hours calc) are out of scope for that feature and
still stubbed -- see NoOpAttendanceEngine.
"""
from __future__ import annotations

from datetime import datetime, timezone
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
    """TODO: implement logout + working-hours-calc writes against the attendance_events table."""

    def on_entry_match(self, employee_id: str, camera_id: str, ts: datetime) -> AttendanceEvent | None:
        return None

    def on_exit_match(self, employee_id: str, camera_id: str, ts: datetime) -> AttendanceEvent | None:
        return None

    def run_end_of_day_auto_logout(self, as_of: datetime) -> list[AttendanceEvent]:
        return []


class DbAttendanceEngine:
    def on_entry_match(self, employee_id: str, camera_id: str, ts: datetime) -> AttendanceEvent | None:
        # Imported lazily so importing this module doesn't require a DB connection to exist --
        # matches zones/monitor.py's DbZoneMonitor, same rationale.
        from visionstack.db.camera_helpers import ensure_camera
        from visionstack.db.models import AttendanceEvent as AttendanceEventRow
        from visionstack.db.session import SessionLocal

        day_start = ts.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        with SessionLocal() as db:
            already_logged_in = (
                db.query(AttendanceEventRow)
                .filter(
                    AttendanceEventRow.employee_id == employee_id,
                    AttendanceEventRow.event_type == "login",
                    AttendanceEventRow.occurred_at >= day_start,
                )
                .first()
            )
            if already_logged_in is not None:
                return None

            ensure_camera(db, camera_id)
            row = AttendanceEventRow(
                employee_id=employee_id,
                event_type="login",
                camera_id=camera_id,
                confidence=1.0,
                occurred_at=ts,
            )
            db.add(row)
            db.commit()

        return AttendanceEvent(
            employee_id=employee_id, camera_id=camera_id, event_type="login", occurred_at=ts, confidence=1.0
        )

    def on_exit_match(self, employee_id: str, camera_id: str, ts: datetime) -> AttendanceEvent | None:
        return None

    def run_end_of_day_auto_logout(self, as_of: datetime) -> list[AttendanceEvent]:
        return []
