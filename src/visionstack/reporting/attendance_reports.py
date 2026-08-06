"""Phase 7 — daily attendance reporting. TODO: query attendance_events for a HR-facing summary."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol


@dataclass
class AttendanceRecord:
    employee_id: str
    date: date
    login_time: datetime | None
    logout_time: datetime | None
    total_hours: float | None


class AttendanceReportGenerator(Protocol):
    def daily_report(self, report_date: date) -> list[AttendanceRecord]:
        """Return one AttendanceRecord per employee with an event on report_date."""
        ...


class NoOpAttendanceReportGenerator:
    def daily_report(self, report_date: date) -> list[AttendanceRecord]:
        return []
