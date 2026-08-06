"""Phase 7 — zone dwell-time analytics. TODO: aggregate zone_events into per-employee dwell summaries."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass
class DwellTimeSummary:
    zone_id: str
    start: datetime
    end: datetime
    total_seconds_by_employee: dict[str, float] = field(default_factory=dict)


class DwellTimeReportGenerator(Protocol):
    def summary(self, zone_id: str, start: datetime, end: datetime) -> DwellTimeSummary:
        """Return total dwell time per employee in a zone over [start, end]."""
        ...


class NoOpDwellTimeReportGenerator:
    def summary(self, zone_id: str, start: datetime, end: datetime) -> DwellTimeSummary:
        return DwellTimeSummary(zone_id=zone_id, start=start, end=end)
