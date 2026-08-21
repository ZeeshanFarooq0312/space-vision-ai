"""Phase 5 — per-frame zone entry/exit detection.

`DbZoneMonitor` is the real implementation: bbox-overlap-based membership tracking per track_id
(whichever zone captures the largest share of a detection's box wins when its box spans more than
one), zones loaded from the DB (see api/routers/zones.py's POST /zones). It does NOT persist ZoneEvent
rows to the `zone_events` table -- that table's `track_id` column is a non-nullable FK to a
`tracks` row, and nothing in this pipeline creates `tracks` rows (PersonDetector.track()'s track
ids are in-memory BoT-SORT ids, not DB rows), so persisting would require building that out too.
For now, `check()`'s returned events are consumed directly by the caller (see
api/live_stream.py's zone-based-login wiring) rather than written to the DB — a real gap, called
out explicitly rather than silently skipped, if zone-visit auditing/history is needed later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from visionstack.common.geometry import Point, bbox_polygon_overlap_area
from visionstack.common.types import BBox, ZoneEvent


class ZoneMonitor(Protocol):
    def check(
        self,
        track_id: str,
        camera_id: str,
        bbox: BBox,
        employee_role: str | None,
        ts: datetime,
    ) -> list[ZoneEvent]:
        """Evaluate a track's current bounding box against configured zones for this camera,
        returning a ZoneEvent for every zone entered/exited since the last call (empty if
        nothing changed)."""
        ...


class NoOpZoneMonitor:
    def check(
        self,
        track_id: str,
        camera_id: str,
        bbox: BBox,
        employee_role: str | None,
        ts: datetime,
    ) -> list[ZoneEvent]:
        return []


@dataclass
class ZoneRecord:
    zone_id: str
    camera_id: str
    name: str
    zone_type: str
    polygon: list[Point]
    triggers_login: bool
    allowed_roles: list[str] = field(default_factory=list)


def _load_zones_from_db(camera_id: str) -> list[ZoneRecord]:
    # Imported lazily so importing this module doesn't require a DB connection to exist (matters
    # for tests and for pipeline/orchestrator.py's offline demo path, neither of which touch
    # Postgres) -- only actually constructing/using DbZoneMonitor does.
    from visionstack.db.models import Zone as ZoneRow
    from visionstack.db.session import SessionLocal

    with SessionLocal() as db:
        rows = db.query(ZoneRow).filter(ZoneRow.camera_id == camera_id).all()
        return [
            ZoneRecord(
                zone_id=row.zone_id,
                camera_id=row.camera_id,
                name=row.name,
                zone_type=row.zone_type,
                polygon=[tuple(point) for point in row.polygon],
                triggers_login=bool(row.triggers_login),
                allowed_roles=[access.role for access in row.role_access],
            )
            for row in rows
        ]


class DbZoneMonitor:
    """Zones for `camera_id` are loaded once, at construction -- a zone created/edited after a
    live session has started won't take effect until the session is restarted (see
    api/live_stream.py, which builds one DbZoneMonitor per session). `zones` can be passed
    directly (bypassing the DB) for testing.
    """

    def __init__(self, camera_id: str, zones: list[ZoneRecord] | None = None) -> None:
        self.camera_id = camera_id
        self._zones = zones if zones is not None else _load_zones_from_db(camera_id)
        self._zones_by_id = {zone.zone_id: zone for zone in self._zones}
        self._membership: dict[str, set[str]] = {}

    def zone(self, zone_id: str) -> ZoneRecord | None:
        return self._zones_by_id.get(zone_id)

    @property
    def zones(self) -> list[ZoneRecord]:
        """All zones loaded for this camera -- used to draw zone outlines onto saved recordings,
        see api/live_stream.py's _draw_zones."""
        return list(self._zones)

    def check(
        self,
        track_id: str,
        camera_id: str,
        bbox: BBox,
        employee_role: str | None,
        ts: datetime,
    ) -> list[ZoneEvent]:
        # A detection can be in at most one zone at a time -- when its box overlaps more than one
        # zone (adjacent or overlapping zones), whichever zone captures the largest share of the
        # box wins, ties broken by zone_id for determinism. Without this, a single detection
        # straddling an overlap region would fire an independent "enter" (and, for a
        # triggers_login zone, a separate login) for every zone it happens to touch at all, even
        # by a sliver.
        overlaps = [
            (zone, bbox_polygon_overlap_area(bbox, zone.polygon)) for zone in self._zones
        ]
        overlapping = [(zone, area) for zone, area in overlaps if area > 0]
        current: set[str] = set()
        if overlapping:
            best_zone, _ = max(overlapping, key=lambda pair: (pair[1], pair[0].zone_id))
            current = {best_zone.zone_id}

        previous = self._membership.get(track_id, set())
        self._membership[track_id] = current

        events = [
            ZoneEvent(zone_id=zid, camera_id=camera_id, track_id=track_id, event_type="enter", occurred_at=ts)
            for zid in current - previous
        ]
        events += [
            ZoneEvent(zone_id=zid, camera_id=camera_id, track_id=track_id, event_type="exit", occurred_at=ts)
            for zid in previous - current
        ]
        return events

    def forget_track(self, track_id: str) -> None:
        """Drop membership state for a track that's left the frame (see api/live_stream.py's
        stale-track cleanup) so it starts fresh (all zones read as "entered") if the same
        track_id is ever reused."""
        self._membership.pop(track_id, None)
