"""Phase 5 — zone configuration. Real: loads and parses configs/zones.yaml into usable Zone objects."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from visionstack.common.config import load_zones
from visionstack.common.geometry import Point, point_in_polygon


@dataclass
class Zone:
    zone_id: str
    camera_id: str
    name: str
    zone_type: Literal["allowed", "restricted", "exit"]
    polygon: list[Point]
    allowed_roles: list[str]

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)

    def is_authorised(self, role: str | None) -> bool:
        if self.zone_type != "restricted":
            return True
        if not self.allowed_roles:
            return False
        return role in self.allowed_roles


class ZoneRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self._zones: dict[str, Zone] = {
            z.zone_id: Zone(
                zone_id=z.zone_id,
                camera_id=z.camera_id,
                name=z.name,
                zone_type=z.zone_type,
                polygon=[tuple(p) for p in z.polygon],
                allowed_roles=z.allowed_roles,
            )
            for z in load_zones(path).zones
        }

    def get(self, zone_id: str) -> Zone:
        return self._zones[zone_id]

    def for_camera(self, camera_id: str) -> list[Zone]:
        return [z for z in self._zones.values() if z.camera_id == camera_id]

    def all(self) -> list[Zone]:
        return list(self._zones.values())
