"""Phase 6 — single-camera multi-object tracking. TODO: implement ByteTrack + Kalman filter."""
from __future__ import annotations

import itertools
from typing import Protocol

from visionstack.common.types import Detection, Track


class LocalTracker(Protocol):
    def update(self, detections: list[Detection]) -> list[Track]:
        """Associate this frame's detections with existing/new local tracklets."""
        ...


class PassthroughLocalTracker:
    """Assigns a fresh track id to every detection each frame — no real association/occlusion
    handling yet. Exists so downstream phases (identity, attendance, zones) receive real Track
    objects today. TODO: replace with ByteTrack for persistent per-camera track continuity."""

    def __init__(self) -> None:
        self._counter = itertools.count()

    def update(self, detections: list[Detection]) -> list[Track]:
        tracks: list[Track] = []
        for detection in detections:
            track_id = f"{detection.camera_id}-{next(self._counter)}"
            tracks.append(
                Track(
                    track_id=track_id,
                    camera_id=detection.camera_id,
                    detections=[detection],
                    started_at=detection.timestamp,
                )
            )
        return tracks
