"""Phase 6 — single-camera multi-object tracking.

Two implementations:
- `PassthroughLocalTracker`: assigns a fresh track id to every detection every frame. No real
  association/occlusion handling. Kept as the zero-dependency default and for tests.
- `TrackTrackLocalTracker`: real persistent tracking via a vendored core from TrackTrack (CVPR
  2025, kamkyu94/TrackTrack, MIT license; see tracking/tracktrack/). Kalman-filter motion +
  appearance-embedding association, ByteTrack-style two-stage (high/low confidence) matching.

TrackTrackLocalTracker is wired into `api/video_upload.py` (uploads all come from one fixed
camera, `UPLOAD_CAMERA_ID` -- see that module), and into the orchestrator/`run_pipeline.py` demo
path. It is NOT used by `api/live_stream.py` (the live webcam path), which uses Ultralytics'
built-in BoT-SORT via `PersonDetector.track()` instead -- see that module's `track()` docstring
for why (BoT-SORT's built-in global motion compensation handles a moving/handheld camera; the
vendored TrackTrack core here does not, see tracking/tracktrack/cmc.py -- don't point it at a
non-fixed camera without adding that first).
"""
from __future__ import annotations

import itertools
from dataclasses import replace
from datetime import datetime
from typing import Protocol

import numpy as np

from visionstack.common.types import BBox, Detection, Track
from visionstack.tracking.tracktrack.params import TrackTrackParams
from visionstack.tracking.tracktrack.tracker import Tracker as _TrackTrackTracker

PERSON_CLASS_ID = 0


class LocalTracker(Protocol):
    def update(self, detections: list[Detection]) -> list[Track]:
        """Associate this frame's detections with existing/new local tracklets."""
        ...


class PassthroughLocalTracker:
    """Assigns a fresh track id to every detection each frame — no real association/occlusion
    handling yet. Exists so downstream phases (identity, attendance, zones) receive real Track
    objects today."""

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
                    detections=[replace(detection, track_id=track_id)],
                    started_at=detection.timestamp,
                )
            )
        return tracks


class TrackTrackLocalTracker:
    """Wraps the vendored TrackTrack core behind the `LocalTracker` protocol.

    One instance tracks one camera's stream (matches the vendored tracker's own statefulness —
    track ids and Kalman filters are internal, per-instance state). Each `Detection` passed in
    should already have `.embedding` set (see `pipeline.orchestrator.Pipeline._process_frame`,
    which runs the body embedder before tracking specifically so this has real appearance features
    to associate on) — a `None` embedding falls back to an all-zero vector. Note that appearance
    similarity is currently weighted to 0 anyway by `TrackTrackParams`' default (see that module's
    docstring: `identity.body_embedder.NoOpBodyEmbedder` is still a stub, so there's no real
    appearance signal to weight in yet regardless of what's passed here) — association runs on
    motion/confidence alone today; a real embedder should also mean turning that weight back up.
    """

    def __init__(
        self,
        sample_fps: float,
        embedding_dim: int = 512,
        max_time_lost_seconds: float = 3.0,
        params: TrackTrackParams | None = None,
    ) -> None:
        self._embedding_dim = embedding_dim
        params = params or TrackTrackParams()
        params.max_time_lost = max(1, round(max_time_lost_seconds * sample_fps))
        self._tracker = _TrackTrackTracker(params)
        self._started_at: dict[int, datetime] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        if not detections:
            self._tracker.update_without_detections()
            return []

        camera_id = detections[0].camera_id
        rows = np.zeros((len(detections), 6 + self._embedding_dim), dtype=np.float64)
        for i, d in enumerate(detections):
            rows[i, :4] = [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]
            rows[i, 4] = d.confidence
            rows[i, 5] = PERSON_CLASS_ID
            if d.embedding is not None:
                rows[i, 6:] = d.embedding

        tt_tracks = self._tracker.update(rows, refs=detections)

        tracks: list[Track] = []
        for tt in tt_tracks:
            matched_detection: Detection = tt.ref
            track_id = f"{camera_id}-{tt.track_id}"
            x1, y1, x2, y2 = tt.x1y1x2y2
            smoothed_detection = replace(
                matched_detection,
                bbox=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                track_id=track_id,
            )
            started_at = self._started_at.setdefault(tt.track_id, matched_detection.timestamp)
            tracks.append(
                Track(
                    track_id=track_id,
                    camera_id=camera_id,
                    detections=[smoothed_detection],
                    started_at=started_at,
                )
            )
        return tracks
