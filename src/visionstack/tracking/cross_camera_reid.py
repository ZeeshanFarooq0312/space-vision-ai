"""Phase 6 — cross-camera identity fusion. TODO: implement face+body embedding fusion with
Hungarian assignment and spatial-temporal graph constraints."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from visionstack.common.types import GlobalIdentity, Track


class CrossCameraReID(Protocol):
    def resolve(
        self,
        local_track: Track,
        face_embedding: np.ndarray | None,
        body_embedding: np.ndarray | None,
    ) -> GlobalIdentity:
        """Resolve (or create) the persistent global identity for a local per-camera track."""
        ...


class NoOpCrossCameraReID:
    """TODO: implement cosine similarity + Hungarian matching across camera handoffs."""

    def resolve(
        self,
        local_track: Track,
        face_embedding: np.ndarray | None,
        body_embedding: np.ndarray | None,
    ) -> GlobalIdentity:
        return GlobalIdentity(
            global_id=local_track.track_id, employee_id=None, is_unknown=True, confidence=0.0
        )
