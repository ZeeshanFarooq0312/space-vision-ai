"""Phase 3 — face detection + landmark alignment. TODO: implement RetinaFace/SCRFD."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from visionstack.common.types import FaceBox


class FaceDetector(Protocol):
    def detect(self, person_crop: np.ndarray) -> list[FaceBox]:
        """Return faces found in a person bounding-box crop, with 5-point landmarks for alignment."""
        ...


class NoOpFaceDetector:
    """TODO: implement RetinaFace/SCRFD face detection + 5-landmark alignment."""

    def detect(self, person_crop: np.ndarray) -> list[FaceBox]:
        return []
