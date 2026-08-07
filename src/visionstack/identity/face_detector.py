"""Phase 3 — face detection + landmark alignment. TODO: implement RetinaFace/SCRFD.

HaarFaceDetector below is a stand-in for that: OpenCV's bundled Haar cascade, with no landmark
output, used only to crop a tight-enough face region out of a person detection before handing it
to the external verify API (see identity/face_api_client.py) -- that service does its own (more
accurate) detection and quality gating, so this only needs to be good enough to avoid sending a
whole-body crop where the face is a tiny fraction of the frame.
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from visionstack.common.types import BBox, FaceBox


class FaceDetector(Protocol):
    def detect(self, person_crop: np.ndarray) -> list[FaceBox]:
        """Return faces found in a person bounding-box crop, with 5-point landmarks for alignment."""
        ...


class NoOpFaceDetector:
    """TODO: implement RetinaFace/SCRFD face detection + 5-landmark alignment."""

    def detect(self, person_crop: np.ndarray) -> list[FaceBox]:
        return []


class HaarFaceDetector:
    """No landmarks (Haar cascades don't produce them) -- unsuitable for alignment-dependent
    embedding models, fine for "is there a face here, roughly where" crops."""

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, person_crop: np.ndarray) -> list[FaceBox]:
        if person_crop.size == 0:
            return []
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [
            FaceBox(bbox=BBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h)), confidence=1.0)
            for x, y, w, h in faces
        ]
