"""Phase 3 — employee face/body embedding gallery. TODO: back with employee_*_embeddings tables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np


@dataclass
class GalleryMatch:
    employee_id: str
    similarity: float


class EmployeeGallery(Protocol):
    def enroll(self, employee_id: str, embedding: np.ndarray, modality: Literal["face", "body"]) -> None:
        """Add an initial embedding for an employee."""
        ...

    def re_enroll(self, employee_id: str, embedding: np.ndarray, modality: Literal["face", "body"]) -> None:
        """Add an additional embedding for an employee whose appearance has changed over time."""
        ...

    def query(
        self, embedding: np.ndarray, modality: Literal["face", "body"], top_k: int = 1
    ) -> list[GalleryMatch]:
        """Return the top_k closest gallery matches by cosine similarity."""
        ...


class NoOpEmployeeGallery:
    """TODO: implement gallery backed by employee_face_embeddings / employee_body_embeddings (pgvector)."""

    def enroll(self, employee_id: str, embedding: np.ndarray, modality: Literal["face", "body"]) -> None:
        return None

    def re_enroll(self, employee_id: str, embedding: np.ndarray, modality: Literal["face", "body"]) -> None:
        return None

    def query(
        self, embedding: np.ndarray, modality: Literal["face", "body"], top_k: int = 1
    ) -> list[GalleryMatch]:
        return []
