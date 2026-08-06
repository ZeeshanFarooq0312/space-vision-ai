"""Phase 3 — full-body ReID embedding, used as a fallback when face data is unavailable.

TODO: implement OSNet.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

EMBEDDING_DIM = 512


class BodyEmbedder(Protocol):
    def embed(self, person_crop: np.ndarray) -> np.ndarray:
        """Return a 512-d, L2-normalized full-body appearance embedding."""
        ...


class NoOpBodyEmbedder:
    """TODO: implement OSNet body ReID embedding extraction."""

    def embed(self, person_crop: np.ndarray) -> np.ndarray:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
