"""Phase 3 — identity matching, fusing face + body embeddings against the employee gallery."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from visionstack.common.types import MatchResult


class IdentityMatcher(Protocol):
    def match(
        self, face_embedding: np.ndarray | None, body_embedding: np.ndarray | None
    ) -> MatchResult:
        """Resolve an employee identity from available face/body embeddings, or flag unknown."""
        ...


class NoOpIdentityMatcher:
    """TODO: implement gallery lookup + face/body fusion, thresholded per configs/pipeline.yaml."""

    def match(
        self, face_embedding: np.ndarray | None, body_embedding: np.ndarray | None
    ) -> MatchResult:
        return MatchResult(employee_id=None, confidence=0.0, is_unknown=True, modality="none")
