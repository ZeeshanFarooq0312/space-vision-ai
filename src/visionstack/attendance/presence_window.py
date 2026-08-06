"""Phase 4 — false-logout prevention via a sustained-presence confirmation window."""
from __future__ import annotations

from datetime import datetime
from typing import Protocol


class PresenceConfirmationWindow(Protocol):
    def observe(self, track_id: str, camera_id: str, ts: datetime) -> bool:
        """Record a presence observation; return True once sustained presence >= the configured
        N seconds (attendance.presence_confirmation_seconds) is confirmed for this track."""
        ...


class NoOpPresenceConfirmationWindow:
    """TODO: implement a sliding time window per track_id, keyed off configs/pipeline.yaml."""

    def observe(self, track_id: str, camera_id: str, ts: datetime) -> bool:
        return False
