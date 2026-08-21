"""Vendored tracking core from TrackTrack (CVPR 2025, kamkyu94/TrackTrack, MIT license).

Only the association/Kalman-filter tracker is vendored here — the upstream repo's YOLOX detector
and FastReID embedder are not used; this pipeline supplies its own detections (PersonDetector) and
appearance embeddings (BodyEmbedder) instead. See tracker.py for what was adapted and why.
"""
from __future__ import annotations

from visionstack.tracking.tracktrack.params import TrackTrackParams
from visionstack.tracking.tracktrack.tracker import Tracker

__all__ = ["Tracker", "TrackTrackParams"]
