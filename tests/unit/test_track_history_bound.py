"""TTTrack.history used to grow one entry per frame for a track's entire lifetime, unbounded --
see tracking/tracktrack/track.py's HISTORY_MAX_ENTRIES docstring for the bug this guards against.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from visionstack.common.types import BBox, Detection
from visionstack.tracking.local_tracker import TrackTrackLocalTracker
from visionstack.tracking.tracktrack.track import HISTORY_MAX_ENTRIES


def test_a_long_lived_track_does_not_accumulate_unbounded_history():
    tracker = TrackTrackLocalTracker(sample_fps=4.0)
    t0 = datetime(2026, 1, 1)

    for i in range(80):
        det = Detection(
            camera_id="cam-1",
            frame_id=i,
            timestamp=t0 + timedelta(seconds=i / 4.0),
            bbox=BBox(x1=100 + i * 0.1, y1=100, x2=180 + i * 0.1, y2=280),
            confidence=0.9,
        )
        tracker.update([det])

    live_tracks = tracker._tracker.tracks
    assert live_tracks, "expected at least one live track after 80 stable frames"
    for t in live_tracks:
        assert len(t.history) <= HISTORY_MAX_ENTRIES
