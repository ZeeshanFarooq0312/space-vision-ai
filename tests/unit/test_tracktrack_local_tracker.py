from datetime import datetime, timedelta, timezone

import numpy as np

from visionstack.common.types import BBox, Detection
from visionstack.tracking.local_tracker import TrackTrackLocalTracker

MIN_LEN = 3  # TrackTrackParams.min_len default -- frames before a track is reported as "Tracked"


def _detection(
    frame_id: int, x1: float, embedding: np.ndarray | None, camera_id: str = "cam-1"
) -> Detection:
    return Detection(
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=frame_id),
        bbox=BBox(x1=x1, y1=100.0, x2=x1 + 50.0, y2=250.0),
        confidence=0.9,
        embedding=embedding,
    )


def test_persists_track_id_once_min_len_frames_matched():
    tracker = TrackTrackLocalTracker(sample_fps=5.0)
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0  # unit vector

    seen_track_ids = []
    for frame_id in range(1, 6):
        detection = _detection(frame_id, x1=100.0 + frame_id, embedding=embedding)
        tracks = tracker.update([detection])
        if frame_id < MIN_LEN:
            assert tracks == []
        else:
            assert len(tracks) == 1
            seen_track_ids.append(tracks[0].track_id)

    # Same person, same track id across every frame it was reported in.
    assert len(set(seen_track_ids)) == 1


def test_two_well_separated_people_get_distinct_track_ids():
    tracker = TrackTrackLocalTracker(sample_fps=5.0)
    emb_a = np.zeros(512, dtype=np.float32)
    emb_a[0] = 1.0
    emb_b = np.zeros(512, dtype=np.float32)
    emb_b[1] = 1.0

    tracks = []
    for frame_id in range(1, MIN_LEN + 1):
        dets = [
            _detection(frame_id, x1=100.0, embedding=emb_a),
            _detection(frame_id, x1=900.0, embedding=emb_b),
        ]
        tracks = tracker.update(dets)

    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def test_zero_embedding_does_not_raise_or_nan_the_track():
    """NoOpBodyEmbedder (Phase 3 stub) returns an all-zero vector -- update_features must not
    divide by a zero norm."""
    tracker = TrackTrackLocalTracker(sample_fps=5.0)

    tracks = []
    zero_embedding = np.zeros(512, dtype=np.float32)
    for frame_id in range(1, MIN_LEN + 1):
        detection = _detection(frame_id, x1=100.0 + frame_id, embedding=zero_embedding)
        tracks = tracker.update([detection])

    assert len(tracks) == 1
    assert np.isfinite(tracks[0].detections[0].bbox.x1)


def test_empty_frame_does_not_raise():
    tracker = TrackTrackLocalTracker(sample_fps=5.0)
    assert tracker.update([]) == []

    detection = _detection(1, x1=100.0, embedding=None)
    assert tracker.update([detection]) == []  # first frame: still "New", not yet Tracked
    assert tracker.update([]) == []


def test_a_moving_person_keeps_their_id_across_a_brief_occlusion():
    """Regression test: a track moving faster than position_freeze_speed_threshold (see
    tracktrack/params.py's docstring for the full history) must NOT have its center-position
    Kalman velocity frozen while Lost -- freezing it unconditionally once fixed a stationary-
    person drift bug, but broke exactly this case: someone walking steadily who gets briefly
    occluded (e.g. by another person crossing paths) never had their frozen, stale-position
    prediction catch up to where they actually reappeared, so they picked up a second id. The
    25px/sampled-frame walking speed here is comfortably above the 15px/frame threshold, so this
    track should be treated as "genuinely moving," not frozen.
    """
    tracker = TrackTrackLocalTracker(sample_fps=4.0)
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    x = 100.0
    seen_track_ids = set()

    for frame_id in range(1, 40):
        # Occluded (by someone else crossing in front) for 2 seconds (8 frames at 4fps) starting
        # once the track is established, while the person keeps walking underneath the occlusion.
        occluded = 10 <= frame_id < 18
        x += 25.0
        if not occluded:
            detection = _detection(frame_id, x1=x, embedding=embedding)
            tracks = tracker.update([detection])
            for t in tracks:
                seen_track_ids.add(t.track_id)
        else:
            tracker.update([])

    assert len(seen_track_ids) == 1


def test_a_moving_person_keeps_their_id_after_slowing_during_a_long_occlusion():
    """Regression test for the failure mode lost_extrapolation_cutoff_frames/
    lost_search_growth_px_per_frame exist to fix (see params.py docstrings for the real-footage
    measurement this is modeled on directly): a track moving 15.8px/sampled-frame -- just above
    position_freeze_speed_threshold, so extrapolation, not freezing, kicks in -- went unmatched for
    48 frames (6s at 8fps) in the reference video. Straight-line extrapolation the whole way would
    land ~758px from where it was last seen, but the person had only actually moved ~74px (slowed
    down/paused, ordinary human behavior, not a tracker bug) -- enough deviation to leave zero IoU
    with the stale prediction and get hard-gated out regardless of appearance, producing a fresh id
    for what should have reconnected as the same person.
    """
    # max_time_lost_seconds must comfortably exceed the occlusion below (default 3.0s would
    # remove the track outright before the extrapolation-cutoff/search-radius logic ever gets a
    # chance to reconnect it) -- matches this project's production video_upload.py setting.
    tracker = TrackTrackLocalTracker(sample_fps=8.0, max_time_lost_seconds=300.0)
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    x = 100.0
    seen_track_ids = set()

    for frame_id in range(1, 20):
        # Establish the track walking steadily at 15.8px/frame -- same speed measured in the real
        # failure case.
        x += 15.8
        detection = _detection(frame_id, x1=x, embedding=embedding)
        tracks = tracker.update([detection])
        for t in tracks:
            seen_track_ids.add(t.track_id)

    # Occluded for 48 frames (6s at 8fps, matching the real case) -- comfortably past
    # lost_extrapolation_cutoff_frames (16), so extrapolation stops well before the real
    # reappearance point, same mechanism the real footage validated.
    for _ in range(48):
        tracker.update([])
    x += 74.0

    detection = _detection(60, x1=x, embedding=embedding)
    tracks = tracker.update([detection])
    for t in tracks:
        seen_track_ids.add(t.track_id)
    # A couple more matched frames confirm it stays the same track, not just a one-off fluke match.
    for frame_id in range(61, 66):
        x += 2.0
        detection = _detection(frame_id, x1=x, embedding=embedding)
        tracks = tracker.update([detection])
        for t in tracks:
            seen_track_ids.add(t.track_id)

    assert len(seen_track_ids) == 1


def test_a_near_stationary_person_with_jittery_height_keeps_one_id():
    """Regression test for the ORIGINAL bug position_freeze_speed_threshold exists to fix: a
    seated person's box height (not x/y) jittering -- legs going in/out of view under a desk --
    was enough noise for the Kalman filter to attribute a small velocity to a track, which then
    ran away in a straight line once Lost (no correction, no decay) until unrecoverable. This
    track's real velocity should measure near-zero (position_freeze_speed_threshold=15px/frame is
    comfortably above what jitter alone produces), so it should get frozen and hold its id.
    """
    tracker = TrackTrackLocalTracker(sample_fps=4.0)
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    cx, cy = 400.0, 300.0
    base_w = 90.0
    heights = [175.0, 174.0, 120.0, 130.0, 150.0, 176.0, 173.0]  # legs visible vs. occluded
    seen_track_ids = set()

    for frame_id in range(1, 60):
        h = heights[frame_id % len(heights)]
        x1 = cx - base_w / 2
        y2 = cy + 85.0
        y1 = y2 - h
        detection = Detection(
            camera_id="cam-1",
            frame_id=frame_id,
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=frame_id / 4.0),
            bbox=BBox(x1=x1, y1=y1, x2=x1 + base_w, y2=y2),
            confidence=0.85,
            embedding=embedding,
        )
        tracks = tracker.update([detection])
        for t in tracks:
            seen_track_ids.add(t.track_id)

    assert len(seen_track_ids) == 1
