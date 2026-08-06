from datetime import datetime, timezone

from visionstack.common.types import BBox, Detection
from visionstack.tracking.cross_camera_reid import NoOpCrossCameraReID
from visionstack.tracking.local_tracker import PassthroughLocalTracker


def _detection(camera_id: str = "cam-1") -> Detection:
    return Detection(
        camera_id=camera_id,
        frame_id=0,
        timestamp=datetime.now(timezone.utc),
        bbox=BBox(0, 0, 10, 10),
        confidence=0.9,
    )


def test_passthrough_local_tracker_assigns_one_track_per_detection():
    tracker = PassthroughLocalTracker()
    tracks = tracker.update([_detection(), _detection()])
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def test_passthrough_local_tracker_ids_increase_across_calls():
    tracker = PassthroughLocalTracker()
    first = tracker.update([_detection()])[0]
    second = tracker.update([_detection()])[0]
    assert first.track_id != second.track_id


def test_noop_cross_camera_reid_passes_through_local_track_id():
    tracker = PassthroughLocalTracker()
    track = tracker.update([_detection()])[0]
    identity = NoOpCrossCameraReID().resolve(track, face_embedding=None, body_embedding=None)
    assert identity.global_id == track.track_id
    assert identity.is_unknown is True
