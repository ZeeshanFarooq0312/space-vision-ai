from datetime import datetime, timezone

from visionstack.common.types import BBox
from visionstack.zones.monitor import DbZoneMonitor, ZoneRecord

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _monitor(triggers_login: bool = True) -> DbZoneMonitor:
    zone = ZoneRecord(
        zone_id="z1", camera_id="cam-1", name="Lobby", zone_type="allowed",
        polygon=SQUARE, triggers_login=triggers_login,
    )
    return DbZoneMonitor("cam-1", zones=[zone])


def test_entering_a_zone_emits_one_enter_event():
    monitor = _monitor()
    events = monitor.check("track-1", "cam-1", BBox(4, 4, 6, 6), None, NOW)
    assert len(events) == 1
    assert events[0].event_type == "enter"
    assert events[0].zone_id == "z1"


def test_staying_in_a_zone_emits_nothing_on_subsequent_frames():
    monitor = _monitor()
    monitor.check("track-1", "cam-1", BBox(4, 4, 6, 6), None, NOW)
    events = monitor.check("track-1", "cam-1", BBox(4.5, 4.5, 6.5, 6.5), None, NOW)
    assert events == []


def test_leaving_a_zone_emits_one_exit_event():
    monitor = _monitor()
    monitor.check("track-1", "cam-1", BBox(4, 4, 6, 6), None, NOW)
    events = monitor.check("track-1", "cam-1", BBox(50, 50, 60, 60), None, NOW)
    assert len(events) == 1
    assert events[0].event_type == "exit"
    assert events[0].zone_id == "z1"


def test_no_overlap_at_all_emits_nothing():
    monitor = _monitor()
    events = monitor.check("track-1", "cam-1", BBox(50, 50, 60, 60), None, NOW)
    assert events == []


def test_zone_metadata_lookup_exposes_triggers_login():
    monitor = _monitor(triggers_login=True)
    assert monitor.zone("z1").triggers_login is True
    assert monitor.zone("missing") is None


def test_forget_track_resets_membership():
    monitor = _monitor()
    monitor.check("track-1", "cam-1", BBox(4, 4, 6, 6), None, NOW)
    monitor.forget_track("track-1")
    events = monitor.check("track-1", "cam-1", BBox(4, 4, 6, 6), None, NOW)
    assert len(events) == 1
    assert events[0].event_type == "enter"


def test_tracks_are_independent():
    monitor = _monitor()
    events_a = monitor.check("track-a", "cam-1", BBox(4, 4, 6, 6), None, NOW)
    events_b = monitor.check("track-b", "cam-1", BBox(50, 50, 60, 60), None, NOW)
    assert len(events_a) == 1
    assert events_b == []


def _two_adjacent_zones() -> DbZoneMonitor:
    # Left half [0,10]x[0,10] and right half [10,20]x[0,10] -- share only the x=10 edge (zero
    # area), so a box straddling the boundary overlaps both with different areas depending on
    # exactly where it sits.
    left = ZoneRecord(
        zone_id="left", camera_id="cam-1", name="Left", zone_type="allowed",
        polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], triggers_login=False,
    )
    right = ZoneRecord(
        zone_id="right", camera_id="cam-1", name="Right", zone_type="allowed",
        polygon=[(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)], triggers_login=False,
    )
    return DbZoneMonitor("cam-1", zones=[left, right])


def test_box_spanning_two_zones_assigns_to_the_one_with_most_overlap():
    monitor = _two_adjacent_zones()
    # x in [2, 14]: 8 units (x=2..10) inside "left", 4 units (x=10..14) inside "right".
    events = monitor.check("track-1", "cam-1", BBox(2, 0, 14, 10), None, NOW)
    assert len(events) == 1
    assert events[0].zone_id == "left"


def test_box_spanning_two_zones_the_other_way():
    monitor = _two_adjacent_zones()
    # x in [6, 18]: 4 units (x=6..10) inside "left", 8 units (x=10..18) inside "right".
    events = monitor.check("track-1", "cam-1", BBox(6, 0, 18, 10), None, NOW)
    assert len(events) == 1
    assert events[0].zone_id == "right"


def test_moving_the_majority_overlap_from_one_zone_to_the_other_fires_exit_and_enter():
    monitor = _two_adjacent_zones()
    monitor.check("track-1", "cam-1", BBox(2, 0, 14, 10), None, NOW)  # mostly "left"
    events = monitor.check("track-1", "cam-1", BBox(6, 0, 18, 10), None, NOW)  # mostly "right" now
    assert {(e.zone_id, e.event_type) for e in events} == {("left", "exit"), ("right", "enter")}
