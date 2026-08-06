from datetime import datetime, timezone

from visionstack.attendance.engine import NoOpAttendanceEngine
from visionstack.attendance.presence_window import NoOpPresenceConfirmationWindow


def test_noop_attendance_engine_returns_none_and_empty_list():
    engine = NoOpAttendanceEngine()
    now = datetime.now(timezone.utc)
    assert engine.on_entry_match("emp-1", "entry-cam-1", now) is None
    assert engine.on_exit_match("emp-1", "exit-cam-1", now) is None
    assert engine.run_end_of_day_auto_logout(now) == []


def test_noop_presence_window_never_confirms():
    window = NoOpPresenceConfirmationWindow()
    now = datetime.now(timezone.utc)
    assert window.observe("track-1", "exit-cam-1", now) is False
