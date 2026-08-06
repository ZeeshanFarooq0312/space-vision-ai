from datetime import date, datetime, timezone

from visionstack.reporting.attendance_reports import NoOpAttendanceReportGenerator
from visionstack.reporting.dwell_time import NoOpDwellTimeReportGenerator
from visionstack.reporting.trajectory_replay import NoOpTrajectoryReplay


def test_noop_attendance_report_generator_returns_empty_list():
    assert NoOpAttendanceReportGenerator().daily_report(date.today()) == []


def test_noop_dwell_time_report_generator_returns_empty_summary():
    now = datetime.now(timezone.utc)
    summary = NoOpDwellTimeReportGenerator().summary("zone-1", now, now)
    assert summary.zone_id == "zone-1"
    assert summary.total_seconds_by_employee == {}


def test_noop_trajectory_replay_returns_empty_list():
    now = datetime.now(timezone.utc)
    assert NoOpTrajectoryReplay().query("zone-1", now, now) == []
