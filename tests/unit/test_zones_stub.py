from datetime import datetime, timezone

from visionstack.zones.monitor import NoOpZoneMonitor
from visionstack.zones.zone_config import Zone, ZoneRegistry


def test_zone_registry_loads_repo_zones_config():
    registry = ZoneRegistry()
    zones = registry.all()
    assert len(zones) >= 1
    assert all(isinstance(z, Zone) for z in zones)


def test_zone_contains_point_inside_polygon():
    zone = Zone(
        zone_id="z1",
        camera_id="cam-1",
        name="Test Zone",
        zone_type="restricted",
        polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        allowed_roles=["it_admin"],
    )
    assert zone.contains((5, 5)) is True
    assert zone.contains((50, 50)) is False


def test_zone_is_authorised_by_role():
    zone = Zone(
        zone_id="z1",
        camera_id="cam-1",
        name="Test Zone",
        zone_type="restricted",
        polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        allowed_roles=["it_admin"],
    )
    assert zone.is_authorised("it_admin") is True
    assert zone.is_authorised("intern") is False
    assert zone.is_authorised(None) is False


def test_allowed_zone_authorises_everyone():
    zone = Zone(
        zone_id="z1",
        camera_id="cam-1",
        name="Lobby",
        zone_type="allowed",
        polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        allowed_roles=[],
    )
    assert zone.is_authorised(None) is True
    assert zone.is_authorised("anyone") is True


def test_noop_zone_monitor_never_emits_events():
    monitor = NoOpZoneMonitor()
    result = monitor.check("track-1", "cam-1", (5, 5), "it_admin", datetime.now(timezone.utc))
    assert result is None
