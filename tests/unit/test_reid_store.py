"""PgVectorTrackReidStore against the real Postgres/pgvector -- there's no fake/in-memory
substitute for pgvector's cosine_distance query, so unlike most of this test suite this one needs
a live DB; skips (not fails) if it's unreachable, same spirit as conftest.sample_video_path.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy.exc import OperationalError

from visionstack.db.models import TrackEmbedding
from visionstack.db.session import SessionLocal
from visionstack.tracking.reid_store import NoOpTrackReidStore, PgVectorTrackReidStore

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def camera_id():
    # Unique per test run so this never collides with / pollutes real data in the shared DB.
    cam_id = f"test-reid-cam-{uuid.uuid4().hex[:8]}"
    try:
        with SessionLocal() as db:
            db.execute(TrackEmbedding.__table__.select().limit(1))
    except OperationalError:
        pytest.skip("Postgres not reachable")
    yield cam_id
    with SessionLocal() as db:
        db.query(TrackEmbedding).filter(TrackEmbedding.camera_id == cam_id).delete()
        db.commit()


def _vec(seed: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed * 1000))
    v = rng.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def _nudge(vec: np.ndarray, amount: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(1)
    nudged = vec + rng.normal(scale=amount, size=vec.shape).astype(np.float32)
    return nudged / np.linalg.norm(nudged)


def test_reappearing_track_within_window_reuses_the_identity(camera_id):
    store = PgVectorTrackReidStore(session_factory=SessionLocal, max_age_seconds=300.0)
    original = _vec(1.0)

    first_id = store.resolve(camera_id, "local-1", original, NOW)
    # A brand new local track_id, e.g. after the person left frame and came back -- but a near-
    # identical appearance embedding, within the age window.
    second_id = store.resolve(camera_id, "local-99", _nudge(original), NOW + timedelta(seconds=30))

    assert second_id == first_id


def test_a_different_person_gets_a_different_identity(camera_id):
    store = PgVectorTrackReidStore(session_factory=SessionLocal, max_age_seconds=300.0)

    first_id = store.resolve(camera_id, "local-1", _vec(1.0), NOW)
    second_id = store.resolve(camera_id, "local-2", _vec(99.0), NOW + timedelta(seconds=5))

    assert second_id != first_id


def test_two_people_simultaneously_visible_are_never_merged_even_with_near_identical_appearance(camera_id):
    # Regression test for a real bug: two different, *simultaneously present* local track_ids
    # (same frame -> same timestamp) got resolved to the same identity because nothing excluded
    # a just-written same-instant candidate. Two people on screen at once can never be the same
    # person, regardless of how close their embeddings are (near-identical office attire, say).
    store = PgVectorTrackReidStore(session_factory=SessionLocal, max_age_seconds=300.0)
    base = _vec(1.0)

    first_id = store.resolve(camera_id, "local-1", base, NOW)
    second_id = store.resolve(camera_id, "local-2", _nudge(base, amount=0.01), NOW)

    assert second_id != first_id


def test_a_continuously_active_track_is_never_matched_by_someone_appearing_while_it_is_still_visible(
    camera_id,
):
    # Regression test for a second, subtler real bug: min_gap_seconds alone only checks how old a
    # *stored row* is, not whether the local track it belongs to is still currently on screen. A
    # track that's been active for a while (regularly "remembered") has rows old enough to clear
    # any gap threshold even though the person never left -- a later-appearing, genuinely
    # different person could still match against one of those stale-but-not-gone rows unless the
    # currently-visible set is passed and honored.
    store = PgVectorTrackReidStore(session_factory=SessionLocal, max_age_seconds=300.0, min_gap_seconds=5.0)
    person_a = _vec(1.0)
    person_b = _nudge(person_a, amount=0.01)  # deliberately near-identical appearance

    # person A appears at t=0 and stays continuously visible/refreshed.
    a_id = store.resolve(camera_id, "local-a", person_a, NOW, currently_visible_track_ids={"local-a"})
    store.remember(camera_id, a_id, "local-a", person_a, NOW + timedelta(seconds=4))

    # person B first appears at t=9s, while A is *still visible in the same frame* -- A's very
    # first row (t=0) is >5s old (clears the naive gap check) but A never actually left.
    b_id = store.resolve(
        camera_id,
        "local-b",
        person_b,
        NOW + timedelta(seconds=9),
        currently_visible_track_ids={"local-a", "local-b"},
    )

    assert b_id != a_id


def test_reappearing_after_the_age_window_gets_a_new_identity(camera_id):
    store = PgVectorTrackReidStore(session_factory=SessionLocal, max_age_seconds=60.0)
    original = _vec(1.0)

    first_id = store.resolve(camera_id, "local-1", original, NOW)
    later_id = store.resolve(
        camera_id, "local-2", _nudge(original), NOW + timedelta(seconds=120)
    )

    assert later_id != first_id


def test_remember_refreshes_the_matchable_window_without_minting_a_new_identity(camera_id):
    store = PgVectorTrackReidStore(session_factory=SessionLocal, max_age_seconds=60.0)
    original = _vec(1.0)

    first_id = store.resolve(camera_id, "local-1", original, NOW)
    store.remember(camera_id, first_id, "local-1", original, NOW + timedelta(seconds=45))

    # Without the remember() call above, this would be 105s past the *first* observation (outside
    # a 60s window) -- but only 60s past the refreshed one.
    later_id = store.resolve(
        camera_id, "local-2", _nudge(original), NOW + timedelta(seconds=105)
    )

    assert later_id == first_id


def test_noop_store_never_reuses_an_identity():
    store = NoOpTrackReidStore()
    a = store.resolve("cam-x", "local-1", _vec(1.0), NOW)
    b = store.resolve("cam-x", "local-2", _vec(1.0), NOW)
    assert a != b
