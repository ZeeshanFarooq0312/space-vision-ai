"""Cross-gap track re-identification: resolves a fresh local track_id back to the same stable
identity after an absence longer than TrackTrackLocalTracker's own in-memory occlusion budget
(TrackTrackParams.max_time_lost -- a few seconds), instead of relying on in-process state that's
gone the moment a track is dropped. Backed by pgvector (see db/models.TrackEmbedding) rather than
an in-memory ring buffer, so "how long can we remember someone" is a query window (max_age_seconds
below), not bounded by how many frames happen to still be resident in a live Python object.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np

from visionstack.db.camera_helpers import ensure_camera
from visionstack.db.models import TrackEmbedding


class TrackReidStore(Protocol):
    def resolve(
        self,
        camera_id: str,
        local_track_id: str,
        embedding: np.ndarray,
        ts: datetime,
        currently_visible_track_ids: set[str] = frozenset(),
    ) -> uuid.UUID:
        """Called once when a local track_id is first seen. Returns a stable identity id -- either
        a recent match's id (found within the store's age/similarity window) or a freshly minted
        one -- and persists this observation so future calls can match against it.

        `currently_visible_track_ids` MUST include every local track_id detected in the same frame
        as this call (this one included is harmless -- see PgVectorTrackReidStore's own
        local_track_id != local_track_id exclusion) -- an age/gap window alone isn't enough to
        guarantee two different, simultaneously-visible people never get merged: a track that's
        been continuously active for a while has old embedding rows that clear any reasonable gap
        threshold even though the person never left. Two people on screen at the same instant can
        never be the same person, full stop, so anyone else visible in this exact frame must be
        excluded from matching regardless of how old their stored rows are.
        """
        ...

    def remember(
        self,
        camera_id: str,
        reid_identity_id: uuid.UUID,
        local_track_id: str,
        embedding: np.ndarray,
        ts: datetime,
    ) -> None:
        """Persists an additional embedding snapshot under an already-resolved identity, keeping
        its appearance signature fresh for later resolve() calls without re-querying or minting a
        new id. Call periodically for an ongoing track, not every frame -- see
        api/video_upload.py's REID_REMEMBER_FRAMES."""
        ...


class NoOpTrackReidStore:
    """Every track is its own identity, nothing persisted -- previous behavior (no cross-gap
    re-identification), kept as the default for callers that don't want a DB dependency (tests,
    pipeline/orchestrator.py's demo path)."""

    def resolve(
        self,
        camera_id: str,
        local_track_id: str,
        embedding: np.ndarray,
        ts: datetime,
        currently_visible_track_ids: set[str] = frozenset(),
    ) -> uuid.UUID:
        return uuid.uuid4()

    def remember(
        self,
        camera_id: str,
        reid_identity_id: uuid.UUID,
        local_track_id: str,
        embedding: np.ndarray,
        ts: datetime,
    ) -> None:
        return None


class PgVectorTrackReidStore:
    """Real implementation. On resolve(): looks for the closest embedding captured on the same
    camera within `max_age_seconds`; reuses its reid_identity_id if within
    `match_distance_threshold` (pgvector's `cosine_distance`, 0=identical, up to 2=opposite),
    otherwise mints a new one. Either way, persists this observation.

    match_distance_threshold reuses IdentityConfig.body_match_threshold (0.45) -- the one existing
    number in this codebase for "how close is close enough" on a body embedding, even though it
    was written for real employee-identity matching (identity/matcher.py, still a stub), not
    anonymous track continuity -- no track-specific number has been validated yet, so this is a
    reasonable starting point rather than an invented one. Revisit empirically once there's real
    reappearance footage to tune against (watch for: two different people getting merged into one
    identity -- threshold too loose; the same person not being recognized on return --
    too strict).

    `min_gap_seconds` matters as much as the threshold, found the hard way: without it, resolve()
    could match against an embedding written moments (even the same frame) ago by a *different,
    simultaneously-visible* person -- confirmed directly against real footage, three distinct
    local track_ids present in the identical processed frame (same captured_at to the microsecond)
    all resolving to the same reid_identity_id, which is only possible if the match query is
    reaching into the current instant instead of a genuine past absence. Two people on screen at
    once can never be the same person, so a candidate has to be older than this floor to be
    considered at all -- set comfortably above TrackTrackParams.max_time_lost's occlusion budget
    (a few seconds) so the two mechanisms don't overlap: the local Kalman tracker owns short gaps,
    this store owns everything past that.

    That alone still isn't sufficient, also found the hard way on real footage: a track that's been
    continuously active for a while accumulates old embedding rows (from earlier remember() calls)
    that clear min_gap_seconds even though the person never left -- a second, later-appearing person
    can then match against that stale-but-not-stale-enough row. `currently_visible_track_ids` closes
    this: every local track_id detected in the SAME frame as the resolve() call is excluded from
    matching outright, regardless of how old its rows are, since simultaneous presence is a hard
    guarantee of being different people.
    """

    def __init__(
        self,
        session_factory,
        max_age_seconds: float = 300.0,
        match_distance_threshold: float = 0.45,
        min_gap_seconds: float = 5.0,
    ) -> None:
        self._session_factory = session_factory
        self._max_age_seconds = max_age_seconds
        self._match_distance_threshold = match_distance_threshold
        self._min_gap_seconds = min_gap_seconds

    def resolve(
        self,
        camera_id: str,
        local_track_id: str,
        embedding: np.ndarray,
        ts: datetime,
        currently_visible_track_ids: set[str] = frozenset(),
    ) -> uuid.UUID:
        vector = embedding.tolist()
        excluded_track_ids = currently_visible_track_ids | {local_track_id}
        with self._session_factory() as db:
            distance_expr = TrackEmbedding.embedding.cosine_distance(vector).label("distance")
            cutoff = ts - timedelta(seconds=self._max_age_seconds)
            gap_floor = ts - timedelta(seconds=self._min_gap_seconds)
            row = (
                db.query(TrackEmbedding, distance_expr)
                .filter(
                    TrackEmbedding.camera_id == camera_id,
                    TrackEmbedding.captured_at >= cutoff,
                    TrackEmbedding.captured_at <= gap_floor,
                    TrackEmbedding.local_track_id.notin_(excluded_track_ids),
                )
                .order_by(distance_expr)
                .first()
            )
            if row is not None and row[1] is not None and row[1] <= self._match_distance_threshold:
                reid_identity_id = row[0].reid_identity_id
            else:
                reid_identity_id = uuid.uuid4()

            # track_embeddings.camera_id is FK'd to cameras.camera_id, and nothing else in this
            # pipeline guarantees a Camera row exists for an ad-hoc id like "upload-cam-1" (see
            # camera_helpers.ensure_camera's own docstring) -- without this, the very first
            # resolve() call on a camera with no zone ever created against it would hit an FK
            # violation.
            ensure_camera(db, camera_id)
            db.add(
                TrackEmbedding(
                    camera_id=camera_id,
                    reid_identity_id=reid_identity_id,
                    local_track_id=local_track_id,
                    embedding=vector,
                    captured_at=ts,
                )
            )
            db.commit()
            return reid_identity_id

    def remember(
        self,
        camera_id: str,
        reid_identity_id: uuid.UUID,
        local_track_id: str,
        embedding: np.ndarray,
        ts: datetime,
    ) -> None:
        with self._session_factory() as db:
            ensure_camera(db, camera_id)
            db.add(
                TrackEmbedding(
                    camera_id=camera_id,
                    reid_identity_id=reid_identity_id,
                    local_track_id=local_track_id,
                    embedding=embedding.tolist(),
                    captured_at=ts,
                )
            )
            db.commit()
