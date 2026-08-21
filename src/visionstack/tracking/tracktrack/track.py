"""Per-track state (box, Kalman mean/covariance, appearance feature, velocity history). Adapted
from TrackTrack (kamkyu94/TrackTrack, trackers/track.py, MIT license). Changes from upstream:

- Renamed `Track` -> `TTTrack` (this codebase's `visionstack.common.types.Track` is a different,
  unrelated dataclass; the two must not collide on import).
- Takes a `TrackTrackParams` instead of an argparse Namespace, and a `ref` passthrough (arbitrary
  caller object, e.g. the originating `visionstack.common.types.Detection`) so the caller can
  recover which of its own objects each surviving track most recently matched -- upstream never
  needed this since it goes straight from tracker output to a MOT-format results file.
- `predict()`'s dataset check (`'Dance' in self.args.data_path`) became the explicit
  `params.zero_wh_velocity_when_not_tracked` flag (upstream's own w/h-only scope) for width/
  height. Center-position velocity went through three revisions before landing on a per-track
  speed gate (`params.position_freeze_speed_threshold`) -- see that field's docstring for the full
  history: always-frozen (fixed a stationary-drift bug, broke crossing paths), always-unfrozen
  (fixed crossing, measurably worsened general fragmentation on real footage), now gated on the
  KF's own last speed estimate so each track gets whichever behavior actually fits it.
- `predict()` also grows `search_pad_px` the longer a track goes unmatched -- a separate, later
  fix for a failure mode the speed gate above doesn't cover: a genuinely-moving, correctly-
  unfrozen track whose real path deviates from straight-line extrapolation enough during a multi-
  second occlusion (a normal turn/slow-down, not a bug) to land at literal zero IoU with its
  prediction, hard-gated out of matching regardless of appearance. See
  `params.lost_search_growth_px_per_frame`'s docstring for the real-footage measurement that
  motivated it, and `association.iou_distance` for where the padding is actually applied.
- `update_features()` guards against a zero appearance feature (norm-dividing a real zero vector
  is a NaN that then poisons every future cosine-distance calculation for that track). This
  matters here specifically because `identity.body_embedder.BodyEmbedder` is a stub
  (`NoOpBodyEmbedder`) returning an all-zero vector until Phase 3's real embedder lands -- upstream
  never hits this because FastReID always produces a real unit vector.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from visionstack.tracking.tracktrack.association import get_prev_box
from visionstack.tracking.tracktrack.kalman_filter import KalmanFilter
from visionstack.tracking.tracktrack.params import TrackTrackParams

# `history` only ever needs to look back `delta_t` (3) frames for velocity (see update() below) or
# 1-2 frames for association.conf_distance's score projection -- but was never pruned upstream, so
# a track alive for a long upload accumulated one full entry (box + Kalman mean/covariance +
# feature snapshot) per frame for its entire lifetime, unbounded. 20 is a generous margin above
# what's actually read (delta_t=3, plus slack for get_prev_box's fallback-to-oldest-available
# behavior when a track has gaps), not a tuned value -- the point is bounding memory to a small
# constant per track, not this exact number.
HISTORY_MAX_ENTRIES = 20


def get_vel(b_1: np.ndarray, b_2: np.ndarray) -> np.ndarray:
    # Get normalization factors
    deltas = b_2 - b_1
    norm_lt = np.sqrt(deltas[0] ** 2 + deltas[1] ** 2) + 1e-5
    norm_lb = np.sqrt(deltas[0] ** 2 + deltas[3] ** 2) + 1e-5
    norm_rt = np.sqrt(deltas[2] ** 2 + deltas[1] ** 2) + 1e-5
    norm_rb = np.sqrt(deltas[2] ** 2 + deltas[3] ** 2) + 1e-5

    # Get velocities
    vel_lt = np.array([b_2[0] - b_1[0], b_2[1] - b_1[1]]) / norm_lt
    vel_lb = np.array([b_2[0] - b_1[0], b_2[3] - b_1[3]]) / norm_lb
    vel_rt = np.array([b_2[2] - b_1[2], b_2[1] - b_1[1]]) / norm_rt
    vel_rb = np.array([b_2[2] - b_1[2], b_2[3] - b_1[3]]) / norm_rb

    return np.stack([vel_lt, vel_lb, vel_rt, vel_rb], axis=0)


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class TrackCounter:
    def __init__(self) -> None:
        self.track_count = 0

    def get_track_id(self) -> int:
        self.track_count += 1
        return self.track_count


class BaseTrack:
    track_id = 0
    end_frame_id = 0
    state = TrackState.New

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


class TTTrack(BaseTrack):
    def __init__(self, params: TrackTrackParams, detection: np.ndarray, ref: Any = None) -> None:
        # Initialize 1
        self.params = params
        self.box = detection[:4]  # x1y1x2y2
        self.score = detection[4]
        self.ref = ref

        # Initialize 2
        self.delta_t = 3
        self.history: dict[int, list] = {}
        self.kalman_filter: KalmanFilter | None = None
        self.mean, self.covariance = None, None
        self.velocity = np.zeros((4, 2))
        # See params.lost_search_growth_px_per_frame's docstring: how many predict() cycles since
        # this track's last real detection update, and the resulting match-time-only search
        # padding grown from it. Both reset to 0 on every successful update().
        self.frames_since_matched = 0
        self.search_pad_px = 0.0

        # Initialize 3
        self.alpha = params.feature_alpha
        self.feat = detection[6:][np.newaxis, :].copy()

    def update_features(self, feat: np.ndarray, score: float) -> None:
        # Update and normalize
        beta = self.alpha + (1 - self.alpha) * (1 - score)
        self.feat = beta * self.feat + (1 - beta) * feat
        norm = np.linalg.norm(self.feat)
        if norm > 1e-12:
            self.feat /= norm

    def initiate(self, frame_id: int, counter: TrackCounter) -> None:
        # Get new track id
        self.track_id = counter.get_track_id()

        # Initiate Kalman filter
        self.kalman_filter = KalmanFilter()
        self.mean, self.covariance = self.kalman_filter.initiate(self.cxcywh.copy())

        # Initiate history
        self.history[frame_id] = [
            self.box.copy(),
            self.score.copy(),
            self.mean.copy(),
            self.covariance.copy(),
            self.feat.copy(),
        ]

        # Initiate parameters
        self.end_frame_id = frame_id
        self.state = TrackState.New

    def predict(self) -> None:
        # How long (in predict() cycles, i.e. sampled frames) since this track's last real
        # detection update -- drives both the extrapolation cutoff and search_pad_px below. Reset
        # to 0 in update(); incremented here first so both use this call's up-to-date count.
        if self.state != TrackState.Tracked:
            self.frames_since_matched += 1

        # Freeze width/height velocity while lost or new -- prevents the box from growing/
        # shrinking unboundedly during an extended occlusion.
        if self.state != TrackState.Tracked and self.params.zero_wh_velocity_when_not_tracked:
            self.mean[6] = 0
            self.mean[7] = 0

        # Center-position velocity: freeze it if the KF's own last estimate is slow (see
        # position_freeze_speed_threshold's docstring for the full history/tradeoff and why a
        # per-track speed gate, not a blanket freeze/unfreeze, is what actually resolves it) OR if
        # this track has been unmatched too long to keep trusting that estimate at all (see
        # lost_extrapolation_cutoff_frames' docstring -- real people don't hold one heading for an
        # entire multi-second occlusion, so a "genuinely moving" track's extrapolation is only
        # trustworthy for a bounded window before it's just as likely to be wrong as frozen).
        if self.state != TrackState.Tracked:
            speed = float(np.hypot(self.mean[4], self.mean[5]))
            past_cutoff = self.frames_since_matched > self.params.lost_extrapolation_cutoff_frames
            if speed < self.params.position_freeze_speed_threshold or past_cutoff:
                self.mean[4] = 0
                self.mean[5] = 0

        # Predict
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

        # Grow the match-time-only search padding (see params.lost_search_growth_px_per_frame's
        # docstring) the longer this track goes without a real detection update -- 0 for a
        # healthily-Tracked track, since frames_since_matched only increments above while Lost/New.
        self.search_pad_px = min(
            self.params.lost_search_max_px,
            self.params.lost_search_growth_px_per_frame * self.frames_since_matched,
        )

    def update(self, frame_id: int, detection: "TTTrack") -> None:
        # Update Kalman filter & Feature
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, detection.cxcywh.copy(), detection.score
        )
        self.update_features(detection.feat.copy(), detection.score)
        self.frames_since_matched = 0
        self.search_pad_px = 0.0

        # Update history
        self.history[frame_id] = [
            detection.box.copy(),
            detection.score,
            self.mean.copy(),
            self.covariance.copy(),
            self.feat.copy(),
        ]

        # Update velocity
        self.velocity = np.zeros((4, 2))
        for d_t in range(1, self.delta_t + 1):
            prev_box = get_prev_box(self.history, frame_id, d_t).copy()
            self.velocity += get_vel(prev_box, detection.x1y1x2y2) / d_t
        self.velocity /= self.delta_t

        # Bound memory: nothing ever reads further back than HISTORY_MAX_ENTRIES (see its
        # docstring) -- get_prev_box's only fallback is the *most recent* key, never the oldest,
        # so dropping old entries here is safe.
        if len(self.history) > HISTORY_MAX_ENTRIES:
            for old_key in sorted(self.history.keys())[: len(self.history) - HISTORY_MAX_ENTRIES]:
                del self.history[old_key]

        # Update parameters
        self.box = detection.box.copy()
        self.score = detection.score
        self.ref = detection.ref
        self.end_frame_id = frame_id
        is_established = len(self.history) >= self.params.min_len
        self.state = TrackState.Tracked if is_established else TrackState.New

    @property
    def cxcywh(self) -> np.ndarray:
        # Get current position in bounding box format `(center x, center y, width, height)`.
        if self.mean is None:
            cx = (self.box[0] + self.box[2]) / 2
            cy = (self.box[1] + self.box[3]) / 2
            w = self.box[2] - self.box[0]
            h = self.box[3] - self.box[1]
        else:
            cx = self.mean[0]
            cy = self.mean[1]
            w = self.mean[2]
            h = self.mean[3]

        return np.array([cx, cy, w, h])

    @property
    def x1y1wh(self) -> np.ndarray:
        if self.mean is None:
            x1 = self.box[0]
            y1 = self.box[1]
            w = self.box[2] - self.box[0]
            h = self.box[3] - self.box[1]
        else:
            x1 = self.mean[0] - self.mean[2] / 2
            y1 = self.mean[1] - self.mean[3] / 2
            w = self.mean[2]
            h = self.mean[3]

        return np.array([x1, y1, w, h])

    @property
    def x1y1x2y2(self) -> np.ndarray:
        if self.mean is None:
            x1 = self.box[0]
            y1 = self.box[1]
            x2 = self.box[2]
            y2 = self.box[3]
        else:
            x1 = self.mean[0] - self.mean[2] / 2
            y1 = self.mean[1] - self.mean[3] / 2
            x2 = self.mean[0] + self.mean[2] / 2
            y2 = self.mean[1] + self.mean[3] / 2

        return np.array([x1, y1, x2, y2])
