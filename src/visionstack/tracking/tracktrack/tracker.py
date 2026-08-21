"""Frame-by-frame tracker: two-stage (high/low confidence) association against existing tracks,
track-aware NMS to seed new ones, camera-motion-compensated Kalman prediction in between.

Adapted from TrackTrack (kamkyu94/TrackTrack, trackers/tracker.py, MIT license). Changes from
upstream:

- Takes a `TrackTrackParams` instead of an argparse Namespace, and an `IdentityCMC` (see cmc.py)
  instead of upstream's per-benchmark-video GMC-file reader.
- `update()` takes optional `refs`/`refs_95` lists (parallel to `dets`/`dets_95`) and returns each
  surviving track paired with the caller-supplied `ref` it most recently matched, rather than
  upstream's own `TTTrack` objects alone -- see track.py's docstring for why (this pipeline needs
  to recover its own `Detection` objects from the tracker's output).
- No `vid_name` / dataset-name plumbing (upstream used it only to pick the right GMC file).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from visionstack.tracking.tracktrack.association import (
    find_deleted_detections_mask,
    iou_distance,
    iterative_assignment,
    track_aware_nms,
)
from visionstack.tracking.tracktrack.cmc import IdentityCMC, apply_cmc
from visionstack.tracking.tracktrack.params import TrackTrackParams
from visionstack.tracking.tracktrack.track import TrackCounter, TrackState, TTTrack


class Tracker:
    def __init__(self, params: TrackTrackParams) -> None:
        self.params = params
        self.max_time_lost = params.max_time_lost

        self.tracks: list[TTTrack] = []
        self.frame_id = 0
        self.counter = TrackCounter()

        self.cmc = IdentityCMC()

    def init_tracks(self, dets: list[TTTrack]) -> None:
        # Get alive tracks, iou_similarity, and scores
        alive_states = (TrackState.Tracked, TrackState.New)
        tracks = [t for t in self.tracks if t.state in alive_states]
        iou_sim = iou_distance(tracks + dets, tracks + dets)[0]
        scores = np.array([d.score for d in dets])

        # Run track aware NMS
        allow_indices = track_aware_nms(
            iou_sim, scores, len(tracks), self.params.tai_thr, self.params.init_thr
        )

        for idx, flag in enumerate(allow_indices):
            if flag:
                dets[idx].initiate(self.frame_id, self.counter)
                self.tracks.append(dets[idx])

    def update(
        self,
        dets: np.ndarray,
        dets_95: np.ndarray | None = None,
        refs: list[Any] | None = None,
        refs_95: list[Any] | None = None,
    ) -> list[TTTrack]:
        # Update frame id
        self.frame_id += 1

        if len(dets) == 0:
            return self.update_without_detections()

        refs = list(refs) if refs is not None else [None] * len(dets)
        if dets_95 is None or len(dets_95) == 0:
            dets_del, refs_del = np.empty((0, dets.shape[1])), []
        else:
            refs_95 = list(refs_95) if refs_95 is not None else [None] * len(dets_95)
            mask = find_deleted_detections_mask(dets, dets_95)
            dets_del = dets_95[mask]
            refs_del = [r for r, keep in zip(refs_95, mask) if keep]

        # Get deleted detections & Encode
        dets = [TTTrack(self.params, d, r) for d, r in zip(dets, refs)]
        dets_del = [TTTrack(self.params, d, r) for d, r in zip(dets_del, refs_del)]

        # Divide detections
        dets_high = [d for d in dets if d.score > self.params.det_thr]
        dets_low = [d for d in dets if d.score <= self.params.det_thr]
        dets_del_high = [d for d in dets_del if d.score > self.params.det_thr]

        # Split tracks
        active_states = (TrackState.Tracked, TrackState.Lost)
        tracked_lost = [t for t in self.tracks if t.state in active_states]
        new = [t for t in self.tracks if t.state == TrackState.New]

        # Camera motion compensation
        warp_matrix = self.cmc.get_warp_matrix()
        apply_cmc(tracked_lost, warp_matrix)
        apply_cmc(new, warp_matrix)

        # Predict the current location with KF
        [t.predict() for t in tracked_lost]
        [t.predict() for t in new]

        # Association between (tracked and lost tracks) & (high confidence detections)
        dets = dets_high + dets_low + dets_del_high
        matches, u_tracks, u_dets = iterative_assignment(
            tracked_lost,
            dets_high,
            dets_low,
            dets_del_high,
            self.params.match_thr,
            self.params.penalty_p,
            self.params.penalty_q,
            self.params.reduce_step,
            self.frame_id,
            iou_weight=self.params.iou_weight,
            appearance_weight=self.params.appearance_weight,
        )

        # Update matched tracks
        for t, d in matches:
            tracked_lost[t].update(self.frame_id, dets[d])

        # Mark "lost" to unmatched tracks
        for t in u_tracks:
            tracked_lost[t].mark_lost()

        # Get remained high confidence detections
        dets_high_left = [dets[i] for i in u_dets if i < len(dets_high)]

        # Association between (new tracks) & (left high confidence detections)
        matches, u_tracks, u_dets = iterative_assignment(
            new,
            dets_high_left,
            [],
            [],
            self.params.match_thr,
            self.params.penalty_p,
            self.params.penalty_q,
            self.params.reduce_step,
            self.frame_id,
            iou_weight=self.params.iou_weight,
            appearance_weight=self.params.appearance_weight,
        )

        # Update matched tracks
        for t, d in matches:
            new[t].update(self.frame_id, dets_high_left[d])

        # Mark "remove" to unmatched tracks
        for t in u_tracks:
            new[t].mark_removed()

        # Mark "remove" lost tracks which are too old and add to finished
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Filter out the removed tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        # Init new tracks
        self.init_tracks([dets_high_left[udx] for udx in u_dets])

        return [t for t in self.tracks if t.state == TrackState.Tracked]

    def update_without_detections(self) -> list[TTTrack]:
        # Only maintain already tracked and new tracks, Drop all the new tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.New]

        # Camera motion compensation
        warp_matrix = self.cmc.get_warp_matrix()
        apply_cmc(self.tracks, warp_matrix)

        # Predict the current location with KF
        [t.predict() for t in self.tracks]

        # Change every track as lost tracks
        for t in self.tracks:
            t.mark_lost()

        # Mark "remove" to lost tracks which are too old
        for track in self.tracks:
            if self.frame_id - track.end_frame_id > self.max_time_lost:
                track.mark_removed()

        # Filter out the removed tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.Removed]

        return []
