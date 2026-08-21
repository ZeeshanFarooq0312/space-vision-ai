"""Cost functions + greedy iterative matching. Vendored from TrackTrack (kamkyu94/TrackTrack,
trackers/utils.py, MIT license), with one change: dropped the unused `lap`-based
`linear_assignment` helper -- upstream's own Tracker never calls it (superseded by `associate` +
the iterative loop below, left in as dead code with a comment marking it disabled).
"""
from __future__ import annotations

import numpy as np


def bbox_overlaps(a_x1y1x2y2: np.ndarray, b_x1y1x2y2: np.ndarray) -> np.ndarray:
    num_a = a_x1y1x2y2.shape[0]
    num_b = b_x1y1x2y2.shape[0]
    overlaps = np.zeros((num_a, num_b))

    for n_b in range(num_b):
        box_area = (b_x1y1x2y2[n_b, 2] - b_x1y1x2y2[n_b, 0] + 1) * (b_x1y1x2y2[n_b, 3] - b_x1y1x2y2[n_b, 1] + 1)
        for n_a in range(num_a):
            iw = min(a_x1y1x2y2[n_a, 2], b_x1y1x2y2[n_b, 2]) - max(a_x1y1x2y2[n_a, 0], b_x1y1x2y2[n_b, 0]) + 1
            if iw > 0:
                ih = min(a_x1y1x2y2[n_a, 3], b_x1y1x2y2[n_b, 3]) - max(a_x1y1x2y2[n_a, 1], b_x1y1x2y2[n_b, 1]) + 1
                if ih > 0:
                    ua = (
                        (a_x1y1x2y2[n_a, 2] - a_x1y1x2y2[n_a, 0] + 1) * (a_x1y1x2y2[n_a, 3] - a_x1y1x2y2[n_a, 1] + 1)
                        + box_area
                        - iw * ih
                    )
                    overlaps[n_a, n_b] = iw * ih / ua
    return overlaps


def find_deleted_detections_mask(dets: np.ndarray, dets_95: np.ndarray) -> np.ndarray:
    """Boolean mask over dets_95: True where a dets_95 row has no near-duplicate (IoU >= 0.97) in
    dets, i.e. it was "deleted" by whatever NMS/threshold produced `dets` and should be recovered
    as a fallback candidate. Returns a mask (rather than upstream's filtered array) so the caller
    can apply the same selection to a parallel `refs` list -- see tracker.py.
    """
    # Check
    if len(dets) == 0 or len(dets_95) == 0:
        return np.ones(len(dets_95), dtype=bool)

    # Get boxes
    a_x1y1x2y2 = np.ascontiguousarray(dets[:, :4], dtype=np.float64)
    b_x1y1x2y2 = np.ascontiguousarray(dets_95[:, :4], dtype=np.float64)

    # Calculate IoU
    ious = bbox_overlaps(a_x1y1x2y2, b_x1y1x2y2)

    return np.max(ious, axis=0) < 0.97


def iou_distance(a_tracks: list, b_tracks: list) -> tuple[np.ndarray, np.ndarray]:
    # Get boxes
    a_boxes = np.ascontiguousarray([track.x1y1x2y2 for track in a_tracks], dtype=np.float64)
    b_boxes = np.ascontiguousarray([track.x1y1x2y2 for track in b_tracks], dtype=np.float64)

    # Calculate IoU distance
    if len(a_boxes) == 0 or len(b_boxes) == 0:
        iou_sim = np.zeros((len(a_boxes), len(b_boxes)), dtype=np.float64)
        iou_dist = 1 - iou_sim
    else:
        # Calculate HIoU
        h_iou = np.minimum(a_boxes[:, 3:4], b_boxes[:, 3:4].T) - np.maximum(a_boxes[:, 1:2], b_boxes[:, 1:2].T)
        h_iou /= np.maximum(a_boxes[:, 3:4], b_boxes[:, 3:4].T) - np.minimum(a_boxes[:, 1:2], b_boxes[:, 1:2].T)

        # Calculate HMIoU
        iou_sim = bbox_overlaps(a_boxes, b_boxes)
        iou_dist = 1 - h_iou * iou_sim

    return iou_sim, iou_dist


def cos_distance(tracks: list, dets: list) -> np.ndarray:
    # Check
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Calculate cosine distance
    t_feat = np.concatenate([t.feat for t in tracks], axis=0)
    d_feat = np.concatenate([d.feat for d in dets], axis=0)
    cos_dist = np.clip(1 - np.dot(t_feat, d_feat.T), a_min=0.0, a_max=1.0)

    return cos_dist


def conf_distance(tracks: list, dets: list) -> np.ndarray:
    # Check
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Get previous scores
    t_score_prev = []
    for t in tracks:
        frame_ids = sorted(t.history.keys(), reverse=True)
        frame_id = frame_ids[min(1, len(frame_ids) - 1)]
        t_score_prev.append(t.history[frame_id][1])

    # Linear projection
    t_score_prev = np.array(t_score_prev)
    t_score = np.array([t.score for t in tracks])
    t_score += t_score - t_score_prev

    # Calculate confidence similarity
    d_score = np.array([d.score for d in dets])
    conf_dist = np.abs(t_score[:, None] - d_score[None, :])

    return conf_dist


def get_prev_box(history: dict, frame_id: int, dt: int) -> np.ndarray:
    # Try
    target_key = frame_id - dt
    if target_key in history:
        return history[target_key][0]

    # If there are no recent observation
    return history[max(history.keys())][0]


def get_vel_t_d(b_1: np.ndarray, b_2: np.ndarray) -> np.ndarray:
    # Expand boxes
    b_1, b_2 = b_1[:, np.newaxis, :], b_2[np.newaxis, :, :]

    # Get normalization factors
    deltas = b_2 - b_1
    norm_lt = np.sqrt(deltas[:, :, 0:1] ** 2 + deltas[:, :, 1:2] ** 2) + 1e-5
    norm_lb = np.sqrt(deltas[:, :, 0:1] ** 2 + deltas[:, :, 3:4] ** 2) + 1e-5
    norm_rt = np.sqrt(deltas[:, :, 2:3] ** 2 + deltas[:, :, 1:2] ** 2) + 1e-5
    norm_rb = np.sqrt(deltas[:, :, 2:3] ** 2 + deltas[:, :, 3:4] ** 2) + 1e-5

    # Get velocities
    vel_lt = np.stack([b_2[:, :, 0] - b_1[:, :, 0], b_2[:, :, 1] - b_1[:, :, 1]], axis=-1) / norm_lt
    vel_lb = np.stack([b_2[:, :, 0] - b_1[:, :, 0], b_2[:, :, 3] - b_1[:, :, 3]], axis=-1) / norm_lb
    vel_rt = np.stack([b_2[:, :, 2] - b_1[:, :, 2], b_2[:, :, 1] - b_1[:, :, 1]], axis=-1) / norm_rt
    vel_rb = np.stack([b_2[:, :, 2] - b_1[:, :, 2], b_2[:, :, 3] - b_1[:, :, 3]], axis=-1) / norm_rb

    return np.stack([vel_lt, vel_lb, vel_rt, vel_rb], axis=2)


def calc_angle(vel_t: np.ndarray, vel_t_d: np.ndarray) -> np.ndarray:
    angle_ = 0
    for vdx in range(vel_t.shape[2]):
        # Divide & Repeat
        vel_t_x = np.repeat(vel_t[:, :, vdx, 0], vel_t_d.shape[1], axis=1)
        vel_t_y = np.repeat(vel_t[:, :, vdx, 1], vel_t_d.shape[1], axis=1)

        # Calculate angle, Normalize to range (0 ~ 1)
        angle = vel_t_x * vel_t_d[:, :, vdx, 0] + vel_t_y * vel_t_d[:, :, vdx, 1]
        angle = np.abs(np.arccos(np.clip(angle, a_min=-1, a_max=1))) / np.pi
        angle_ += angle / 4

    return angle_


def angle_distance(tracks: list, dets: list, frame_id: int, d_t: int = 3) -> np.ndarray:
    # Initialization
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float64)

    # Get velocity between track and detections
    track_boxes = np.stack([get_prev_box(t.history, frame_id, d_t) for t in tracks], axis=0)
    vel_t_d = get_vel_t_d(track_boxes, np.stack([d.x1y1x2y2 for d in dets], axis=0))

    # Get angle distance
    angle_dist = calc_angle(np.stack([t.velocity for t in tracks], axis=0)[:, np.newaxis], vel_t_d)

    # Fuse score
    scores = np.array([d.score for d in dets])[np.newaxis, :]
    angle_dist *= scores

    return angle_dist


def associate(cost: np.ndarray, match_thr: float) -> list[list[int]]:
    # Initialization
    matches = []

    # Run
    if cost.shape[0] > 0 and cost.shape[1] > 0:
        # Get index for minimum similarity
        min_ddx = np.argmin(cost, axis=1)
        min_tdx = np.argmin(cost, axis=0)

        # Match tracks with detections
        for tdx, ddx in enumerate(min_ddx):
            if min_tdx[ddx] == tdx and cost[tdx, ddx] < match_thr:
                matches.append([tdx, ddx])

    return matches


def iterative_assignment(
    tracks: list,
    dets_high: list,
    dets_low: list,
    dets_del_high: list,
    match_thr: float,
    penalty_p: float,
    penalty_q: float,
    reduce_step: float,
    frame_id: int,
    d_t: int = 3,
    iou_weight: float = 0.5,
    appearance_weight: float = 0.5,
) -> tuple[list[list[int]], list[int], list[int]]:
    # Initialization
    matches: list[list[int]] = []
    dets = dets_high + dets_low + dets_del_high

    # Calculate preliminaries
    iou_sim, iou_dist = iou_distance(tracks, dets)
    cos_dist = cos_distance(tracks, dets)

    # Calculate cost. iou_weight/appearance_weight default to upstream's fixed 0.5/0.5 split, but
    # the caller (Tracker, via TrackTrackParams) can shift the balance -- see that class's
    # docstring for why the project currently runs with appearance_weight=0 (no real body-ReID
    # embedder wired in yet, see identity/body_embedder.NoOpBodyEmbedder).
    cost = iou_weight * iou_dist + appearance_weight * cos_dist
    cost += 0.10 * conf_distance(tracks, dets) + 0.05 * angle_distance(tracks, dets, frame_id, d_t)

    # Give penalty
    cost[:, len(dets_high) : len(dets_high + dets_low)] += penalty_p
    cost[:, len(dets_high + dets_low) :] += penalty_q

    # Constraint: a pair with essentially no box overlap (iou_sim<=0.10) is normally unmatchable
    # regardless of how good its other terms are -- upstream's own hard backstop against a wildly
    # wrong match. But that backstop was found (via real footage, see
    # params.lost_search_growth_px_per_frame's docstring) to also block *correct* matches for a
    # track that's been genuinely unmatched long enough that its predicted box has drifted well
    # away from where the person actually reappeared -- ordinary path deviation during a multi-
    # second occlusion, not a tracker bug. So the 0.10 floor is escaped -- not lowered, not
    # removed, just escaped on a per-pair basis -- when the candidate detection's center falls
    # within that specific track's own growing search_pad_px radius (0 for a healthily-Tracked
    # track, so this changes nothing for the vast majority of pairs). Deliberately a *distance*
    # check on centers, not a box-inflation trick: inflating the boxes themselves before computing
    # IoU would corrupt the IoU value used in the weighted cost above too (a huge padded box
    # divided by a small real overlap tanks the ratio even for a dead-on match) -- confirmed by
    # hand, an earlier version of this fix did exactly that and broke the crossing-paths case that
    # was already working. Once escaped, the pair's cost still comes from the ordinary weighted
    # formula above (iou_sim itself stays genuinely ~0 for a real-distance pair, so iou_weight's
    # 0.7 contribution stays high) -- appearance/confidence/angle still have to actually agree for
    # match_thr to clear, this only stops zero-IoU from being an automatic, appearance-blind veto.
    a_centers = np.array([[(t.x1y1x2y2[0] + t.x1y1x2y2[2]) / 2, (t.x1y1x2y2[1] + t.x1y1x2y2[3]) / 2] for t in tracks])
    d_centers = np.array([[(d.x1y1x2y2[0] + d.x1y1x2y2[2]) / 2, (d.x1y1x2y2[1] + d.x1y1x2y2[3]) / 2] for d in dets])
    if len(tracks) and len(dets):
        center_dist = np.linalg.norm(a_centers[:, None, :] - d_centers[None, :, :], axis=2)
        search_radius = np.array([getattr(t, "search_pad_px", 0.0) for t in tracks])[:, None]
        escaped = center_dist <= search_radius
    else:
        escaped = np.zeros_like(iou_sim, dtype=bool)
    cost[(iou_sim <= 0.10) & ~escaped] = 1.0

    # Clip
    cost = np.clip(cost, 0, 1)

    # Match
    while True:
        # Match tracks with detections
        matches_ = associate(cost, match_thr)
        match_thr -= reduce_step

        # Check (if there are no more matchable pairs)
        if len(matches_) == 0:
            break

        # Append
        matches += matches_

        # Update cost matrix
        for t, d in matches:
            cost[t, :] = 1.0
            cost[:, d] = 1.0

    # Find indices of unmatched tracks and detections
    m_tracks = [t for t, _ in matches]
    u_tracks = [t for t in range(len(tracks)) if t not in m_tracks]
    m_dets = [d for _, d in matches]
    u_dets = [d for d in range(len(dets)) if d not in m_dets]

    return matches, u_tracks, u_dets


def track_aware_nms(
    pair_sims: np.ndarray, scores: np.ndarray, num_tracks: int, nms_thresh: float, score_thresh: float
) -> np.ndarray:
    # Initialization
    num_dets = len(pair_sims) - num_tracks
    allow_indices = np.ones(num_dets) * (scores > score_thresh)

    # Run
    for idx in range(num_dets):
        # Check 1
        if allow_indices[idx] == 0:
            continue

        # Check 2
        if num_tracks > 0:
            if np.max(pair_sims[num_tracks + idx, :num_tracks]) > nms_thresh:
                allow_indices[idx] = 0
                continue

        # Check 3
        for jdx in range(num_dets):
            if idx != jdx and allow_indices[jdx] == 1 and scores[idx] > scores[jdx]:
                if pair_sims[num_tracks + idx, num_tracks + jdx] > nms_thresh:
                    allow_indices[jdx] = 0

    return allow_indices == 1
