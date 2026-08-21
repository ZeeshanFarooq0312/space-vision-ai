"""Camera motion compensation (CMC): warps each track's Kalman mean/covariance by the estimated
inter-frame camera motion before prediction, so panning/handheld motion doesn't get mistaken for
every tracked person suddenly moving.

Adapted from TrackTrack (kamkyu94/TrackTrack, trackers/cmc.py, MIT license). Upstream's `CMC`
class reads a precomputed per-benchmark-video GMC file (one line of affine-warp coefficients per
frame, generated offline for each MOT17/MOT20/DanceTrack sequence) -- there's no such file for an
arbitrary live camera, and the repo doesn't include a live estimator either. `IdentityCMC` below
is a no-op stand-in (always the identity transform, i.e. "assume the camera doesn't move") so the
call site stays wired for a future real implementation (e.g. an OpenCV ECC/optical-flow estimator
run against consecutive frames) without changing `Tracker`'s structure. See
`visionstack.detection.person_detector.PersonDetector.track` for why real CMC matters for a
moving/handheld camera -- BoT-SORT (used there) has one built in; this vendored tracker does not,
so avoid it for anything but a fixed camera mount until a live estimator is added here.
"""
from __future__ import annotations

import numpy as np


class IdentityCMC:
    def get_warp_matrix(self) -> np.ndarray:
        return np.eye(2, 3, dtype=np.float64)


def apply_cmc(tracks: list, warp_matrix: np.ndarray = np.eye(2, 3)) -> int:
    # Check
    if len(tracks) == 0:
        return 0

    # Get mean, covariance
    multi_mean = np.asarray([t.mean.copy() for t in tracks])
    multi_covariance = np.asarray([t.covariance for t in tracks])

    # Get warp matrix
    rot = warp_matrix[:, :2]
    rot_8x8 = np.kron(np.eye(4, dtype=float), rot)
    trans = warp_matrix[:, 2]

    # Warp
    for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
        mean = rot_8x8 @ mean
        mean[:2] += trans
        cov = rot_8x8 @ cov @ rot_8x8.T

        tracks[i].mean = mean
        tracks[i].covariance = cov

    return 0
