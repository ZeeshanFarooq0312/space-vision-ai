"""Tuning knobs for the vendored TrackTrack tracker.

Upstream (kamkyu94/TrackTrack) hardcodes a different set of these per benchmark (MOT17/MOT20/
DanceTrack, see its utils/etc.py:set_parameters) with dataset-specific det/init/match thresholds
tuned against each benchmark's ground truth. None of those apply to an arbitrary CCTV camera, so
this defaults to the closest upstream analogue -- DanceTrack's generic (non-per-sequence) baseline
-- rather than any MOT17/20 sequence-specific tuning.

`appearance_weight` was 0.0 for most of this project's life -- a deliberate departure from
upstream's fixed 0.5/0.5 IoU/appearance split, found by diagnosing a stationary/slow-moving person
in an uploaded video getting a new track id every few frames (Track #1 -> #99 -> #107 -> #112 ->
#453 -> #731 for one person sitting at a desk). Root cause: `identity/body_embedder.NoOpBodyEmbedder`
(Phase 3 was still a stub) fed TrackTrack an all-zero appearance vector for every detection, so
cosine distance between any two detections was a constant ~1.0 (maximally "different") regardless
of whether they were the same person -- appearance_weight=0.5 was therefore adding a constant ~0.5
penalty to every single match's cost on top of genuine motion cost, silently requiring near-perfect
IoU (~0.6+) to clear match_thr=0.7 at all.

Now that `identity/body_embedder.OSNetBodyEmbedder` (real OSNet ReID, via boxmot) is wired into
video_upload.py, appearance_weight=0.3 restores a real (non-zero-vector) appearance term -- see
that field's own docstring below for how the value was picked.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackTrackParams:
    # Detections above this score are "high confidence" for the first association pass;
    # at/below it, they're "low confidence" and only matched against tracks the high-conf pass
    # left unmatched (ByteTrack-style two-stage association). Only high-confidence detections can
    # ever seed a brand-new track (see init_thr/init_tracks) -- a detection stuck below det_thr on
    # every single frame can extend an existing track but can never create one, no matter how many
    # frames it's genuinely, correctly detected on.
    #
    # Lowered from upstream's 0.6 after measuring real confidence output from stock yolo12m.pt on
    # this project's actual (crowded, wide/fisheye-angle) office footage: traced a single frame
    # with 14 real people detected at [0.88, 0.81, 0.76, 0.74, 0.56, 0.49, 0.48, 0.46, 0.36, 0.33,
    # 0.32, 0.24, 0.23, 0.21] -- only 4/14 cleared 0.6, so 10 real, correctly-detected people were
    # permanently invisible (never became a track) on every single frame, not just a slow warm-up.
    # 0.3 recovers 11/14 in that same sample. Revisit if this model/camera combo changes.
    det_thr: float = 0.3
    # Minimum score for a leftover high-confidence detection to seed a brand-new track. Kept equal
    # to det_thr (upstream's own convention) -- see that field's docstring for why it moved.
    init_thr: float = 0.3
    # Cost ceiling (0-1, lower is a better match) for the greedy iterative association. Raised
    # from upstream's 0.7 after tracing real per-track fragmentation on this project's own
    # footage: every track that actually died (vs. recovering from a brief Lost) belonged to a
    # marginal, low-confidence, often-partially-occluded person (score hovering 0.18-0.30) whose
    # match cost crept just over 0.7 during a real reappearance -- not a wrong-direction match,
    # just a genuine same-person match that scored slightly too expensive. Measured on a 240-frame
    # (60s) real-footage window with ~13 continuously-present people: match_thr=0.7 produced 20
    # distinct track ids; 0.95 produced 16, with per-frame recall improving too (12.44 -> 12.86
    # avg tracked/frame) -- i.e. this recovers genuine matches, it doesn't just paper over misses.
    # The iou_sim<=0.10 hard gate in association.iterative_assignment (zero real overlap -> cost
    # forced to 1.0, unmatchable regardless of match_thr) is the remaining backstop against a
    # wildly wrong match, so this isn't as unbounded as the raw number suggests.
    match_thr: float = 0.95
    # IoU ceiling for track-aware NMS when seeding new tracks from leftover detections.
    tai_thr: float = 0.55
    # Extra cost added to low-confidence detections when matching (discourages weak matches).
    # Lowered from upstream's 0.20 alongside match_thr -- see that field's docstring for the same
    # measurement (0.05 was part of the tested combination that produced the improvement).
    penalty_p: float = 0.05
    # Extra cost added to "recovered" (see association.find_deleted_detections) detections.
    penalty_q: float = 0.40
    # How much match_thr relaxes each pass of the iterative greedy-matching loop.
    reduce_step: float = 0.05
    # A track must accumulate this many updates before it's reported as "Tracked" (filters
    # single-frame false-positive detections from ever surfacing as a track).
    min_len: int = 3
    # EMA smoothing factor for the appearance-embedding running average kept per track.
    feature_alpha: float = 0.80
    # Frames a track may go unmatched before being dropped, converted from
    # max_time_lost_seconds * sample_fps by the caller (see local_tracker.TrackTrackLocalTracker).
    max_time_lost: int = 30
    # Freeze width/height Kalman velocity (upstream's own scope for this flag) while a track is
    # New/Lost, i.e. not currently confirmed by a real detection -- stops the box growing/shrinking
    # unboundedly during an extended occlusion. See position_freeze_speed_threshold below for the
    # separate, more involved story on center-position velocity.
    zero_wh_velocity_when_not_tracked: bool = True
    # Center-position velocity: frozen while Lost/New IF the KF's own last speed estimate
    # (hypot(mean[4], mean[5]), real pixels/sampled-frame) is below this. Three revisions to get
    # here, each measured against real footage/synthetic repros, not assumed:
    #   1. Always frozen -- fixed a real bug (seated person's box center drifting a few px/frame
    #      from pure detector height noise; a single noisy jump got attributed to KF velocity, and
    #      with no correction while Lost, predict() extrapolated it in a straight line until the
    #      track was unrecoverable -- measured IoU decay 0.50 -> 0.0 within ~10 unmatched frames).
    #      Cost: a person walking steadily who got briefly occluded (e.g. crossing paths with
    #      someone else) could pick up a second id, since the frozen position never advanced to
    #      meet them on return.
    #   2. Always unfrozen -- tried once match_thr moved 0.7 -> 0.95 (see that field's docstring):
    #      the looser cost ceiling seemed to absorb the original drift on its own in a 60s test
    #      (38/40 seeds still held one id; 20/20 crossing cases fixed, both speeds). But a longer,
    #      more realistic 150s/4-segment real-footage test told a different story: always-unfrozen
    #      measured 15-40% MORE distinct ids than always-frozen, consistently across every segment
    #      (28 vs 20, 20 vs 18, 15 vs 14, 19 vs 16) -- the 60s test was just too short to see it.
    #   3. Speed-gated (this): the real per-track distinguishing signal was never "is match_thr
    #      loose enough" -- it's "is this specific track's own velocity estimate small (genuinely
    #      near-stationary, freeze is safe and helpful) or large (genuinely walking, freezing
    #      breaks their reappearance)". Note self.velocity (track.py) is NOT usable for this gate
    #      -- it's a per-corner UNIT direction vector with no magnitude information, confirmed by
    #      getting identical results at every threshold before switching to the KF's own mean[4:6].
    #      15 px/sampled-frame chosen from a joint sweep: on the same 150s/4-segment real-footage
    #      test, thresholds 8-20 all passed both synthetic repros perfectly (0/20 crossing failures
    #      at two speeds, same 2/40 stationary-edge-case seeds as every prior revision -- pre-
    #      existing and unrelated to this gate) while 15 landed closest to always-frozen's id counts
    #      (22/18/14/17 vs frozen's 20/18/14/16) -- i.e. keeps stage-1's fragmentation win almost
    #      entirely while still fully fixing stage-2's crossing case.
    position_freeze_speed_threshold: float = 15.0
    # Association cost weights (must sum to <=1 alongside the fixed 0.10 conf_distance + 0.05
    # angle_distance terms in association.iterative_assignment -- upstream's own split). See this
    # module's docstring for the appearance_weight=0 era and why it ended.
    #
    # 0.3, not upstream's 0.5: re-ran this project's own raw-vs-tracked recovery measurement (see
    # det_thr's docstring for the method) with real OSNet embeddings across appearance_weight in
    # [0.0, 0.2, 0.3, 0.5] on the same real footage, using the custom fine-tuned detector -- all
    # four gave identical avg_tracked/distinct_ids (12.2 tracked of 13.9 raw, 13 ids). That's
    # expected, not a sign appearance doesn't matter: with the custom detector, boxes are stable
    # and high-confidence enough that IoU alone already resolves nearly every match in this
    # sample, so this count-based metric can't see appearance's actual job, which is disambiguating
    # id-swaps when two people cross paths or occlude each other -- a scenario this metric doesn't
    # exercise and un-annotated real footage can't easily measure either. Picked 0.3 (not upstream's
    # 0.5) to keep IoU/motion as the dominant signal given that's what's actually been validated
    # here, while giving appearance real (non-zero) weight for exactly the crossing/occlusion cases
    # it exists for. Revisit if id-swaps are observed in practice -- that's the failure mode raising
    # this further would target.
    iou_weight: float = 0.7
    appearance_weight: float = 0.3
    # association.iterative_assignment hard-gates any match at iou_sim<=0.10 to cost=1.0
    # (unmatchable), regardless of how good the appearance/confidence/angle terms are -- see that
    # function's own comment. That's correct for a track that's been Lost only a frame or two (its
    # straight-line Kalman prediction is still trustworthy), but real footage showed it's the
    # actual cause of ids that "should" reconnect but don't: measured directly on the reference
    # office video (15min, 27 distinct track ids), 3 tracks went Lost then a fresh id appeared
    # within 1.6-6s at 74-191px away -- e.g. one person was moving at 32px/sampled-frame right
    # before occlusion (so position_freeze_speed_threshold correctly did NOT freeze them), yet
    # 1.6s (13 frames) later they reappeared only 191px away, far short of the ~416px a straight-
    # line extrapolation at 32px/frame would predict -- i.e. they slowed or changed direction
    # while occluded (behind a pillar/desk, out of a doorway at an angle -- ordinary human motion,
    # not a tracker bug), which is enough deviation for iou_sim to hit exactly 0 and get hard-
    # gated regardless of appearance. This is NOT the same failure as the crossing-paths bug
    # position_freeze_speed_threshold fixes (that one was measured recovering correctly in 762/764
    # real overlap-then-Lost events, >99% -- this is a distinct, longer-gap failure mode).
    #
    # Fix: escape the 0.10 hard gate, per (track, detection) pair, when the detection's center
    # falls within this specific track's own growing search radius (TTTrack.search_pad_px) -- NOT
    # by inflating the boxes fed into the IoU/cost formula itself. That was tried first and reverted
    # by hand after it broke the already-working crossing-paths case: padding a box directly changes
    # its own area, and IoU is an area ratio, so a big padded box divided by a small real overlap
    # tanks the score even for an otherwise dead-on match -- confirmed by tracing the crossing
    # regression test's cost matrix, where a near-perfect (~1.0 unpadded) match dropped to ~0.07
    # once padded and got wrongly rejected. Using a separate center-distance check for the gate
    # escape only, while leaving iou_distance's actual IoU computation untouched, avoids that: a
    # pair that escapes the gate this way still gets its real (near-zero) iou_sim in the weighted
    # cost above, so appearance/confidence/angle still have to genuinely agree for match_thr to
    # clear -- this only stops zero-IoU from being an automatic, appearance-blind veto. Leaning on
    # appearance ALONE (i.e. dropping the geometric constraint entirely) was deliberately not done
    # instead: this project's own OSNet measurements found different real people scoring as more
    # "similar" (cosine distance 0.208) than a legitimate same-person cross-video match (0.48-0.52)
    # on this camera -- see video_upload.py's reid_store TODO -- so a match still has to be
    # *somewhere plausible*, not just look right. Raised from an initial 15 to 20px/sampled-frame
    # after the first deployed value only closed 2 of 3 real swap cases: the third (32px/frame
    # mover, unmatched only 13 frames -- short enough that lost_extrapolation_cutoff_frames never
    # even engages, so this is purely about the radius keeping pace with a genuinely fast walker)
    # needed >=17.3px/frame (225px gap / 13 frames) of growth to close; 20 covers it with margin
    # while the slower real case (74px over 48 frames) needed far less either way. Capped at 250px
    # so a long-Lost track can't out-compete an unrelated bystander merely by being old enough --
    # Tracked tracks reset this to 0 every frame they're matched, so this only ever activates for
    # genuinely Lost/unconfirmed tracks.
    lost_search_growth_px_per_frame: float = 20.0
    lost_search_max_px: float = 250.0
    # Companion to the above: sustaining a track's last-measured velocity indefinitely (what
    # position_freeze_speed_threshold's "moving" branch does) is only trustworthy for a couple of
    # seconds -- real people don't hold a fixed heading for the length of an entire occlusion.
    # Checked directly against the same real-footage case that motivated lost_search_growth_px_per_
    # frame above: a track moving 15.8px/sampled-frame (just above position_freeze_speed_threshold,
    # so not frozen) stayed unmatched for 48 frames -- straight-line extrapolation the whole way
    # would land ~758px from start, but the person had only actually moved 74px in that time. No
    # sane search-radius cap bridges a 684px miss, so past this many frames unmatched, position
    # extrapolation freezes (same effect as the speed gate's "frozen" branch) regardless of the
    # track's speed at the moment it was lost -- the growing search radius above then does the
    # work of finding the real reappearance point from wherever extrapolation stopped, instead of
    # the runaway prediction being the search center. 16 frames (~2s at this project's 8fps
    # default) chosen to sit past the crossing-paths window that position_freeze_speed_threshold's
    # 762/764 real-footage validation covers (occlusions in that measurement were consistently
    # shorter), so short crossings still get the full benefit of extrapolation.
    lost_extrapolation_cutoff_frames: int = 16
