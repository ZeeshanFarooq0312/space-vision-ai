"""Background processing of an uploaded (pre-recorded) video file through Phase 1-3
(detection + recognition), writing the same annotated-recording + metadata format
live_stream.py writes for webcam sessions -- uploaded videos show up in the same /videos
list and Recordings UI, no separate view needed.

Exists for diagnosing recognition behavior against footage the live pipeline already
captured (or any other higher-quality/pre-recorded CCTV clip) without needing to
reproduce a live camera session: a fixed file can be re-processed, watched frame by frame,
and its log lines correlated with the annotated output on demand.

Tracking here uses the vendored TrackTrack core (tracking/local_tracker.TrackTrackLocalTracker)
rather than live_stream.py's BoT-SORT, since every upload comes from one fixed camera
(UPLOAD_CAMERA_ID below) -- BoT-SORT was chosen for the live path specifically to handle a
moving/handheld camera (see PersonDetector.track()'s docstring), which doesn't apply here.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import cv2

from visionstack.api.live_stream import (
    JPEG_QUALITY,
    PROCESSED_VIDEOS_DIR,
    _draw_detections,
    _draw_zones,
    _transcode_to_h264,
    build_face_crops,
    # verify_crops_batch,  # TODO(tracking-first): re-import when verification is turned back on
)
from visionstack.attendance.engine import DbAttendanceEngine
from visionstack.common.config import REPO_ROOT, get_settings
from visionstack.common.types import Detection, Frame
from visionstack.db.session import SessionLocal
from visionstack.detection.person_detector import PersonDetector
from visionstack.identity.body_embedder import OSNetBodyEmbedder
from visionstack.identity.face_detector import HaarFaceDetector
from visionstack.ingestion.frame_sampler import FrameSampler
from visionstack.ingestion.video_source import FileVideoSource
from visionstack.tracking.local_tracker import TrackTrackLocalTracker
from visionstack.tracking.reid_store import NoOpTrackReidStore
from visionstack.tracking.tracktrack.params import TrackTrackParams
from visionstack.zones.monitor import DbZoneMonitor

logger = logging.getLogger("visionstack.api.video_upload")

UPLOADS_DIR = REPO_ROOT / "data" / "video_uploads"
JobStatus = Literal["processing", "done", "error"]
# One physical camera for now (see api/live_stream.py's PersonDetector.track() docstring for the
# same assumption on the live side) -- standardized rather than derived per-upload-filename so a
# zone drawn once (see the preview-frame endpoint below + frontend ZoneDrawer reused against it)
# applies to every future upload, not just the one it was drawn against.
UPLOAD_CAMERA_ID = "upload-cam-1"
# Per-track retry cadence, in processed frames rather than wall-clock time: a file is processed
# as fast as the CPU/GPU allow (not paced to real time like a live session), so a time-based
# throttle would fire far more or less often than intended depending on machine speed. An
# unverified track is retried at most every this-many processed frames; once verified, never
# retried -- same per-track caching as live_stream.py, see _TrackState there.
VERIFY_RETRY_FRAMES = 8
# A track must be seen this many consecutive frames before its first verify attempt -- see
# live_stream.py's MIN_TRACK_FRAMES_BEFORE_VERIFY for why (filters ByteTrack's occasional 1-2
# frame false-positive tracks). TrackTrackLocalTracker already withholds a track until it has
# `min_len` (default 3) matched frames of its own, so this mostly just adds a small extra margin.
MIN_TRACK_FRAMES_BEFORE_VERIFY = 3
# Cadence (processed frames, same reasoning as VERIFY_RETRY_FRAMES) for refreshing an already-
# resolved track's stored embedding (see tracking/reid_store.py) -- keeps the appearance snapshot
# a later re-appearance would match against reasonably current, rather than frozen at whatever the
# person looked like in this track's first few frames (different pose/lighting/angle by minute 10
# of a long video).
REID_REMEMBER_FRAMES = 8


@dataclass
class _TrackState:
    """Per-track_id verification cache, frame-indexed (not wall-clock, see VERIFY_RETRY_FRAMES).
    Once verified, a track keeps its name for the rest of the video without re-hitting /verify.
    """

    name: str | None = None
    employee_id: str | None = None
    verified: bool = False
    seen_frames: int = 0
    # Starts eligible for verification as soon as seen_frames crosses its own threshold.
    last_attempt_frame: int = -VERIFY_RETRY_FRAMES
    # Resolved once (see reid_store.TrackReidStore.resolve) the first time this local track_id is
    # seen -- stable across this local track_id's lifetime, and equal to a *previous*, since-
    # dropped track_id's identity if the store recognized this as the same person returning.
    reid_identity_id: str | None = None
    last_remember_frame: int = -REID_REMEMBER_FRAMES


@dataclass
class _ZoneVisit:
    zone_id: str
    zone_name: str
    track_id: str
    reid_identity_id: str
    employee_id: str | None
    employee_name: str | None
    occurred_at: datetime
    crop_filename: str | None


def _extract_preview_frame(raw_path: Path) -> tuple[bytes, int, int] | None:
    """Grabs the uploaded video's first frame and JPEG-encodes it, so the zone-drawing UI
    (frontend/src/components/ZoneDrawer.tsx, reused as-is -- it only needs an <img> src, static
    or MJPEG both work) has something to draw a zone against before/independent of background
    processing. Returns (jpeg_bytes, width, height), or None if the file couldn't be read."""
    cap = cv2.VideoCapture(str(raw_path))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    height, width = frame.shape[:2]
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    return buf.tobytes(), width, height


def _crop_jpeg(image, bbox) -> bytes | None:
    h, w = image.shape[:2]
    x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
    x2, y2 = min(w, int(bbox.x2)), min(h, int(bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    ok, buf = cv2.imencode(".jpg", image[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


def _record_zone_visits(
    zone_monitor: DbZoneMonitor,
    attendance_engine: DbAttendanceEngine,
    track_states: dict[str, _TrackState],
    id_labels: dict[str, str],
    camera_id: str,
    frame: Frame,
    detections: list[Detection],
    crops_dir: Path,
    visits: list[_ZoneVisit],
) -> None:
    """For every detection with a track id, checks its bounding box against `camera_id`'s zones
    (whichever zone captures the most of the box wins when it spans more than one -- see
    DbZoneMonitor.check()).
    Every zone entry (regardless of whether the person has been identified yet) is recorded as a
    `_ZoneVisit` with a cropped image of that detection -- this is the "how many people in which
    zone, with their crop" result attached to each uploaded video (see _summarize_zone_visits and
    ProcessedVideo.zone_results). Separately, entries into a `triggers_login` zone by an already
    -verified track still fire AttendanceEngine.on_entry_match(), same as the live path.
    """
    for d in detections:
        if d.track_id is None:
            continue
        state = track_states.get(d.track_id)

        events = zone_monitor.check(d.track_id, camera_id, d.bbox, None, frame.timestamp)
        for event in events:
            if event.event_type != "enter":
                continue
            zone = zone_monitor.zone(event.zone_id)
            if zone is None:
                continue

            crop_filename: str | None = None
            crop_bytes = _crop_jpeg(frame.image, d.bbox)
            if crop_bytes is not None:
                crops_dir.mkdir(parents=True, exist_ok=True)
                crop_filename = f"{uuid.uuid4().hex}.jpg"
                (crops_dir / crop_filename).write_bytes(crop_bytes)

            visits.append(
                _ZoneVisit(
                    zone_id=zone.zone_id,
                    zone_name=zone.name,
                    # The stable reid-resolved display label ("#3"), not the raw local track_id --
                    # keeps the crop grid consistent with both the video overlay (_draw_detections'
                    # id_labels) and person_count (which dedups on reid_identity_id below). Falls
                    # back to the raw track_id if somehow never resolved/labeled.
                    track_id=id_labels.get(d.track_id, d.track_id),
                    # Falls back to the raw track_id if this track was never resolved (shouldn't
                    # happen once the main loop always resolves before calling this, but keeps
                    # _summarize_zone_visits' dedup from crashing on a None key either way).
                    reid_identity_id=(state.reid_identity_id if state and state.reid_identity_id else d.track_id),
                    employee_id=state.employee_id if state else None,
                    employee_name=state.name if state else None,
                    occurred_at=frame.timestamp,
                    crop_filename=crop_filename,
                )
            )

            if zone.triggers_login and state is not None and state.employee_id is not None:
                attendance_engine.on_entry_match(state.employee_id, camera_id, frame.timestamp)


def _summarize_zone_visits(visits: list[_ZoneVisit], video_id: str) -> list[dict]:
    """Groups raw visits by zone: how many distinct *people* (reid_identity_id, not raw track_id)
    were seen in each zone, plus every visit's crop for display. `crop_url` points at the
    zone-crop endpoint below.

    Deduping on reid_identity_id rather than track_id matters specifically because someone leaving
    frame and re-entering gets a brand new track_id from TrackTrackLocalTracker (no memory of the
    old one) -- without this, that single person inflated person_count once per re-entry (see
    tracking/reid_store.py for how reid_identity_id survives that gap).
    """
    by_zone: dict[str, dict] = {}
    for visit in visits:
        entry = by_zone.setdefault(
            visit.zone_id,
            {"zone_id": visit.zone_id, "zone_name": visit.zone_name, "identity_ids": set(), "visits": []},
        )
        entry["identity_ids"].add(visit.reid_identity_id)
        entry["visits"].append(
            {
                "track_id": visit.track_id,
                "employee_id": visit.employee_id,
                "employee_name": visit.employee_name,
                "occurred_at": visit.occurred_at.isoformat(),
                "crop_url": (
                    f"/api/videos/{video_id}/zone-crop/{visit.crop_filename}" if visit.crop_filename else None
                ),
            }
        )

    results = []
    for entry in by_zone.values():
        results.append(
            {
                "zone_id": entry["zone_id"],
                "zone_name": entry["zone_name"],
                "person_count": len(entry["identity_ids"]),
                "visits": entry["visits"],
            }
        )
    return results


@dataclass
class _Job:
    video_id: str
    original_filename: str
    status: JobStatus = "processing"
    frame_count: int = 0
    detection_count: int = 0
    max_people_in_frame: int = 0
    error: str | None = None
    preview_jpeg: bytes | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    latest_jpeg: bytes | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class VideoUploadProcessor:
    """Process-wide registry of upload-processing jobs, keyed by video_id."""

    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def start(self, raw_path: Path, original_filename: str, sample_fps: float) -> str:
        video_id = uuid.uuid4().hex
        job = _Job(video_id=video_id, original_filename=original_filename)
        preview = _extract_preview_frame(raw_path)
        if preview is not None:
            job.preview_jpeg, job.frame_width, job.frame_height = preview
        with self._lock:
            self._jobs[video_id] = job
        threading.Thread(
            target=self._process, args=(job, raw_path, sample_fps), daemon=True,
            name=f"upload-{video_id}",
        ).start()
        return video_id

    def preview_frame(self, video_id: str) -> bytes | None:
        job = self._jobs.get(video_id)
        return job.preview_jpeg if job else None

    def latest_frame(self, video_id: str) -> bytes | None:
        """Most recently processed annotated frame -- lets the frontend show a "live" preview of
        an in-progress upload the same way it does for a live camera session (see
        LiveStreamManager.latest_frame), even though processing isn't paced to real time."""
        job = self._jobs.get(video_id)
        if job is None:
            return None
        with job.lock:
            return job.latest_jpeg

    def mjpeg_frames(self, video_id: str, push_fps: float = 15.0):
        """Yields multipart/x-mixed-replace chunks for a StreamingResponse, same shape as
        LiveStreamManager.mjpeg_frames. Stops once the job is no longer 'processing' (or never
        existed) -- the frontend falls back to the static preview-frame / final video at that
        point."""
        interval = 1 / push_fps
        while True:
            frame = self.latest_frame(video_id)
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            job = self._jobs.get(video_id)
            if job is None or job.status != "processing":
                break
            time.sleep(interval)

    def status(self, video_id: str) -> dict | None:
        job = self._jobs.get(video_id)
        if job is None:
            return None
        return {
            "video_id": job.video_id,
            "status": job.status,
            "frame_count": job.frame_count,
            "detection_count": job.detection_count,
            "error": job.error,
            "frame_width": job.frame_width,
            "frame_height": job.frame_height,
        }

    def zone_crop_path(self, video_id: str, filename: str) -> Path | None:
        # Path(filename).name strips any directory components -- filename comes straight from a
        # URL path segment, so this is the guard against a ../ path-traversal attempt.
        path = PROCESSED_VIDEOS_DIR / f"{video_id}_zone_crops" / Path(filename).name
        return path if path.exists() else None

    def _process(self, job: _Job, raw_path: Path, sample_fps: float) -> None:
        PROCESSED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        working_mp4 = PROCESSED_VIDEOS_DIR / f"{job.video_id}_raw.mp4"
        output_mp4 = PROCESSED_VIDEOS_DIR / f"{job.video_id}.mp4"
        crops_dir = PROCESSED_VIDEOS_DIR / f"{job.video_id}_zone_crops"
        started_at = datetime.now(timezone.utc)
        all_recognized_names: set[str] = set()
        zone_visits: list[_ZoneVisit] = []
        writer: cv2.VideoWriter | None = None
        camera_id = UPLOAD_CAMERA_ID

        try:
            detector = PersonDetector(weights_path=get_settings().detection_weights_path, device="auto")
            face_detector = HaarFaceDetector()
            body_embedder = OSNetBodyEmbedder()
            # max_time_lost_seconds -- how long a track survives with no matching detection (e.g.
            # genuinely occluded behind a pillar, not just missed for a frame) before being dropped
            # and needing a brand-new id on return. Bumped 5 -> 10 minutes. Position is NOT frozen
            # indefinitely while Lost -- see tracktrack/params.py's position_freeze_speed_threshold
            # and lost_extrapolation_cutoff_frames docstrings for the current (speed-gated, then
            # cut off after ~2s) behavior -- so this budget isn't relying on a stale straight-line
            # extrapolation to still be accurate 10 minutes later. What actually makes a late
            # reappearance matchable at all is lost_search_growth_px_per_frame's growing search
            # radius (capped at 250px) plus appearance -- both bounded, so this larger window mainly
            # just keeps the track's identity (and its accumulated appearance feature) available in
            # memory for longer, it doesn't by itself make far-future geometric matches easier.
            # TEMPORARY (requested for a side-by-side comparison, not a tuning change): force
            # position_freeze_speed_threshold high enough that a Lost track is ALWAYS frozen in
            # place, regardless of its measured speed -- this is deliberately the pre-speed-gate
            # behavior (see that field's docstring, revision 1: "always frozen"), which real-
            # footage testing earlier this session showed breaks crossing-paths recovery (a moving
            # person occluded by someone else never catches up to where they actually reappear).
            # Revert by deleting this override block once the comparison is done -- the tuned
            # default (15.0, speed-gated) lives in tracktrack/params.py untouched.
            # _params = TrackTrackParams()
            # _params.position_freeze_speed_threshold = 1_000_000.0
            # local_tracker = TrackTrackLocalTracker(
            #     sample_fps=sample_fps, max_time_lost_seconds=600.0, params=_params
            # )
            local_tracker = TrackTrackLocalTracker(
               sample_fps=sample_fps, max_time_lost_seconds=600.0
              )
            # TODO(reid-model): PgVectorTrackReidStore's cross-gap matching is disabled again --
            # measured directly against real footage that osnet_x0_25_msmt17 (and the larger
            # osnet_x1_0_msmt17, also tested) don't reliably separate different people on this
            # camera (fisheye angle, similar office attire, moderate crop res): two confirmed-
            # different, simultaneously-visible people measured at cosine distance 0.208, while a
            # presumed-legitimate same-person cross-video match measured 0.48-0.52 -- LOWER
            # distance for different people than for the same one, so no threshold can separate
            # them. False-merging different real people into one identity is worse than the
            # original per-track fragmentation this was meant to fix, so defaulting back to
            # NoOpTrackReidStore (every track is its own identity, matches previous behavior)
            # until a body-ReID model is validated to actually discriminate this population.
            # PgVectorTrackReidStore itself, the content-relative clock, and the same-frame
            # co-occurrence exclusion are still correct and still tested -- swap this back in
            # once that model exists, nothing else needs to change.
            reid_store = NoOpTrackReidStore()
            zone_monitor = DbZoneMonitor(camera_id)
            attendance_engine = DbAttendanceEngine()
            video_source = FileVideoSource(path=str(raw_path), camera_id=camera_id)

            with video_source as source:
                sampler = FrameSampler(source, source_fps=source.fps, sample_fps=sample_fps)
                track_states: dict[str, _TrackState] = {}
                # reid_identity_id (a UUID) -> a small sequential display number, so the video
                # overlay and zone-visit labels read "#3" the same way the old raw-track_id labels
                # did, but for the STABLE identity, not TrackTrackLocalTracker's own fragmentation
                # -prone counter -- see _draw_detections' id_labels param below.
                identity_display_numbers: dict[str, int] = {}
                for processed_index, frame in enumerate(sampler):
                    # reid_store's age/gap windowing needs *video-content* time, not frame.timestamp
                    # (wall-clock datetime.now(), see ingestion/video_source.py) -- this file is
                    # processed as fast as the CPU/GPU allow, not paced to real time (same reasoning
                    # as VERIFY_RETRY_FRAMES above), so wall-clock deltas between frames reflect
                    # inference speed, not how much footage elapsed. Confirmed this was live: a
                    # 9.3s/96-frame video took ~4 real minutes to process (YOLO+OSNet on ~13
                    # people/frame), so min_gap_seconds=5 was satisfied after 2-3 frames of actual
                    # video content -- multiple still-seated, simultaneously-visible people were
                    # getting merged into one identity as a direct result. content_ts advances in
                    # lockstep with sample_fps instead, immune to however fast this machine runs.
                    content_ts = started_at + timedelta(seconds=processed_index / sample_fps)
                    raw_detections = detector.detect(frame)
                    embeddings = body_embedder.embed_batch(frame.image, [d.bbox for d in raw_detections])
                    for d, embedding in zip(raw_detections, embeddings):
                        d.embedding = embedding
                    detections = [t.detections[-1] for t in local_tracker.update(raw_detections)]
                    # Everyone detected in this exact frame -- passed to resolve() below so it can
                    # never match a new track against someone else who's simultaneously on screen
                    # right now, no matter how old their stored embedding rows are (see
                    # reid_store.py's currently_visible_track_ids docstring).
                    visible_track_ids = {d.track_id for d in detections if d.track_id is not None}

                    for d in detections:
                        if d.track_id is None:
                            continue
                        is_new = d.track_id not in track_states
                        state = track_states.setdefault(d.track_id, _TrackState())
                        state.seen_frames += 1
                        if is_new:
                            # First sight of this local track_id -- ask the vector store whether
                            # it recognizes this appearance from a recently-dropped track on this
                            # camera (see reid_store.py) before assuming it's a new person.
                            state.reid_identity_id = str(
                                reid_store.resolve(
                                    camera_id, d.track_id, d.embedding, content_ts, visible_track_ids
                                )
                            )
                            identity_display_numbers.setdefault(
                                state.reid_identity_id, len(identity_display_numbers) + 1
                            )
                            state.last_remember_frame = processed_index
                        elif processed_index - state.last_remember_frame >= REID_REMEMBER_FRAMES:
                            # Keep the stored embedding fresh so a *different* track's later
                            # resolve() call matches against a recent appearance, not this
                            # person's first few frames.
                            state.last_remember_frame = processed_index
                            reid_store.remember(
                                camera_id, uuid.UUID(state.reid_identity_id), d.track_id, d.embedding, content_ts
                            )

                    due = [
                        d
                        for d in detections
                        if d.track_id is not None
                        and not track_states[d.track_id].verified
                        and track_states[d.track_id].seen_frames >= MIN_TRACK_FRAMES_BEFORE_VERIFY
                        and processed_index - track_states[d.track_id].last_attempt_frame
                        >= VERIFY_RETRY_FRAMES
                    ]
                    if due:
                        for d in due:
                            track_states[d.track_id].last_attempt_frame = processed_index
                        # Face crops are still extracted (tracking is the current focus, not
                        # identity) -- see build_face_crops below -- but never sent to /verify.
                        # TODO(tracking-first): restore the commented block to re-enable
                        # verification. Every track stays unverified until then, so zone-based
                        # login (which requires employee_id) won't fire either -- see
                        # _record_zone_visits below, which still records the zone visit itself
                        # (with employee_id/employee_name left None) regardless.
                        _face_crops = build_face_crops(due, frame.image, face_detector)  # noqa: F841
                        # results = verify_crops_batch(_face_crops, context=camera_id)
                        # for d, result in zip(due, results):
                        #     if result is None:
                        #         continue
                        #     employee_id, name = result
                        #     state = track_states[d.track_id]
                        #     state.name = name
                        #     state.employee_id = employee_id
                        #     state.verified = True
                        #     all_recognized_names.add(name)

                    id_labels = {
                        tid: f"#{identity_display_numbers[s.reid_identity_id]}"
                        for tid, s in track_states.items()
                        if s.reid_identity_id in identity_display_numbers
                    }
                    _record_zone_visits(
                        zone_monitor, attendance_engine, track_states, id_labels, camera_id, frame,
                        detections, crops_dir, zone_visits,
                    )

                    track_names = {tid: s.name for tid, s in track_states.items() if s.name}
                    annotated = _draw_zones(frame.image.copy(), zone_monitor.zones)
                    annotated = _draw_detections(annotated, detections, track_names, id_labels)

                    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if ok:
                        with job.lock:
                            job.latest_jpeg = buf.tobytes()

                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            str(working_mp4), cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (w, h)
                        )
                    writer.write(annotated)

                    job.frame_count += 1
                    job.detection_count += len(detections)
                    job.max_people_in_frame = max(job.max_people_in_frame, len(detections))
        except Exception as e:  # noqa: BLE001 -- surface via status() instead of a silent thread death
            logger.exception("upload processing failed for '%s'", job.original_filename)
            job.status = "error"
            job.error = str(e)
            if writer is not None:
                writer.release()
            working_mp4.unlink(missing_ok=True)
            raw_path.unlink(missing_ok=True)
            return

        if writer is not None:
            writer.release()
        raw_path.unlink(missing_ok=True)

        if job.frame_count == 0:
            working_mp4.unlink(missing_ok=True)
            job.status = "error"
            job.error = "No frames could be read from the uploaded video."
            return

        playable = _transcode_to_h264(working_mp4, output_mp4)
        if not playable:
            working_mp4.replace(output_mp4)

        ended_at = datetime.now(timezone.utc)
        metadata = {
            "video_id": job.video_id,
            "camera_id": camera_id,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": (ended_at - started_at).total_seconds(),
            "frame_count": job.frame_count,
            "detection_count": job.detection_count,
            "max_people_in_frame": job.max_people_in_frame,
            "filename": output_mp4.name,
            "browser_playable": playable,
            "recognized_names": sorted(all_recognized_names),
            "zone_results": _summarize_zone_visits(zone_visits, job.video_id),
        }
        output_mp4.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        job.status = "done"


video_upload_processor = VideoUploadProcessor()
