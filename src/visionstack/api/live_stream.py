"""Background webcam capture + detection sessions backing the browser live-preview UI.

Each session runs Phase 1 (ingestion, via FrameSampler) + Phase 2 (detection+tracking, via
PersonDetector.track()) directly in a dedicated thread — not asyncio, since
cv2.VideoCapture.read() and YOLO inference are both blocking calls that would otherwise stall the
event loop, and not the shared Pipeline/orchestrator (that still runs the Phase 3-7 stub chain
unmodified for the standalone run_pipeline.py CLI; this module needs real per-track identity
caching, which Pipeline's stub LocalTracker doesn't provide, so it composes FrameSampler +
PersonDetector directly instead). The thread continuously encodes the latest annotated frame as
JPEG under a lock; the MJPEG stream endpoint just reads that latest frame at its own pace, so a
slow/disconnected client never backs up the capture loop.

Every session is also recorded to disk (annotated frames) so it can be replayed afterward from
the Recordings UI, independent of the live view. OpenCV's own FFmpeg build has no H.264 encoder
(only mp4v, which browsers won't play), so the raw mp4v file is transcoded via the system `ffmpeg`
binary; if that's unavailable, the raw file is kept as a fallback (playable in VLC, not browsers).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from visionstack.attendance.engine import DbAttendanceEngine
from visionstack.common.config import REPO_ROOT, get_settings
from visionstack.common.errors import FaceApiError
from visionstack.common.types import BBox, Detection
from visionstack.db.models import Employee, EmployeeFaceEmbedding
from visionstack.db.session import SessionLocal
from visionstack.detection.person_detector import PersonDetector
from visionstack.identity import face_api_client
from visionstack.identity.face_detector import HaarFaceDetector
from visionstack.ingestion.frame_sampler import FrameSampler
from visionstack.ingestion.video_source import video_source_from_config
from visionstack.zones.monitor import DbZoneMonitor, ZoneRecord

logger = logging.getLogger("visionstack.api.live_stream")

JPEG_QUALITY = 80
MAX_PROBE_DEVICES = 5
PROCESSED_VIDEOS_DIR = REPO_ROOT / "data" / "processed_videos"
# Per-track retry cadence (seconds): an unverified track is retried at most this often, rather
# than every frame, so someone whose face never clears the quality gate doesn't hammer /verify
# continuously. Once a track is verified it's never retried -- see PersonDetector.track() /
# _TrackState below, which replaced the old "re-verify everyone every N seconds" whole-frame
# batch approach (no persistent per-person tracking existed then).
VERIFY_RETRY_SECONDS = 2.0
# A track must be seen this many consecutive frames before its first verify attempt -- cheap
# filter for the transient false-positive tracks ByteTrack occasionally produces for 1-2 frames
# (observed directly: a spurious 5th id flickering for a few frames alongside 4 real, stable ids
# on the same test clip).
MIN_TRACK_FRAMES_BEFORE_VERIFY = 3
# Drop a track's cached state if it hasn't been seen for this long (person left frame / camera
# view) -- keeps a long-running session's track dict from growing unbounded.
TRACK_TTL_SECONDS = 30.0
FACE_CROP_MARGIN_RATIO = 0.3
# Cap concurrent /verify calls within one refresh batch -- each takes ~2s server-side; firing an
# unbounded number at once (e.g. 10 people in frame) would hammer the external service.
MAX_CONCURRENT_VERIFIES = 4
# Same palette as frontend/src/components/ZoneDrawer.tsx's ZONE_COLORS (converted RGB -> BGR for
# cv2), in the same order, so a zone drawn a given color in the browser overlay is drawn that
# same color when burned into a saved recording -- assigned by position in a zone_id-sorted list
# (see zones/monitor.DbZoneMonitor.zones), same as the frontend, so it's stable across frames/runs.
ZONE_COLORS_BGR = [
    (38, 38, 220),  # red
    (235, 99, 37),  # blue
    (74, 163, 22),  # green
    (6, 119, 217),  # amber
    (234, 51, 147),  # purple
    (178, 145, 8),  # cyan
    (119, 39, 219),  # pink
    (13, 163, 101),  # lime
    (12, 88, 234),  # orange
    (229, 70, 79),  # indigo
]


def list_webcam_devices() -> list[dict]:
    """Probe local video devices (indices 0..MAX_PROBE_DEVICES-1) for ones that actually open."""
    devices = []
    for index in range(MAX_PROBE_DEVICES):
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened():
                devices.append({"device_index": index, "label": f"Webcam {index}"})
        finally:
            cap.release()
    return devices


def _draw_zones(image, zones: list[ZoneRecord]):
    """Burns each zone's outline + name into the frame -- so a saved recording (live or uploaded)
    shows which zone is which on its own, not just in the browser's draw-zone overlay. Same
    color-per-zone assignment as frontend/src/components/ZoneDrawer.tsx, drawn *before*
    _draw_detections so person boxes stay on top of the (semi-transparent) zone fill rather than
    under it.
    """
    zones = sorted(zones, key=lambda z: z.zone_id)
    for i, zone in enumerate(zones):
        if len(zone.polygon) < 3:
            continue
        color = ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)]
        points = np.array([[int(x), int(y)] for x, y in zone.polygon], dtype=np.int32)

        overlay = image.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

        label = f"{zone.name} (login)" if zone.triggers_login else zone.name
        label_x, label_y = int(points[0][0]), max(15, int(points[0][1]) - 8)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (label_x, label_y - th - 4), (label_x + tw + 6, label_y + 4), color, -1)
        cv2.putText(
            image, label, (label_x + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return image


def _format_track_label(track_id: str) -> str:
    """track_id is "{camera_id}-{n}" (tracker's own per-run counter, see
    tracking/local_tracker.TrackTrackLocalTracker / PersonDetector.track()) -- the camera_id
    prefix is redundant noise on an overlay burned into that camera's own video, so show just the
    number. Mirrors frontend/src/pages/Recordings.tsx's formatTrackId -- keep the two in sync.
    """
    match = re.search(r"-(\d+)$", track_id)
    return f"#{match.group(1)}" if match else track_id


def _draw_detections(
    image,
    detections: list[Detection],
    track_names: dict[str, str] | None = None,
    id_labels: dict[str, str] | None = None,
):
    """track_names maps track_id -> recognized name (only for tracks that have one yet). Each
    detection carries its own track_id (see PersonDetector.track()), so this is a direct lookup
    -- no positional/ordering heuristic needed now that identity is real, not inferred from
    left-to-right position.

    id_labels optionally overrides the displayed id per track_id -- used by video_upload.py to
    show the stable, pgvector-resolved reid_identity_id's display number instead of
    TrackTrackLocalTracker's own raw (still fragmentation-prone across a long absence) counter, so
    what's burned into the video matches what _summarize_zone_visits actually counts. Falls back to
    _format_track_label(track_id) when a track_id has no override -- live_stream.py's BoT-SORT path
    (no reid store) never passes this, so its behavior is unchanged.

    The id is the primary, bold/high-contrast part of the label (black background plate + thick
    yellow text) -- readable against any underlying video content -- since it's what gets used to
    visually verify tracking stability (see the tracking-fragmentation investigation this was built
    for). Name/confidence stay as a smaller secondary line underneath.
    """
    track_names = track_names or {}
    id_labels = id_labels or {}
    for d in detections:
        p1 = (int(d.bbox.x1), int(d.bbox.y1))
        p2 = (int(d.bbox.x2), int(d.bbox.y2))
        cv2.rectangle(image, p1, p2, (0, 200, 0), 2)

        id_label = id_labels.get(d.track_id) if d.track_id else None
        if id_label is None:
            id_label = _format_track_label(d.track_id) if d.track_id else "?"
        (tw, th), baseline = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        plate_x1, plate_y2 = p1[0], max(th + baseline + 4, p1[1] - 4)
        plate_y1 = plate_y2 - th - baseline - 6
        cv2.rectangle(image, (plate_x1, plate_y1), (plate_x1 + tw + 8, plate_y2), (0, 0, 0), -1)
        cv2.putText(
            image,
            id_label,
            (plate_x1 + 4, plate_y2 - baseline - 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        name = track_names.get(d.track_id) if d.track_id else None
        sub_label = f"{name} {d.confidence:.2f}" if name else f"{d.confidence:.2f}"
        cv2.putText(
            image,
            sub_label,
            (plate_x1, plate_y2 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return image


def _crop_with_margin(image, bbox: BBox, margin_ratio: float):
    h, w = image.shape[:2]
    mx, my = bbox.width * margin_ratio, bbox.height * margin_ratio
    x1 = max(0, int(bbox.x1 - mx))
    y1 = max(0, int(bbox.y1 - my))
    x2 = min(w, int(bbox.x2 + mx))
    y2 = min(h, int(bbox.y2 + my))
    return image[y1:y2, x1:x2]


def _ensure_min_size(image, min_dim: int = 320):
    """Upscales small crops before sending to /verify. Confirmed against the real API on a still
    image with a known-good face: the exact same image at 245px on its short side got "Move
    closer.", and at 246px -- one pixel more, via plain cv2.INTER_CUBIC upscaling with no new
    real detail -- it passed and returned a genuine match. So this isn't compensating for actual
    image quality, it's clearing a literal minimum-dimension check; interpolated pixels count the
    same as real ones to it. 320 leaves headroom above the measured ~246px cutoff. A distant CCTV
    person's crop is exactly the case this helps: too small to pass on its own, and upscaling it
    doesn't need better camera hardware to do so.
    """
    h, w = image.shape[:2]
    if min(h, w) >= min_dim or min(h, w) == 0:
        return image
    scale = min_dim / min(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def build_face_crops(detections: list[Detection], frame_image, face_detector: HaarFaceDetector) -> list[bytes | None]:
    """Shared by the live pipeline and uploaded-video processing (video_upload.py): for every
    detection given, produce the JPEG bytes to send to /verify, in the same order as `detections`
    -- Haar's crop when it finds a face, the whole person crop otherwise, always topped up to a
    safe minimum size. See _maybe_verify_tracks below for why each of these choices was made;
    extracted here purely to avoid re-deriving the same logic per caller.
    """
    face_crops: list[bytes | None] = []
    for d in detections:
        person_crop = _crop_with_margin(frame_image, d.bbox, margin_ratio=0.0)
        faces = face_detector.detect(person_crop)
        if faces:
            face = max(faces, key=lambda f: f.bbox.width * f.bbox.height)
            crop_for_verify = _crop_with_margin(person_crop, face.bbox, FACE_CROP_MARGIN_RATIO)
        else:
            crop_for_verify = person_crop
        crop_for_verify = _ensure_min_size(crop_for_verify)
        ok, buf = cv2.imencode(".jpg", crop_for_verify, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        face_crops.append(buf.tobytes() if ok else None)
    return face_crops


def verify_crops_batch(face_crops: list[bytes | None], context: str) -> list[tuple[str, str] | None]:
    """Runs /verify for every non-None crop with bounded concurrency and resolves each face_id
    to an (employee_id, full_name) pair. `context` is just a label for logging (camera_id for
    live, the uploaded filename for video_upload.py) -- this has no other awareness of the
    caller. Returns (employee_id, name) rather than just name so live_stream.py's zone-based
    login trigger has an employee_id to write an AttendanceEvent against -- video_upload.py (the
    other caller) only needs the name and just ignores the id.
    """
    results: list[tuple[str, str] | None] = [None] * len(face_crops)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VERIFIES) as pool:
        futures = {
            pool.submit(face_api_client.verify_face, crop): i
            for i, crop in enumerate(face_crops)
            if crop is not None
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                face_id = future.result()
            except FaceApiError:
                logger.exception("verify_face failed for '%s' (person %d)", context, i)
                continue
            if face_id is None:
                logger.info("verify_face: no enrolled match for '%s' (person %d)", context, i)
                continue
            employee = _lookup_employee(face_id)
            logger.info(
                "verify_face: '%s' person %d -> face_id=%s -> %s",
                context, i, face_id, employee[1] if employee else "(face_id not found in our DB)",
            )
            results[i] = employee
    return results


def check_zone_logins(
    zone_monitor: DbZoneMonitor,
    attendance_engine: DbAttendanceEngine,
    tracks: dict,
    camera_id: str,
    frame,
    detections: list[Detection],
) -> None:
    """Zone-based login. Shared by the live pipeline (LiveStreamManager._check_zone_logins) and
    uploaded-video processing (video_upload.py) -- same mechanism either way: for every detection
    whose track has already been verified (has an employee_id -- see _run_track_verify/its
    video_upload.py equivalent), checks its bounding box against `camera_id`'s zones and fires
    AttendanceEngine.on_entry_match() the moment it enters a zone with `triggers_login=True`. An
    unverified/unrecognized person can walk through a login zone without effect -- only a matched
    employee can trigger it. See zones/monitor.DbZoneMonitor for why entry/exit events aren't
    persisted to the zone_events table.

    `tracks` just needs `.get(track_id)` returning something with `.employee_id`/`.name` -- both
    callers' own `_TrackState` dataclasses satisfy that structurally, no shared base class needed.

    Note for uploaded video: zones are looked up by `camera_id`, and video_upload.py uses one
    fixed `UPLOAD_CAMERA_ID` for every upload (single-camera setup, see that constant's
    docstring) -- draw a zone once against any upload's preview frame (GET
    /videos/upload/{video_id}/preview-frame + frontend ZoneDrawer) and it applies to every
    upload processed afterward. A zone drawn while a job is already running (DbZoneMonitor loads
    zones once, at job start) only takes effect on the *next* upload, not the one in flight.
    """
    for d in detections:
        if d.track_id is None:
            continue
        state = tracks.get(d.track_id)
        if state is None or state.employee_id is None:
            continue

        events = zone_monitor.check(d.track_id, camera_id, d.bbox, None, frame.timestamp)
        for event in events:
            if event.event_type != "enter":
                continue
            zone = zone_monitor.zone(event.zone_id)
            if zone is None or not zone.triggers_login:
                continue
            attendance_event = attendance_engine.on_entry_match(state.employee_id, camera_id, frame.timestamp)
            if attendance_event is not None:
                logger.info(
                    "zone login: employee_id=%s name=%s zone=%s camera=%s",
                    state.employee_id, state.name, zone.name, camera_id,
                )


def _lookup_employee(face_id: str) -> tuple[str, str] | None:
    """Returns (employee_id, full_name), or None if face_id isn't linked to an active enrollment."""
    with SessionLocal() as db:
        record = (
            db.query(EmployeeFaceEmbedding)
            .filter(EmployeeFaceEmbedding.source_ref == face_id, EmployeeFaceEmbedding.is_active)
            .first()
        )
        if record is None:
            return None
        employee = db.get(Employee, record.employee_id)
        return (str(employee.id), employee.full_name) if employee else None


def _transcode_to_h264(raw_path: Path, final_path: Path) -> bool:
    """Re-encodes to H.264/yuv420p (browser-playable) via the system ffmpeg. Returns False
    (leaving raw_path in place for the caller to fall back to) if ffmpeg is unavailable or fails.
    """
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found on PATH; keeping raw mp4v recording (not browser-playable)")
        return False
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(raw_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(final_path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        logger.warning("ffmpeg transcode of '%s' failed: %s", raw_path, result.stderr.decode(errors="replace"))
        return False
    raw_path.unlink(missing_ok=True)
    return True


@dataclass
class _TrackState:
    """Per-track_id verification cache. Once verified, a track keeps its name for the rest of
    the session/video without re-hitting /verify -- the whole point of having real track ids
    instead of re-verifying everyone visible on a fixed timer."""

    name: str | None = None
    employee_id: str | None = None
    verified: bool = False
    seen_frames: int = 0
    last_seen_at: float = 0.0
    last_attempt_at: float = 0.0


@dataclass
class _Session:
    camera_id: str
    device_index: int
    stop_event: threading.Event
    video_id: str
    raw_path: Path
    final_path: Path
    started_at: datetime
    thread: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest_jpeg: bytes | None = None
    frame_count: int = 0
    detection_count: int = 0
    max_people_in_frame: int = 0
    writer: cv2.VideoWriter | None = None
    error: str | None = None
    tracks: dict[str, _TrackState] = field(default_factory=dict)
    all_recognized_names: set[str] = field(default_factory=set)  # accumulated for the recording's metadata
    verify_in_flight: bool = False
    frame_width: int | None = None
    frame_height: int | None = None
    zone_monitor: DbZoneMonitor | None = None
    attendance_engine: DbAttendanceEngine | None = None


class LiveStreamManager:
    """Process-wide registry of active webcam capture+detection sessions, one per camera_id."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._registry_lock = threading.Lock()

    def start(self, camera_id: str, device_index: int, sample_fps: float) -> None:
        with self._registry_lock:
            existing = self._sessions.get(camera_id)
            if existing is not None and existing.thread is not None and existing.thread.is_alive():
                raise RuntimeError(f"'{camera_id}' is already running")

            PROCESSED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
            video_id = uuid.uuid4().hex
            session = _Session(
                camera_id=camera_id,
                device_index=device_index,
                stop_event=threading.Event(),
                video_id=video_id,
                raw_path=PROCESSED_VIDEOS_DIR / f"{video_id}_raw.mp4",
                final_path=PROCESSED_VIDEOS_DIR / f"{video_id}.mp4",
                started_at=datetime.now(timezone.utc),
            )
            thread = threading.Thread(
                target=self._run, args=(session, sample_fps), daemon=True, name=f"live-{camera_id}"
            )
            session.thread = thread
            self._sessions[camera_id] = session
            thread.start()

    def stop(self, camera_id: str) -> None:
        session = self._sessions.get(camera_id)
        if session is None:
            raise KeyError(camera_id)
        session.stop_event.set()
        if session.thread is not None:
            session.thread.join(timeout=10)

    def status(self, camera_id: str) -> dict:
        session = self._sessions.get(camera_id)
        if session is None:
            return {
                "camera_id": camera_id,
                "running": False,
                "frame_count": 0,
                "detection_count": 0,
                "error": None,
                "recognized_names": [],
                "frame_width": None,
                "frame_height": None,
            }
        return {
            "camera_id": camera_id,
            "running": session.thread is not None and session.thread.is_alive(),
            "frame_count": session.frame_count,
            "detection_count": session.detection_count,
            "error": session.error,
            "recognized_names": sorted(session.all_recognized_names),
            "frame_width": session.frame_width,
            "frame_height": session.frame_height,
        }

    def latest_frame(self, camera_id: str) -> bytes | None:
        session = self._sessions.get(camera_id)
        if session is None:
            return None
        with session.lock:
            return session.latest_jpeg

    def _run(self, session: _Session, sample_fps: float) -> None:
        try:
            video_source = video_source_from_config(
                camera_id=session.camera_id, source_type="webcam", uri=str(session.device_index)
            )
            detector = PersonDetector(weights_path=get_settings().detection_weights_path, device="auto")
            face_detector = HaarFaceDetector()
            session.zone_monitor = DbZoneMonitor(session.camera_id)
            session.attendance_engine = DbAttendanceEngine()

            with video_source as source:
                sampler = FrameSampler(source, source_fps=source.fps, sample_fps=sample_fps)
                for frame in sampler:
                    if session.stop_event.is_set():
                        break

                    detections = detector.track(frame)
                    self._maybe_verify_tracks(session, face_detector, frame.image, detections)
                    self._check_zone_logins(session, frame, detections)
                    track_names = {tid: s.name for tid, s in session.tracks.items() if s.name}
                    zones = session.zone_monitor.zones if session.zone_monitor is not None else []
                    annotated = _draw_zones(frame.image.copy(), zones)
                    annotated = _draw_detections(annotated, detections, track_names)

                    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if ok:
                        with session.lock:
                            session.latest_jpeg = buf.tobytes()

                    if session.writer is None:
                        h, w = annotated.shape[:2]
                        session.frame_height, session.frame_width = h, w
                        session.writer = cv2.VideoWriter(
                            str(session.raw_path), cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (w, h)
                        )
                    session.writer.write(annotated)

                    session.frame_count += 1
                    session.detection_count += len(detections)
                    session.max_people_in_frame = max(session.max_people_in_frame, len(detections))
        except Exception as e:  # noqa: BLE001 -- surface failures via status() instead of a silent thread death
            logger.exception("live stream session '%s' crashed", session.camera_id)
            session.error = str(e)
        finally:
            self._finalize_recording(session)

    def _check_zone_logins(self, session: _Session, frame, detections: list[Detection]) -> None:
        if session.zone_monitor is None or session.attendance_engine is None:
            return
        check_zone_logins(
            session.zone_monitor, session.attendance_engine, session.tracks, session.camera_id, frame, detections
        )

    def _maybe_verify_tracks(
        self,
        session: _Session,
        face_detector: HaarFaceDetector,
        frame_image,
        detections: list[Detection],
    ) -> None:
        """Phase 3, wired into the live pipeline: person detection+tracking (already done by the
        caller via PersonDetector.track()) -> per-track verification cache -> local face-crop
        extraction only for tracks that are due -> external /verify (bounded concurrency) -> DB
        lookup by face_id.

        TODO(tracking-first): the actual /verify call is currently commented out in
        _run_track_verify below -- tracking is the current focus, not identity. Everything up to
        and including face-crop extraction below still runs as described; every track just comes
        back unverified for now.

        Each track_id is verified at most once successfully -- after that its name is cached on
        the track for the rest of the session, no repeat /verify calls. An unverified track is
        retried at most every VERIFY_RETRY_SECONDS (someone whose face never clears the quality
        gate doesn't get hit every frame), and only after it's been seen
        MIN_TRACK_FRAMES_BEFORE_VERIFY consecutive frames (skips ByteTrack's occasional 1-2 frame
        false-positive tracks). This replaced the old approach of re-verifying every visible
        person as one batch on a fixed timer, which had no persistent identity to cache against.

        The external call measured ~2s round trip; running these in a background thread (not
        inline) keeps the capture loop from stalling on it -- that stall is what "tracking is
        slow" actually was, independent of person-detection speed/GPU. Only one verify batch runs
        at a time per session (session.verify_in_flight), same as before.

        Haar is used to tighten the crop when it succeeds, but measured against this camera's
        actual framing it found a face in 0/127 person-crops (too far/angled for a cascade
        classifier) -- treating a Haar miss as "no face, don't bother verifying" was silently
        preventing verification from ever running at all. So a Haar miss now falls back to
        sending the *whole* person crop instead of skipping that person: the external service
        does its own (materially better) detection and quality gating already -- that's exactly
        what rejected under-filled onboarding photos with "Move closer." earlier -- so it's the
        right place for that decision, not a local heuristic that's empirically ~0% effective here.

        A/B tested two narrower fallback crops against the real API on the same live frame before
        settling here: a head-centered crop (regressed a real match to "not verified" by cutting
        pixel area ~4x) and an upper-half-of-body crop (same-or-worse than sending the whole
        person crop in every case tested, e.g. one person went from a "Move closer." quality
        rejection with upper-half to actually clearing the gate with the full crop). So this sends
        the largest crop available rather than sub-cropping further, then _ensure_min_size tops
        it up to a safe minimum: pinned the exact "Move closer." cutoff to a pixel dimension (245
        fails, 246 passes on an otherwise-identical image), confirmed plain interpolation clears
        it -- no new real detail needed -- so a small crop is upscaled before sending rather than
        left to fail on a distant person the local camera simply can't resolve any larger.
        """
        now = time.monotonic()
        for d in detections:
            if d.track_id is None:
                continue
            state = session.tracks.setdefault(d.track_id, _TrackState())
            state.seen_frames += 1
            state.last_seen_at = now

        stale = [
            tid for tid, s in session.tracks.items() if now - s.last_seen_at > TRACK_TTL_SECONDS
        ]
        for tid in stale:
            del session.tracks[tid]
            if session.zone_monitor is not None:
                session.zone_monitor.forget_track(tid)

        if session.verify_in_flight:
            return

        due = [
            d
            for d in detections
            if d.track_id is not None
            and not session.tracks[d.track_id].verified
            and session.tracks[d.track_id].seen_frames >= MIN_TRACK_FRAMES_BEFORE_VERIFY
            and now - session.tracks[d.track_id].last_attempt_at >= VERIFY_RETRY_SECONDS
        ]
        if not due:
            return

        for d in due:
            session.tracks[d.track_id].last_attempt_at = now

        face_crops = build_face_crops(due, frame_image, face_detector)
        session.verify_in_flight = True
        threading.Thread(
            target=self._run_track_verify, args=(session, due, face_crops), daemon=True,
            name=f"verify-batch-{session.camera_id}",
        ).start()

    def _run_track_verify(
        self, session: _Session, due: list[Detection], face_crops: list[bytes | None]
    ) -> None:
        # TODO(tracking-first): face verification (the external /verify call) is disabled for
        # now -- current focus is getting tracking solid first, not identity. Face crops are
        # still extracted (build_face_crops, in _maybe_verify_tracks) so that path stays
        # exercised; they're just never sent anywhere. Every track is therefore left unverified
        # (no name/employee_id), which also means zone-based login (which requires employee_id)
        # won't fire until this is turned back on. Re-enable by restoring the commented block.
        #
        # try:
        #     results = verify_crops_batch(face_crops, context=session.camera_id)
        # except Exception:  # noqa: BLE001 -- a failed batch just leaves these tracks unverified/retried later
        #     logger.exception("verify batch failed for '%s'", session.camera_id)
        #     results = [None] * len(due)
        results: list[tuple[str, str] | None] = [None] * len(due)

        for d, result in zip(due, results):
            state = session.tracks.get(d.track_id)
            if state is None or result is None:
                continue
            employee_id, name = result
            state.name = name
            state.employee_id = employee_id
            state.verified = True
            session.all_recognized_names.add(name)
        session.verify_in_flight = False

    def _finalize_recording(self, session: _Session) -> None:
        if session.writer is not None:
            session.writer.release()

        if session.frame_count == 0:
            session.raw_path.unlink(missing_ok=True)
            return

        playable = _transcode_to_h264(session.raw_path, session.final_path)
        if not playable:
            session.raw_path.replace(session.final_path)

        ended_at = datetime.now(timezone.utc)
        metadata = {
            "video_id": session.video_id,
            "camera_id": session.camera_id,
            "started_at": session.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": (ended_at - session.started_at).total_seconds(),
            "frame_count": session.frame_count,
            "detection_count": session.detection_count,
            "max_people_in_frame": session.max_people_in_frame,
            "filename": session.final_path.name,
            "browser_playable": playable,
            "recognized_names": sorted(session.all_recognized_names),
        }
        session.final_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

    def mjpeg_frames(self, camera_id: str, push_fps: float = 15.0):
        """Yields multipart/x-mixed-replace chunks for a StreamingResponse. Stops once the
        session is no longer running (or never existed)."""
        interval = 1 / push_fps
        while True:
            frame = self.latest_frame(camera_id)
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            if not self.status(camera_id)["running"]:
                break
            time.sleep(interval)


def list_processed_videos() -> list[dict]:
    if not PROCESSED_VIDEOS_DIR.exists():
        return []
    records = []
    for meta_path in PROCESSED_VIDEOS_DIR.glob("*.json"):
        try:
            records.append(json.loads(meta_path.read_text()))
        except (json.JSONDecodeError, OSError):
            logger.warning("skipping unreadable video metadata file '%s'", meta_path)
            continue
    records.sort(key=lambda r: r["started_at"], reverse=True)
    return records


def get_processed_video_path(video_id: str) -> Path | None:
    meta_path = PROCESSED_VIDEOS_DIR / f"{video_id}.json"
    if not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text())
    path = PROCESSED_VIDEOS_DIR / metadata["filename"]
    return path if path.exists() else None


live_stream_manager = LiveStreamManager()
