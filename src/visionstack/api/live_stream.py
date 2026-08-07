"""Background webcam capture + detection sessions backing the browser live-preview UI.

Each session runs Phase 1 (ingestion) + Phase 2 (detection) from the real Pipeline in a
dedicated thread — not asyncio, since cv2.VideoCapture.read() and YOLO inference are both
blocking calls that would otherwise stall the event loop. The thread continuously encodes the
latest annotated frame as JPEG under a lock; the MJPEG stream endpoint just reads that latest
frame at its own pace, so a slow/disconnected client never backs up the capture loop.

Every session is also recorded to disk (annotated frames) so it can be replayed afterward from
the Recordings UI, independent of the live view. OpenCV's own FFmpeg build has no H.264 encoder
(only mp4v, which browsers won't play), so the raw mp4v file is transcoded via the system `ffmpeg`
binary; if that's unavailable, the raw file is kept as a fallback (playable in VLC, not browsers).
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2

from visionstack.common.config import REPO_ROOT
from visionstack.common.types import Detection
from visionstack.detection.person_detector import PersonDetector
from visionstack.ingestion.video_source import video_source_from_config
from visionstack.pipeline.orchestrator import Pipeline

logger = logging.getLogger("visionstack.api.live_stream")

JPEG_QUALITY = 80
MAX_PROBE_DEVICES = 5
PROCESSED_VIDEOS_DIR = REPO_ROOT / "data" / "processed_videos"


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


def _draw_detections(image, detections: list[Detection]):
    for d in detections:
        p1 = (int(d.bbox.x1), int(d.bbox.y1))
        p2 = (int(d.bbox.x2), int(d.bbox.y2))
        cv2.rectangle(image, p1, p2, (0, 200, 0), 2)
        cv2.putText(
            image,
            f"person {d.confidence:.2f}",
            (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return image


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
            return {"camera_id": camera_id, "running": False, "frame_count": 0, "detection_count": 0, "error": None}
        return {
            "camera_id": camera_id,
            "running": session.thread is not None and session.thread.is_alive(),
            "frame_count": session.frame_count,
            "detection_count": session.detection_count,
            "error": session.error,
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
            detector = PersonDetector(device="auto")
            pipeline = Pipeline(video_source=video_source, detector=detector, sample_fps=sample_fps)

            gen = pipeline.run()
            try:
                for result in gen:
                    if session.stop_event.is_set():
                        break
                    annotated = _draw_detections(result.frame.image.copy(), result.detections)

                    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if ok:
                        with session.lock:
                            session.latest_jpeg = buf.tobytes()

                    if session.writer is None:
                        h, w = annotated.shape[:2]
                        session.writer = cv2.VideoWriter(
                            str(session.raw_path), cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (w, h)
                        )
                    session.writer.write(annotated)

                    session.frame_count += 1
                    session.detection_count += len(result.detections)
                    session.max_people_in_frame = max(session.max_people_in_frame, len(result.detections))
            finally:
                gen.close()  # propagates into Pipeline.run()'s `with video_source:` so the device is released
        except Exception as e:  # noqa: BLE001 -- surface failures via status() instead of a silent thread death
            logger.exception("live stream session '%s' crashed", session.camera_id)
            session.error = str(e)
        finally:
            self._finalize_recording(session)

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
