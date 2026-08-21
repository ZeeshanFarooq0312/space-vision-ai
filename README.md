# Vision-Stack AI

AI-powered employee attendance and zone monitoring system — a multi-camera CCTV pipeline that
detects, identifies, and tracks employees from an entry camera through interior building zones to
an exit camera, recording login/logout attendance and firing real-time alerts on unauthorized zone
entry.

## Architecture

A modular monolith: one Python service with a clean module per pipeline phase, so it can be split
into separate services later without a rewrite. See `src/visionstack/`:

| Phase | Module | Status |
|---|---|---|
| 1. Ingestion | `ingestion/` | **Real** — RTSP/file/webcam video sources via OpenCV |
| 2. Detection | `detection/` | **Real** — YOLOv8 person detection, GPU-accelerated |
| 3. Identity | `identity/` | **Real (via external service)** — face onboarding/verification calls out to a FaceVerify API (RetinaFace + ArcFace/Facenet512/GhostFaceNet ensemble); no local model yet, see below |
| 4. Attendance | `attendance/` | Stub — login/logout events, false-logout prevention (TODO) |
| 5. Zones | `zones/` | Config loading real; violation detection stub (TODO) |
| 6. Tracking | `tracking/` | Local tracking: **real** (`TrackTrackLocalTracker`, opt-in via `tracking.tracker: tracktrack` — vendored Kalman+ReID tracker, fixed cameras only) or stub (`PassthroughLocalTracker`, default). Cross-camera ReID still stub (TODO: Hungarian fusion). Not used by the live-camera/video-upload API, which uses Ultralytics BoT-SORT instead — see `tracking/local_tracker.py` |
| 7. Reporting | `reporting/` | Stub — attendance reports, dwell time, trajectory replay (TODO) |

Stub phases are callable no-ops (not `NotImplementedError`), so the full pipeline runs end-to-end
today even though phases 4, 6, and 7 don't do real work yet. See each module's `TODO:` comments
for what to implement next.

On top of the phase pipeline, `src/visionstack/api/` is a FastAPI service exposing employee
management, live camera capture with browser preview, recorded-video processing/upload, and the
usual attendance/zones/alerts CRUD, backed by a React/Vite frontend (`frontend/`).

## Live camera pipeline

`POST /live/{camera_id}/start` opens a webcam (`--device-index`) as a background capture session
and streams annotated frames back over MJPEG (`GET /live/{camera_id}/stream`, i.e.
`multipart/x-mixed-replace`) for a `<img>` tag to render directly — no WebRTC/HLS needed for local
preview. `GET /live/{camera_id}/status` reports frame/detection counts and, once available, the
names recognized so far.

Each frame runs through YOLOv8 person detection every frame, but face **recognition** is
batched: rather than tracking individuals across frames or hitting the FaceVerify API per person
per frame, every detected person's face crop is (re-)verified as one batch every
`VERIFY_INTERVAL_SECONDS` (2s, see `live_stream.py`), with bounded concurrency
(`MAX_CONCURRENT_VERIFIES = 4`) against the FaceVerify `/verify` endpoint. Labels from the last
batch are drawn on every frame in between. This trades a couple seconds of label latency for
avoiding an API call per person per frame, which doesn't scale with multiple people on camera.

Face crops sent for verification are the *whole person crop*, refined to a tighter *face* crop via
a lightweight Haar cascade detector (`identity/face_detector.py`) when a face is found in it, then
padded (`FACE_CROP_MARGIN_RATIO = 0.3`) and upscaled if below a minimum size
(`_ensure_min_size`) — the FaceVerify service rejects crops below a face-size quality gate, and
plain interpolation reliably clears it for small/far faces.

Recordings from a live session (annotated video + JSON metadata: duration, frame/detection counts,
peak people-in-frame, recognized names) are written to `data/processed_videos/` and listed at
`GET /videos`, viewable in the frontend's **Recordings** page.

## Recorded-video processing (upload)

Besides live webcam capture, a pre-recorded video (e.g. a higher-quality CCTV export) can be
uploaded and run through the same detection + recognition pipeline in the background:

```
POST /videos/upload          (multipart: file, sample_fps)  -> { video_id, status: "processing" }
GET  /videos/upload/{id}/status                              -> { status, frame_count, detection_count, error }
```

Verification here is throttled by *processed-frame count* (`VERIFY_EVERY_N_FRAMES = 8`, see
`video_upload.py`), not wall-clock time — a file processes as fast as the CPU/GPU allow, not paced
to real time like a live session, so a time-based throttle would fire inconsistently depending on
machine speed. Once done, the result is written to the same `data/processed_videos/` format as a
live recording and shows up in `GET /videos` / the Recordings page automatically — no separate
list or view. Do this from the frontend's **Recordings** page ("Upload a CCTV video" panel), which
polls job status and shows live progress.

## Face verification service (FaceVerify)

Employee onboarding (3 enrollment photos → embeddings) and live/recorded verification both go
through a separately-hosted **FaceVerify Mobile API**: DeepFace with a RetinaFace detector, a
3-model embedding ensemble (ArcFace, Facenet512, GhostFaceNet, all 512-d) with 2-of-3 priority-vote
consensus, confidence zones, and an anti-spoof distance-spread check, backed by Qdrant. There's no
local face-embedding model in this repo yet (`identity/face_embedder.py`'s TODO); see
`identity/face_api_client.py` for the client and `.env.example`'s `FACE_API_BASE_URL`.

A full copy of that service lives in `faceverify/` (`main.py`, its own `Dockerfile`) and is wired
into `docker-compose.yml` as the `faceverify` + `qdrant` services, so onboarding/verify calls can
be debugged locally (real logs, breakpoints, an inspectable Qdrant) instead of only against the
black-box remote deployment at `182.180.87.19:3009`. A couple of local-only quirks to know about:

- The client (`face_api_client.py`) calls `/face/mobile/{onboard,verify,compare,embeddings}` —
  that `/face` prefix comes from a reverse proxy in front of the remote deployment that isn't part
  of this copy, so `faceverify/main.py` registers those paths as aliases of its native
  `/mobile/...` routes.
- `/mobile/onboard`'s third multipart field is literally named `blink` (a generic third-pose slot,
  not an actual liveness/blink check) — `onboard_face()` submits the left-turn photo under that
  field name, with `poses=front,right,left` so error messages still label it correctly.
- The local `faceverify` service points at the **host's existing standalone `qdrant` container**
  (via `host.docker.internal`, not its own fresh one) so local verification runs against the same
  already-enrolled employees as the real deployment, rather than an empty database.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements-dev.txt
cp .env.example .env
```

## Run the detection pipeline

```bash
python -m visionstack.pipeline.run_pipeline --source samples/sample.mp4 --camera-id entry-cam-1
```

No `samples/sample.mp4` yet? See `samples/README.md`. You can also point `--source` at any local
video file or, with `--source-type rtsp`, a live RTSP URL.

YOLOv8 weights (`models/yolov8n.pt`) are downloaded automatically by `ultralytics` on first run.

## Database

```bash
docker compose up -d postgres
alembic -c db/alembic.ini upgrade head
```

## API

```bash
uvicorn visionstack.api.main:app --reload
curl http://localhost:8000/health
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Pages: **Dashboard**, **Live** (start/stop a webcam capture session with MJPEG preview),
**Recordings** (past live sessions + uploaded-video processing, with playback and recognized
names), **Onboarding** (capture/upload 3 enrollment photos for a new employee),
**Employees** (list, individual delete, bulk "delete selected" / "delete all"), **Attendance**,
**Alerts**, **Zones**.

## Tests

```bash
pytest
```

Unit tests run without any external data. The integration test
(`tests/integration/test_run_pipeline_on_sample_video.py`) skips automatically until you add
`samples/sample.mp4`.

## Docker

```bash
docker compose up -d postgres api faceverify frontend
```

Services:

| Service | Purpose |
|---|---|
| `postgres` | pgvector-enabled Postgres — employees, attendance, embeddings |
| `api` | FastAPI backend (this repo's `src/`) |
| `faceverify` | Local copy of the FaceVerify embedding/matching service (see above) |
| `frontend` | Vite dev server, proxied to `api` |
| `inference` | Optional, `--profile gpu` — standalone GPU inference worker |

`api` and `faceverify` both request GPU passthrough (`deploy.resources.reservations.devices`,
NVIDIA Container Toolkit required) and fall back to CPU automatically if no CUDA device is
available. `api` also mounts the host's webcam devices (`/dev/video0`, `/dev/video1` — adjust to
match what's actually on the host) for the Live page, and `./data` for recordings/enrollment
photos/uploads to persist across rebuilds. `faceverify` persists downloaded DeepFace model weights
in a named volume (`deepface_weights`) so a rebuild doesn't re-download them.

GPU-accelerated inference (`docker compose --profile gpu up inference`) requires the NVIDIA
Container Toolkit and is not verified to work out of the box on Windows/Docker Desktop — the
primary dev path for the standalone detection pipeline is running `run_pipeline.py` directly on
the host, which falls back to CPU automatically if no CUDA device is available (`--device auto`,
the default).

## Known gaps (intentional, for later passes)

- No local face-embedding model — onboarding/verification depend on the external/local FaceVerify
  service (see above), not a model shipped in this repo.
- No RTSP live-stream integration test — Phase 1 is verified via file/video/webcam input only.
- Biometric embedding tables are isolated by schema but have no encryption-at-rest, row-level
  security, or audit logging yet.
- Phases 4 (attendance), 5 (zone violation detection), 6 (tracking), and 7 (reporting) are stub
  interfaces — see each module's `TODO:` comments for the real model/algorithm to implement.
- Live recognition currently re-verifies every detected person as a batch on a fixed interval
  rather than tracking individuals across frames — fine for the current person counts, but doesn't
  scale indefinitely (see "Live camera pipeline" above).
