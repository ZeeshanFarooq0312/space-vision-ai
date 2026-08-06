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
| 1. Ingestion | `ingestion/` | **Real** — RTSP/file video sources via OpenCV |
| 2. Detection | `detection/` | **Real** — YOLOv8 person detection |
| 3. Identity | `identity/` | Stub — face/body embedding + gallery matching (TODO: RetinaFace/SCRFD, ArcFace+GhostNet, OSNet) |
| 4. Attendance | `attendance/` | Stub — login/logout events, false-logout prevention (TODO) |
| 5. Zones | `zones/` | Config loading real; violation detection stub (TODO) |
| 6. Tracking | `tracking/` | Stub local tracking + cross-camera ReID (TODO: ByteTrack, Hungarian fusion) |
| 7. Reporting | `reporting/` | Stub — attendance reports, dwell time, trajectory replay (TODO) |

Stub phases are callable no-ops (not `NotImplementedError`), so the full pipeline runs end-to-end
today even though only ingestion and detection do real work. See each module's `TODO:` comments
for what to implement next.

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

## Tests

```bash
pytest
```

Unit tests run without any external data. The integration test
(`tests/integration/test_run_pipeline_on_sample_video.py`) skips automatically until you add
`samples/sample.mp4`.

## Docker

```bash
docker compose up -d postgres api frontend
```

GPU-accelerated inference (`docker compose --profile gpu up inference`) requires the NVIDIA
Container Toolkit and is not verified to work out of the box on Windows/Docker Desktop — the
primary dev path for the detection pipeline is running `run_pipeline.py` directly on the host,
which falls back to CPU automatically if no CUDA device is available (`--device auto`, the default).

## Known gaps (intentional, for later passes)

- No RTSP live-stream integration test — Phase 1 is verified via file/video input only.
- Biometric embedding tables are isolated by schema but have no encryption-at-rest, row-level
  security, or audit logging yet.
- Phases 3, 4 (partially), 5 (partially), 6, and 7 are stub interfaces — see each module's `TODO:`
  comments for the real model/algorithm to implement.
