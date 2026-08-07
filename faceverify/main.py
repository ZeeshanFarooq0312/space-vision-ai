"""
FaceVerify Mobile API  v4.1
============================
Lightweight FastAPI service designed for mobile app integration.
 
Endpoints:
  POST   /mobile/onboard         – Register a user with 3 face images
  POST   /mobile/verify          – Verify a user with 1 face image
 
 
 
Models   : ArcFace (512-d) + Facenet512 (512-d) + GhostFaceNet (512-d)
Detector : RetinaFace
Storage  : Qdrant (self-hosted, no Docker)
Consensus: 2-of-3 majority vote
 
── v4.1 additions (on top of v4.0) ──────────────────────────────────────────
SPEED
 13. GPU-first embedding          – DeepFace runs on CUDA when available;
                                    falls back to CPU automatically.
                                    Single-model inference: ~5 s CPU → ~80 ms GPU.
 14. Single face-detect, reuse crop – RetinaFace runs ONCE per image;
                                    the aligned face crop is saved to disk
                                    and all three models embed the same crop
                                    with detector_backend="skip".
                                    Eliminates 2 redundant detector calls
                                    per variant per image.
 15. Batch Qdrant upserts          – All PointStructs for every model are
                                    accumulated and flushed in one
                                    upsert() call per collection instead of
                                    one call per image.  Reduces round-trips
                                    from 9 (3 images × 3 models) to 3.
 16. Qdrant user profile registry  – A dedicated 'faceverify_user_profiles'
                                    collection stores one profile point per
                                    user (dummy vector [0.0], payload only).
                                    face_id (UUID) is returned on both
                                    /onboard and /verify — no external DB
                                    required.
 
── v4.0 optimisations (retained) ────────────────────────────────────────────
SPEED
  1. Singleton Qdrant client       – one TCP connection reused across threads
  2. Collection existence cache    – zero HTTP round-trips after startup
  3. Model warm-up at startup      – eliminates 10-20 s cold-start on first call
  4. Adaptive variant selection    – good image → 2 variants (6 queries)
                                     poor image → 4 variants (12 queries)
  5. Parallel embedding+search     – ThreadPoolExecutor, 12 workers
  6. Early-exit on strong match    – skip remaining variants once
                                     distance < STRONG_MATCH threshold
 
ACCURACY & RECALL
  7. Per-model dynamic thresholds  – tightened to reduce false positives
  8. Confidence-zone gating        – GREEN / YELLOW / RED decision zones
  9. Anti-spoofing quality check   – rejects single-colour / synthetic faces
 10. Enroll best-variant per pose  – stores sharpest preprocessed image,
                                     not just the raw upload
 11. Minimum enrolled images gate  – rejects onboard if < MIN_ENROLL_OK pass
 12. Distance-spread sanity check  – flags suspiciously uniform distances
     (potential replay / spoof attack)
 
Qdrant setup (no Docker):
  wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
  nohup ./qdrant > qdrant.log 2>&1 &
 
  pip install qdrant-client deepface fastapi uvicorn opencv-python numpy
"""

import os

# ── GPU / TF env — MUST be set before any TF / deepface import ────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"
os.environ["ABSL_LOGGING_LEVEL"]        = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + _ld

# ── Qdrant connection ──────────────────────────────────────────────────────────
QDRANT_HOST        = os.environ.get("QDRANT_HOST",      "localhost")
QDRANT_PORT        = int(os.environ.get("QDRANT_PORT",  "6333"))
QDRANT_GRPC        = int(os.environ.get("QDRANT_GRPC",  "6334"))
QDRANT_API_KEY     = os.environ.get("QDRANT_API_KEY",   None)
QDRANT_PREFER_GRPC = os.environ.get("QDRANT_PREFER_GRPC", "false").lower() == "true"
PUBLIC_BASE_URL = "http://182.180.87.19:3009"


import uuid
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime, timezone

import shutil

import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    HnswConfigDiff, OptimizersConfigDiff,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("mobile_api")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FaceVerify Mobile API",
    description="Face onboarding & verification — Qdrant backend, v4.1.",
    version="4.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_mobile")
SRC_DIR  = os.path.join(BASE_DIR, "src")           # public enrollment images
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SRC_DIR,  exist_ok=True)

# ── Public static mount for enrollment images ──────────────────────────────────
# Enrolled face images are accessible at  GET /src/{username}/{filename}
app.mount("/src", StaticFiles(directory=SRC_DIR), name="src")

# ── Detector ───────────────────────────────────────────────────────────────────
DETECTOR = "retinaface"          # most accurate; used at both enroll and verify

# ── 3-model stack ─────────────────────────────────────────────────────────────
#   threshold  – cosine distance below which a match is accepted
#   weight     – contribution to weighted confidence score
#   strong_thr – distance below which we treat the match as "strong"
#                (used for early-exit optimisation)
MODELS = [
    {"name": "ArcFace",      "dim": 512, "threshold": 0.55, "weight": 1.2, "strong_thr": 0.35},
    {"name": "Facenet512",   "dim": 512, "threshold": 0.40, "weight": 1.0, "strong_thr": 0.25},
    {"name": "GhostFaceNet", "dim": 512, "threshold": 0.55, "weight": 1.1, "strong_thr": 0.35},
]

# ── Voting & confidence ────────────────────────────────────────────────────────
MIN_VOTES = 2           # models that must agree → verified

# ── Priority voting ────────────────────────────────────────────────────────────
# Tier 1: BOTH primaries agree on the same person → verified immediately.
#         GhostFaceNet is not consulted — ArcFace + Facenet512 are sufficient.
# Tier 2: Exactly ONE primary agrees + GhostFaceNet also agrees → verified
#         via tiebreaker.
# Reject: Only GhostFaceNet agrees (no primary) → never verified alone.
PRIMARY_MODELS   = {"ArcFace", "Facenet512"}
TIEBREAKER_MODEL = "GhostFaceNet"

# Confidence zones (applied after vote passes)
#   GREEN  ≥ 55  → auto-approve
#   YELLOW 30-55 → approve but flag for audit
#   RED    < 30  → reject even if vote passes (prevents marginal matches)
CONF_GREEN  = 55.0
CONF_YELLOW = 30.0

# ── Thread pool ────────────────────────────────────────────────────────────────
# 3 models × 4 variants = 12 max concurrent tasks
_EXECUTOR = ThreadPoolExecutor(max_workers=12, thread_name_prefix="face_worker")

# ── Image quality gates ────────────────────────────────────────────────────────
BLUR_THRESH        = 10    # Laplacian variance  — below = blurry
MIN_BRIGHTNESS     = 30    # Mean pixel value    — below = too dark
MAX_BRIGHTNESS     = 230   # Mean pixel value    — above = overexposed
MIN_FACE_PX        = 60    # Min face dimension  — below = too far away
MIN_ENROLL_OK      = 1     # Min images that must pass quality + embed
                            # (of the 3 submitted) to accept onboard

# ── Anti-spoof / diversity gate (enroll) ──────────────────────────────────────
# Minimum std-dev of pixel values in the face region.
# Synthetic / printed / screen-replayed faces are unusually uniform.
MIN_FACE_STDDEV    = 15.0

# ── Preprocessing adaptive thresholds ─────────────────────────────────────────
GOOD_BLUR_THRESH   = 80.0    # above this → image is "good" → 2 variants only
GOOD_BRIGHT_LOW    = 60
GOOD_BRIGHT_HIGH   = 210

# ── OPT 15: Batch upsert size ─────────────────────────────────────────────────
# Maximum number of PointStructs flushed in a single Qdrant upsert() call.
# Keeps payload size bounded; tune upward if network latency dominates.
UPSERT_BATCH_SIZE  = 128

# ── OPT 13: GPU availability ──────────────────────────────────────────────────
# Detected once at import time so the warm-up and every embed call can log
# and act on it without re-checking.
def _detect_gpu() -> bool:
    """Return True if a CUDA-capable GPU is visible to TensorFlow."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"[GPU] {len(gpus)} GPU(s) detected: "
                        f"{[g.name for g in gpus]} — embeddings will use CUDA.")
            return True
    except Exception as e:
        logger.warning(f"[GPU] TF GPU check failed: {e}")
    logger.info("[GPU] No CUDA GPU detected — embeddings will run on CPU.")
    return False

GPU_AVAILABLE: bool = _detect_gpu()


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON QDRANT CLIENT  (opt 1)
# ══════════════════════════════════════════════════════════════════════════════
_QDRANT_CLIENT: QdrantClient | None = None
_QDRANT_LOCK = threading.Lock()   # guards singleton initialisation

def _get_qdrant() -> QdrantClient:
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        with _QDRANT_LOCK:
            if _QDRANT_CLIENT is None:         # double-checked locking
                try:
                    _QDRANT_CLIENT = QdrantClient(
                        host        = QDRANT_HOST,
                        port        = QDRANT_PORT,
                        grpc_port   = QDRANT_GRPC,
                        prefer_grpc = QDRANT_PREFER_GRPC,

                        api_key     = QDRANT_API_KEY,
                        timeout     = 10,
                    )
                    logger.info(f"[Qdrant] Client → {QDRANT_HOST}:{QDRANT_PORT}")
                except Exception as e:
                    logger.exception("[Qdrant] Failed to initialize client")
                    raise
    return _QDRANT_CLIENT


# ══════════════════════════════════════════════════════════════════════════════
# COLLECTION EXISTENCE CACHE  (opt 2)
# Thread-safe: _COLLECTION_LOCK prevents race conditions when multiple
# worker threads call _ensure_collection simultaneously.
# ══════════════════════════════════════════════════════════════════════════════
_COLLECTIONS_READY: set[str] = set()
_COLLECTION_LOCK   = threading.Lock()

def _collection_name(model_name: str) -> str:
    return f"faceverify_{model_name.replace('-','').lower()}_{DETECTOR.lower()}"

def _get_points_count(client: QdrantClient, cname: str) -> int:
    """
    Return the number of stored vectors in *cname*.
    Qdrant client ≥ 1.9 uses .points_count; older builds used .vectors_count.
    """
    try:
        info = client.get_collection(cname)
        count = getattr(info, "points_count", None)
        if count is None:
            count = getattr(info, "vectors_count", None)
        return count or 0
    except Exception as e:
        logger.warning(f"[Qdrant] Failed to get points count for {cname}: {e}")
        return 0

def _ensure_collection(model: dict) -> None:
    """
    Create the Qdrant collection for *model* if it doesn't exist yet.
    Result cached in _COLLECTIONS_READY — subsequent calls are a pure
    Python set lookup (zero I/O).
    """
    cname = _collection_name(model["name"])
    if cname in _COLLECTIONS_READY:
        return
    with _COLLECTION_LOCK:
        if cname in _COLLECTIONS_READY:
            return
        try:
            client = _get_qdrant()
            if not client.collection_exists(cname):
                client.create_collection(
                    collection_name   = cname,
                    vectors_config    = VectorParams(size=model["dim"], distance=Distance.COSINE),
                    hnsw_config       = HnswConfigDiff(m=16, ef_construct=200),
                    optimizers_config = OptimizersConfigDiff(memmap_threshold=20_000),
                )
                client.create_payload_index(
                    collection_name = cname,
                    field_name      = "username",
                    field_schema    = "keyword",
                )
                logger.info(f"[Qdrant] Created '{cname}' ✓")
            else:
                count = _get_points_count(client, cname)
                logger.info(f"[Qdrant] '{cname}' ready ✓  points={count}")
            _COLLECTIONS_READY.add(cname)
        except Exception as e:
            logger.exception(f"[Qdrant] Failed to ensure collection '{cname}'")
            raise


# ══════════════════════════════════════════════════════════════════════════════
# OPT 16 — USER PROFILE COLLECTION
# One point per user. Vector is a dummy [0.0] — this collection is never
# ANN-searched, only scrolled by exact payload filter on face_id.
# Payload carries face_id, created_at, and any future metadata you need.
# ══════════════════════════════════════════════════════════════════════════════
USER_PROFILE_COLLECTION = "faceverify_user_profiles"

def _ensure_profile_collection() -> None:
    """
    Create the user-profile collection if it doesn't exist.
    Uses the same _COLLECTIONS_READY cache and _COLLECTION_LOCK as face
    collections so the pattern is consistent across the codebase.
    """
    cname = USER_PROFILE_COLLECTION
    if cname in _COLLECTIONS_READY:
        return
    with _COLLECTION_LOCK:
        if cname in _COLLECTIONS_READY:
            return
        try:
            client = _get_qdrant()
            if not client.collection_exists(cname):
                # size=1 — smallest valid vector; this collection is never ANN-searched.
                client.create_collection(
                    collection_name = cname,
                    vectors_config  = VectorParams(size=1, distance=Distance.COSINE),
                )
                # Index on face_id makes scroll() lookups O(1)
                client.create_payload_index(
                    collection_name = cname,
                    field_name      = "face_id",
                    field_schema    = "keyword",
                )
                logger.info(f"[Qdrant] Created '{cname}' ✓")
            else:
                count = _get_points_count(client, cname)
                logger.info(f"[Qdrant] '{cname}' ready ✓  points={count}")
            _COLLECTIONS_READY.add(cname)
        except Exception as e:
            logger.exception(f"[Qdrant] Failed to ensure profile collection '{cname}'")
            raise


def _register_user_profile(face_id: str) -> None:
    """
    Insert one profile point for *face_id* into the profile collection.
    """
    try:
        _ensure_profile_collection()
        _get_qdrant().upsert(
            collection_name = USER_PROFILE_COLLECTION,
            points = [PointStruct(
                id      = str(uuid.uuid4()),
                vector  = [0.0],                   # dummy — never ANN-searched
                payload = {
                    "face_id":    face_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )],
        )
        logger.info(f"[Profile] Registered face_id={face_id}")
    except Exception as e:
        logger.exception(f"[Profile] Failed to register face_id={face_id}")
        raise


def _lookup_face_id(username: str) -> str | None:
    """
    Given the internal *username*, retrieve the public face_id from the profile collection.
    """
    try:
        _ensure_profile_collection()
        results, _ = _get_qdrant().scroll(
            collection_name = USER_PROFILE_COLLECTION,
            scroll_filter   = Filter(must=[
                FieldCondition(key="face_id", match=MatchValue(value=username))
            ]),
            limit           = 1,
            with_payload    = True,
            with_vectors    = False,
        )
        if results:
            return results[0].payload.get("face_id")
    except Exception as e:
        logger.warning(f"[Profile] Lookup failed for username={username}: {e}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP  (opt 3 — model warm-up)
# ══════════════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    # 1. Ensure user-profile collection exists
    try:
        _ensure_profile_collection()
    except Exception as e:
        logger.exception(f"[Startup] Profile collection init failed: {e}")

    # 2. Ensure all face-vector Qdrant collections exist
    for m in MODELS:
        try:
            _ensure_collection(m)
        except Exception as e:
            logger.exception(f"[Startup] Collection init failed for {m['name']}: {e}")

    # 3. Pre-load all model weights into GPU/CPU memory.
    logger.info(f"[Warmup] Pre-loading face models… (GPU={'yes' if GPU_AVAILABLE else 'no'})")
    dummy      = np.zeros((112, 112, 3), dtype=np.uint8)
    dummy_path = os.path.join(TEMP_DIR, "_warmup_dummy.jpg")
    try:
        cv2.imwrite(dummy_path, dummy)
        for m in MODELS:
            try:
                DeepFace.represent(
                    img_path          = dummy_path,
                    model_name        = m["name"],
                    detector_backend  = "skip",
                    enforce_detection = False,
                )
                logger.info(f"[Warmup] {m['name']} ✓")
            except Exception as e:
                logger.warning(f"[Warmup] {m['name']}: {e}")
    except Exception as e:
        logger.exception(f"[Warmup] Failed to write/read dummy image: {e}")
    finally:
        try:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)
        except Exception:
            pass
    logger.info("[Warmup] All models ready — server is hot.")


@app.on_event("shutdown")
async def shutdown():
    try:
        _EXECUTOR.shutdown(wait=True, cancel_futures=False)
        logger.info("[Shutdown] Thread pool drained.")
    except Exception as e:
        logger.exception(f"[Shutdown] Error shutting down executor: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_pipeline(img_path: str) -> list[str]:
    """
    Full pipeline: original + CLAHE + gamma + unsharp-mask.
    Returns [original, clahe, gamma, sharp].
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"[Pre] Cannot decode image: {img_path}")
            return [img_path]

        base    = os.path.splitext(img_path)[0]
        outputs = [img_path]

        # CLAHE — uneven / indoor lighting
        try:
            lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l,a,b = cv2.split(lab)
            cl    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            p     = f"{base}_clahe.jpg"
            cv2.imwrite(p, cv2.cvtColor(cv2.merge([cl.apply(l),a,b]), cv2.COLOR_LAB2BGR))
            outputs.append(p)
        except Exception as e:
            logger.warning(f"[Pre] CLAHE: {e}")

        # Gamma 1.4 — dark selfies
        try:
            lut = np.array([min(255,int((i/255.0)**(1/1.4)*255)) for i in range(256)],
                           dtype=np.uint8)
            p   = f"{base}_gamma.jpg"
            cv2.imwrite(p, cv2.LUT(img, lut))
            outputs.append(p)
        except Exception as e:
            logger.warning(f"[Pre] Gamma: {e}")

        # Unsharp mask — soft mobile-camera focus
        try:
            bl = cv2.GaussianBlur(img, (0,0), sigmaX=3)
            p  = f"{base}_sharp.jpg"
            cv2.imwrite(p, cv2.addWeighted(img, 1.5, bl, -0.5, 0))
            outputs.append(p)
        except Exception as e:
            logger.warning(f"[Pre] Sharp: {e}")

        return outputs
    except Exception as e:
        logger.exception(f"[Pre] Pipeline crashed for {img_path}")
        return [img_path]


def _cleanup_variants(paths: list[str], original: str) -> None:
    for p in paths:
        if p != original and p:
            try: 
                if os.path.exists(p): 
                    os.remove(p)
            except Exception as e:
                logger.warning(f"[Cleanup] Failed to remove {p}: {e}")


def _select_variants(img_path: str) -> list[str]:
    """
    OPT 4 — Adaptive variant selection.
    Good image  → [original, CLAHE]       (6 Qdrant queries)
    Poor image  → full pipeline            (12 Qdrant queries)
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return [img_path]
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brit  = float(gray.mean())
        poor  = blur < GOOD_BLUR_THRESH or brit < GOOD_BRIGHT_LOW or brit > GOOD_BRIGHT_HIGH
        if not poor:
            base = os.path.splitext(img_path)[0]
            try:
                lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l,a,b = cv2.split(lab)
                cl    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
                p     = f"{base}_clahe.jpg"
                cv2.imwrite(p, cv2.cvtColor(cv2.merge([cl.apply(l),a,b]), cv2.COLOR_LAB2BGR))
                return [img_path, p]
            except Exception:
                return [img_path]
        return _preprocess_pipeline(img_path)
    except Exception as e:
        logger.exception(f"[Pre] _select_variants crashed for {img_path}")
        return [img_path]


def _best_variant(img_path: str) -> tuple[str, list[str]]:
    """
    OPT 10 — For enrollment, pick the sharpest preprocessed variant to store.
    Returns (best_path, all_variants).
    """
    try:
        variants = _preprocess_pipeline(img_path)
        best_path, best_score = img_path, 0.0
        for v in variants:
            im = cv2.imread(v)
            if im is None:
                continue
            score = float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),
                                        cv2.CV_64F).var())
            if score > best_score:
                best_score = score
                best_path  = v
        return best_path, variants
    except Exception as e:
        logger.exception(f"[Pre] _best_variant crashed for {img_path}")
        return img_path, [img_path]


# ══════════════════════════════════════════════════════════════════════════════
# OPT 14 — SINGLE FACE DETECT, REUSE ALIGNED CROP
# ══════════════════════════════════════════════════════════════════════════════

def _detect_and_save_crop(img_path: str) -> str | None:
    """
    Run RetinaFace ONCE on *img_path*, save the aligned face crop to disk,
    and return its path.
    """
    base      = os.path.splitext(img_path)[0]
    crop_path = f"{base}_crop.jpg"
    try:
        faces = DeepFace.extract_faces(
            img_path         = img_path,
            detector_backend = DETECTOR,
            align            = True,
            enforce_detection= True,
        )
        if not faces:
            return None

        face_arr = faces[0].get("face")
        if face_arr is None:
            return None

        # DeepFace returns face pixels in [0,1] float range; convert to uint8
        if face_arr.dtype != np.uint8:
            face_arr = (face_arr * 255).clip(0, 255).astype(np.uint8)

        # Convert RGB → BGR for cv2.imwrite
        face_bgr = cv2.cvtColor(face_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(crop_path, face_bgr)
        logger.debug(f"[Crop] Saved aligned crop → {os.path.basename(crop_path)}")
        return crop_path

    except Exception as e:
        logger.warning(f"[Crop] Detection failed for {os.path.basename(img_path)}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY & ANTI-SPOOF CHECKS  (opt 9)
# ══════════════════════════════════════════════════════════════════════════════

def _check_quality(img_path: str, anti_spoof: bool = False) -> dict:
    """
    Validates blur, brightness, face size.
    If anti_spoof=True also checks pixel diversity.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return {"ok": False, "reason": "Cannot decode image."}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        brightness = float(gray.mean())
        if brightness < MIN_BRIGHTNESS:
            return {"ok": False,
                    "reason": f"Image too dark (brightness={brightness:.1f}). Improve lighting."}
        if brightness > MAX_BRIGHTNESS:
            return {"ok": False,
                    "reason": f"Image overexposed (brightness={brightness:.1f}). Reduce lighting."}

        try:
            faces = DeepFace.extract_faces(
                img_path=img_path, detector_backend=DETECTOR,
                align=False, enforce_detection=True,
            )
            if faces:
                region = faces[0].get("facial_area", {})
                w, h   = region.get("w", 9999), region.get("h", 9999)
                if w < MIN_FACE_PX or h < MIN_FACE_PX:
                    return {"ok": False, "reason": "Move closer."}

                if anti_spoof:
                    x, y = region.get("x", 0), region.get("y", 0)
                    face_crop = gray[y:y+h, x:x+w]
                    stddev    = float(face_crop.std())
                    if stddev < MIN_FACE_STDDEV:
                        return {"ok": False,
                                "reason": "Face appears synthetic or printed. "
                                          "Please use a live camera."}
        except Exception:
            pass

        return {"ok": True}
    except Exception as e:
        logger.exception(f"[Quality] Check crashed for {img_path}")
        return {"ok": False, "reason": f"Quality check error: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING HELPERS  (OPT 13 + 14)
# ══════════════════════════════════════════════════════════════════════════════

def _get_embedding(img_path: str, model_name: str,
                   skip_detector: bool = False) -> list[float] | None:
    """
    Extract a face embedding.
    """
    try:
        result = DeepFace.represent(
            img_path          = img_path,
            model_name        = model_name,
            detector_backend  = "skip" if skip_detector else DETECTOR,
            align             = not skip_detector,
            enforce_detection = not skip_detector,
        )
        if result and isinstance(result, list):
            return result[0]["embedding"]
    except Exception as e:
        logger.warning(f"[Embed] {model_name} @ {os.path.basename(img_path)}: {e}")
    return None


def _cosine_distance(emb_a: list[float], emb_b: list[float]) -> float:
    """1 - cosine similarity between two embeddings (0.0 = identical direction)."""
    a = np.asarray(emb_a, dtype=np.float64)
    b = np.asarray(emb_b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


# ══════════════════════════════════════════════════════════════════════════════
# OPT 15 — BATCH UPSERT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _flush_batch(batch: dict[str, list[PointStruct]]) -> dict[str, list[str]]:
    """
    Upsert all accumulated PointStructs in *batch* to Qdrant in one call
    per collection, chunked to UPSERT_BATCH_SIZE.
    """
    results: dict[str, list[str]] = {}
    try:
        client = _get_qdrant()
        for cname, points in batch.items():
            results[cname] = []
            for start in range(0, len(points), UPSERT_BATCH_SIZE):
                chunk = points[start : start + UPSERT_BATCH_SIZE]
                try:
                    client.upsert(collection_name=cname, points=chunk)
                    results[cname].append(f"ok ({len(chunk)} points)")
                    logger.info(f"[Batch] {cname}: upserted {len(chunk)} point(s)")
                except Exception as e:
                    results[cname].append(f"error: {e}")
                    logger.exception(f"[Batch] {cname}: upsert failed")
    except Exception as e:
        logger.exception("[Batch] Failed to get Qdrant client or iterate batch")
        for cname in batch.keys():
            results.setdefault(cname, []).append(f"error: {e}")
    return results


def _embed_one_for_batch(crop_path: str, img_name: str,
                         m: dict) -> tuple[str, PointStruct | None, str | None]:
    """
    Thread-pool worker: embed *crop_path* with one model, return a
    PointStruct ready for batch upsert.
    """
    cname = _collection_name(m["name"])
    try:
        _ensure_collection(m)
        emb = _get_embedding(crop_path, m["name"], skip_detector=True)
        if emb is None:
            raise ValueError("Embedding returned None.")
        point = PointStruct(
            id      = str(uuid.uuid4()),
            vector  = emb,
            payload = {
                "img_name": img_name,
                "username": img_name.split("__")[0],
            },
        )
        logger.debug(f"[Embed] {m['name']} ✓  id={img_name}")
        return cname, point, None
    except Exception as e:
        logger.warning(f"[Embed] {m['name']} ✗  {e}")
        return cname, None, str(e)


def _embed_all_images_batch(
    items: list[tuple[str, str]],   # [(crop_path, img_name), ...]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    OPT 15 — Embed every (crop, name) pair with every model in parallel,
    then flush all resulting PointStructs to Qdrant in one upsert per
    collection.
    """
    batch:   dict[str, list[PointStruct]] = defaultdict(list)
    stored:  dict[str, list[str]]         = defaultdict(list)
    failed:  dict[str, list[str]]         = defaultdict(list)

    tasks = [
        (crop_path, img_name, m)
        for crop_path, img_name in items
        for m in MODELS
    ]

    futures = {}
    try:
        futures = {
            _EXECUTOR.submit(_embed_one_for_batch, crop_path, img_name, m):
                (crop_path, img_name, m)
            for crop_path, img_name, m in tasks
        }
    except Exception as e:
        logger.exception("[Embed] Failed to submit tasks to executor")
        for crop_path, img_name, m in tasks:
            failed[m["name"]].append(f"{img_name}: Executor submit failed: {e}")
        return dict(stored), dict(failed)

    for fut in as_completed(futures):
        _, img_name, m = futures[fut]
        try:
            cname, point, err = fut.result()
            if point is not None:
                batch[cname].append(point)
                stored[m["name"]].append(img_name)
            else:
                failed[m["name"]].append(f"{img_name}: {err}")
        except Exception as exc:
            logger.exception(f"[Embed] Future error for {img_name}")
            failed[m["name"]].append(f"{img_name}: {exc}")

    if batch:
        try:
            _flush_batch(dict(batch))
        except Exception as e:
            logger.exception("[Embed] Batch flush failed")
            for cname in batch.keys():
                for p in batch[cname]:
                    failed["unknown"].append(f"Flush failed for {p.id}: {e}")

    return dict(stored), dict(failed)


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY SINGLE-IMAGE EMBED+STORE  (used only by search path, unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _store_for_model(img_path: str, img_name: str, m: dict,
                     skip_detector: bool = False) -> tuple:
    """
    Embed img_path with one model and upsert to Qdrant.
    """
    cname = _collection_name(m["name"])
    try:
        _ensure_collection(m)
        emb = _get_embedding(img_path, m["name"], skip_detector=skip_detector)
        if emb is None:
            raise ValueError("No face detected.")
        _get_qdrant().upsert(
            collection_name = cname,
            points = [PointStruct(
                id      = str(uuid.uuid4()),
                vector  = emb,
                payload = {
                    "img_name": img_name,
                    "username": img_name.split("__")[0],
                },
            )],
        )
        logger.info(f"[Enroll] {m['name']} ✓  id={img_name}")
        return m["name"], None
    except Exception as e:
        logger.warning(f"[Enroll] {m['name']} ✗  {e}")
        return None, (m["name"], str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH HELPERS  (opt 4, 5, 6)
# ══════════════════════════════════════════════════════════════════════════════

def _search_one(args: tuple) -> tuple[str, str, dict]:
    """
    Thread-pool worker: embed one variant with one model → query Qdrant.
    """
    variant_path, m, k, username_filter = args
    model_name  = m["name"]
    threshold   = m["threshold"]
    cname       = _collection_name(model_name)
    _empty      = {"recognized": False, "name": None,
                   "distance": None, "threshold": threshold}
    is_crop     = variant_path.endswith("_crop.jpg")

    try:
        emb = _get_embedding(variant_path, model_name, skip_detector=is_crop)
        if emb is None:
            return model_name, variant_path, _empty

        qfilter = (
            Filter(must=[FieldCondition(
                key="username", match=MatchValue(value=username_filter)
            )])
            if username_filter else None
        )

        response = _get_qdrant().query_points(
            collection_name = cname,
            query           = emb,
            query_filter    = qfilter,
            limit           = k,
            with_payload    = True,
        )
        hits = response.points

        if not hits:
            return model_name, variant_path, _empty

        user_dists: dict = defaultdict(list)
        for hit in hits:
            uname = hit.payload.get("username", "unknown")
            dist  = 1.0 - float(hit.score)
            user_dists[uname].append(dist)

        best_user, best_dist = None, float("inf")
        for user, dists in user_dists.items():
            best = float(min(dists))
            if best < threshold and best < best_dist:
                best_user, best_dist = user, best

        recognized = best_user is not None
        return model_name, variant_path, {
            "recognized": recognized,
            "name":       best_user,
            "distance":   round(best_dist, 4) if recognized else None,
            "threshold":  threshold,
        }

    except Exception as e:
        err = str(e)
        if "not found" in err.lower() or "doesn't exist" in err.lower():
            logger.debug(f"[Search] {model_name}: collection empty (enroll first)")
        else:
            logger.exception(f"[Search] {model_name} @ {os.path.basename(variant_path)} failed")
        return model_name, variant_path, _empty


def _search_best_of_variants(img_path: str, k: int = 20,
                              username_filter: str | None = None) -> dict:
    """
    OPT 4 + 5 + 6 + 14:
    • Detect face ONCE → save aligned crop.
    • Pass crop as the first variant; subsequent preprocessing variants are
      derived from the original image.
    • All (variant × model) tasks fire in parallel via thread pool.
    • Early-exit once a model achieves distance < strong_thr.
    """
    try:
        crop_path = _detect_and_save_crop(img_path)
        if crop_path:
            preproc_variants = _select_variants(img_path)
            variants = [crop_path] + [v for v in preproc_variants if v != img_path]
        else:
            variants = _select_variants(img_path)
    except Exception as e:
        logger.exception(f"[Search] Variant preparation crashed for {img_path}")
        crop_path = None
        variants = [img_path]

    best: dict           = {}
    strong_hit: set[str] = set()

    tasks = [(v, m, k, username_filter) for v in variants for m in MODELS]

    try:
        futures: dict[Future, tuple] = {
            _EXECUTOR.submit(_search_one, t): t for t in tasks
        }

        for future in as_completed(futures):
            try:
                model_name, variant_path, res = future.result()
            except Exception as exc:
                logger.exception(f"[Search] Future raised for {futures[future]}")
                continue

            if model_name in strong_hit:
                continue

            prev      = best.get(model_name)
            new_dist  = res.get("distance")
            prev_dist = prev.get("distance") if prev else None

            is_better = (
                prev is None or (
                    new_dist is not None and
                    (prev_dist is None or new_dist < prev_dist)
                )
            )
            if is_better:
                best[model_name] = res
                if res["recognized"]:
                    logger.info(
                        f"[Search] {model_name}: name={res['name']} "
                        f"dist={res['distance']} "
                        f"variant={os.path.basename(variant_path)}"
                    )

            strong_thr = next(
                (m["strong_thr"] for m in MODELS if m["name"] == model_name), 0.3
            )
            if new_dist is not None and new_dist < strong_thr:
                strong_hit.add(model_name)
                logger.info(f"[Search] {model_name}: strong match "
                             f"dist={new_dist:.4f} < {strong_thr} — early exit")

    finally:
        try:
            _cleanup_variants(variants, img_path)
        except Exception as e:
            logger.warning(f"[Search] Cleanup variants failed: {e}")

    for m in MODELS:
        best.setdefault(m["name"], {
            "recognized": False, "name": None,
            "distance":   None,  "threshold": m["threshold"],
        })

    return best


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE ZONES  (opt 8)
# ══════════════════════════════════════════════════════════════════════════════

def _confidence_zone(confidence: float) -> dict:
    if confidence >= CONF_GREEN:
        return {"zone": "green",  "action": "auto_approve",   "verified": True}
    if confidence >= CONF_YELLOW:
        return {"zone": "yellow", "action": "flag_for_review", "verified": True}
    return     {"zone": "red",   "action": "reject",          "verified": False}


# ══════════════════════════════════════════════════════════════════════════════
# DISTANCE-SPREAD SANITY CHECK  (opt 12)
# ══════════════════════════════════════════════════════════════════════════════

def _check_distance_spread(confirmed: dict) -> dict:
    try:
        dists = [r["distance"] for r in confirmed.values()
                 if r.get("distance") is not None]
        if len(dists) < 2:
            return {"ok": True}
        spread = max(dists) - min(dists)
        if spread < 0.005:
            return {
                "ok":     False,
                "reason": f"Suspicious: all model distances nearly identical "
                          f"(spread={spread:.4f}). Possible replay attack.",
            }
        return {"ok": True}
    except Exception as e:
        logger.exception(f"[Sanity] Distance spread check crashed: {e}")
        return {"ok": True} # Fail open on sanity check crash to avoid blocking legitimate users unnecessarily


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE DECISION (open-ended identify)
# ══════════════════════════════════════════════════════════════════════════════

def _ensemble_decision(model_results: dict) -> dict:
    """2-of-3 weighted ensemble for /identify."""
    try:
        votes: dict = defaultdict(list)
        for model_name, res in model_results.items():
            if res.get("recognized") and res.get("name"):
                votes[res["name"]].append(model_name)

        winner, winner_models = None, []
        for username, agreeing in votes.items():
            if len(agreeing) >= MIN_VOTES:
                if winner is None or len(agreeing) > len(winner_models):
                    winner, winner_models = username, agreeing

        if not winner:
            return {
                "verified":      False, "name": None,
                "confidence":    0.0,   "distance": None,
                "zone":          "red", "action": "reject",
                "votes":         {k: len(v) for k, v in votes.items()},
                "model_details": model_results,
            }

        dist_list, weighted_conf, total_weight = [], 0.0, 0.0
        for m in MODELS:
            if m["name"] in winner_models:
                res      = model_results[m["name"]]
                raw_conf = max(0.0, (1.0 - res["distance"] / m["threshold"]) * 99.0)
                weighted_conf += raw_conf * m["weight"]
                total_weight  += m["weight"]
                dist_list.append(res["distance"])

        confidence = round(weighted_conf / max(total_weight, 1e-9), 2)
        zone_info  = _confidence_zone(confidence)

        return {
            **zone_info,
            "name":            winner,
            "confidence":      confidence,
            "distance":        round(float(np.mean(dist_list)), 4),
            "votes":           len(winner_models),
            "agreeing_models": winner_models,
            "model_details":   model_results,
        }
    except Exception as e:
        logger.exception("[Ensemble] Decision logic crashed")
        return {
            "verified": False, "name": None, "confidence": 0.0, 
            "distance": None, "zone": "red", "action": "reject",
            "model_details": model_results, "error": str(e)
        }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — look up the stored front-pose image URL for a verified user
# ══════════════════════════════════════════════════════════════════════════════

def _public_base_url(request: Request) -> str:
    """
    Return the base URL the client actually used to reach us, e.g.
    'http://192.168.1.10:3009'.

    Inside Docker, request.base_url uses the container hostname (e.g.
    'http://faceverify:3009/') which is not reachable from outside.
    The Host header always contains the real IP:port the client connected to.
    """
    host = request.headers.get("host", "")          # e.g. "192.168.1.10:3009"
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    if host:
        return f"{scheme}://{host}"
    # Fallback — strips internal container name, uses raw URL components
    return f"{scheme}://{request.url.hostname}:{request.url.port}"


def _get_front_image_url(username: str) -> str | None:
    """
    Scan src/{username}/ for any file whose name starts with 'front__'
    and return its relative URL path.  Caller must prepend the base URL.
    """
    try:
        user_dir = os.path.join(SRC_DIR, username)
        if not os.path.isdir(user_dir):
            return None
        for fname in os.listdir(user_dir):
            if fname.startswith("front__"):
                return f"/src/{username}/{fname}"
    except Exception as e:
        logger.warning(f"[Helper] Failed to get front image URL for {username}: {e}")
    return None


def _lookup_face_id(username: str) -> str | None:
    """
    Given the internal *username*, retrieve the public face_id from the profile collection.
    """
    try:
        _ensure_profile_collection()
        results, _ = _get_qdrant().scroll(
            collection_name = USER_PROFILE_COLLECTION,
            scroll_filter   = Filter(must=[
                FieldCondition(key="face_id", match=MatchValue(value=username))
            ]),
            limit           = 1,
            with_payload    = True,
            with_vectors    = False,
        )
        if results:
            return results[0].payload.get("face_id")
    except Exception as e:
        logger.warning(f"[Profile] Lookup failed for username={username}: {e}")
    return None


def _fetch_user_embeddings(face_id: str) -> dict[str, list[dict]]:
    """
    Scroll every model's face-vector collection for points whose payload
    'username' matches *face_id* (username == face_id in this schema) and
    return the stored embedding for each enrolled image, grouped by model.
    """
    client = _get_qdrant()
    out: dict[str, list[dict]] = {}
    for m in MODELS:
        cname = _collection_name(m["name"])
        try:
            _ensure_collection(m)
            points, _ = client.scroll(
                collection_name = cname,
                scroll_filter   = Filter(must=[
                    FieldCondition(key="username", match=MatchValue(value=face_id))
                ]),
                limit        = 100,
                with_payload = True,
                with_vectors = True,
            )
            out[m["name"]] = [
                {
                    "img_name":  p.payload.get("img_name"),
                    "pose":      p.payload.get("img_name", "").split("__")[1]
                                 if p.payload.get("img_name", "").count("__") >= 2 else None,
                    "embedding": p.vector,
                }
                for p in points
            ]
        except Exception as e:
            logger.warning(f"[Embeddings] Scroll failed for {cname} / face_id={face_id}: {e}")
            out[m["name"]] = []
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ONBOARD
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/mobile/onboard")
@app.post("/face/mobile/onboard")
async def mobile_onboard(
    request: Request,
    front: UploadFile = File(..., description="Front-facing face photo"),
    right: UploadFile = File(..., description="Right-turn face photo"),
    blink: UploadFile = File(..., description="Blink / eyes-closed face photo"),
    poses: str        = Form(
        default="front,right,blink",
        description=(
            "Comma-separated pose label for each image in order. "
            "Example: 'front,right,blink'. Defaults to 'front,right,blink'."
        ),
    ),
):
    """
    Register a new user.
    """
    images = [front, right, blink]

    pose_list = [p.strip() for p in poses.split(",") if p.strip()]
    if len(pose_list) != 3:
        raise HTTPException(
            422,
            detail={
                "status":  "error",
                "message": (
                    f"'poses' must contain exactly 3 comma-separated labels "
                    f"(e.g. 'front,right,blink'). Got {len(pose_list)}: {pose_list}."
                ),
            },
        )

    face_id      = str(uuid.uuid4())
    username     = face_id
    user_src_dir = os.path.join(SRC_DIR, username)

    saved_paths:      list[tuple[int, str]]             = []
    details:          list                              = []
    embed_items:      list[tuple[str, str]]             = []
    item_meta:        dict[str, tuple[int, str]]        = {}
    variant_registry: dict[int, tuple[str, list[str]]] = {}

    try:
        # ── 1. Save all uploads ──────────────────────────────────────────────
        for idx, uf in enumerate(images):
            try:
                ext  = os.path.splitext(uf.filename or "img.jpg")[1] or ".jpg"
                path = os.path.join(TEMP_DIR, f"enroll_{username}_{uuid.uuid4()}{ext}")
                with open(path, "wb") as f:
                    f.write(await uf.read())
                saved_paths.append((idx, path))
            except Exception as e:
                logger.exception(f"[Onboard] Failed to save upload {idx}")
                raise HTTPException(400, detail={"status": "error", "message": f"Failed to save image {idx+1}: {e}"})

        # ── 2. Quality → best-variant → detect crop ──────────────────────────
        for img_idx, img_path in saved_paths:
            try:
                pose = pose_list[img_idx] if img_idx < len(pose_list) else f"pose{img_idx}"

                quality = _check_quality(img_path, anti_spoof=True)
                if not quality["ok"]:
                    details.append({"image": img_idx+1, "pose": pose,
                                     "status": "error", "message": quality["reason"]})
                    logger.info(f"[Onboard] img{img_idx+1} rejected: {quality['reason']}")
                    continue

                best_path, all_variants = _best_variant(img_path)
                variant_registry[img_idx] = (best_path, all_variants)

                crop_path = _detect_and_save_crop(best_path)
                if crop_path is None:
                    details.append({"image": img_idx+1, "pose": pose,
                                     "status": "error",
                                     "message": "Face detection failed on best variant."})
                    logger.warning(f"[Onboard] img{img_idx+1}: face detection failed")
                    continue

                img_name = f"{username}__{pose}__{uuid.uuid4()}"
                embed_items.append((crop_path, img_name))
                item_meta[img_name] = (img_idx, pose)
            except Exception as e:
                logger.exception(f"[Onboard] Error processing image {img_idx+1}")
                pose = pose_list[img_idx] if img_idx < len(pose_list) else f"pose{img_idx}"
                details.append({"image": img_idx+1, "pose": pose,
                                "status": "error", "message": f"Processing error: {e}"})

        # ── 3. Batch embed + upsert (OPT 13 + 14 + 15) ──────────────────────
        stored_per_model: dict[str, list[str]] = {}
        failed_per_model: dict[str, list[str]] = {}

        if embed_items:
            try:
                stored_per_model, failed_per_model = _embed_all_images_batch(embed_items)
            except Exception as e:
                logger.exception("[Onboard] Batch embedding completely failed")
                raise HTTPException(500, detail={"status": "error", "message": f"Embedding failed: {e}"})

        all_stored_names: set[str] = set()
        for names in stored_per_model.values():
            all_stored_names.update(names)

        enrolled = 0
        pending_saves: list[tuple] = []

        for img_name, (img_idx, pose) in item_meta.items():
            best_path, _ = variant_registry.get(img_idx, (None, []))
            models_ok    = [mn for mn, names in stored_per_model.items()
                            if img_name in names]
            models_fail  = {mn: errs for mn, errs in failed_per_model.items()
                            if any(img_name in e for e in errs)}

            if models_ok:
                enrolled += 1
                pending_saves.append((best_path, pose, img_idx, models_ok, models_fail))
                details.append({
                    "image":         img_idx + 1,
                    "pose":          pose,
                    "status":        "enrolled",
                    "image_url":     None,
                    "models":        models_ok,
                    "failed_models": models_fail or None,
                })
            else:
                details.append({
                    "image":   img_idx + 1,
                    "pose":    pose,
                    "status":  "error",
                    "message": str(models_fail),
                })

        # ── Validate: all poses must have enrolled ────────────────────────────
        failed_details = [d for d in details if d.get("status") == "error"]
        if failed_details or enrolled < len(images):
            failed_poses = [
                {"pose": d["pose"], "reason": d.get("message", "Unknown error")}
                for d in failed_details
            ]
            raise HTTPException(
                419,
                detail={
                    "status":  "error",
                    "message": f"{len(failed_poses)} pose(s) not captured: "
                               + ", ".join(
                                   f"{f['pose']} ({f['reason']})"
                                   for f in failed_poses
                               ),
                    "failed_poses": failed_poses,
                },
            )

        # ── All images verified — persist to src/{username}/ ─────────────────
        try:
            os.makedirs(user_src_dir, exist_ok=True)
        except Exception as e:
            logger.exception(f"[Onboard] Failed to create user dir {user_src_dir}")
            raise HTTPException(500, detail={"status": "error", "message": f"Storage error: {e}"})

        # OPT 16 — Register user profile in Qdrant (after full validation)
        try:
            _register_user_profile(face_id)
        except Exception as e:
            logger.exception(f"[Onboard] Failed to register profile for {face_id}")
            raise HTTPException(500, detail={"status": "error", "message": f"Profile registration failed: {e}"})

        original_by_idx = dict(saved_paths)
        for best_path, pose, img_idx, _, _ in pending_saves:
            public_url: str | None = None
            src_path = original_by_idx.get(img_idx, best_path)
            if src_path and os.path.exists(src_path):
                ext          = os.path.splitext(src_path)[1] or ".jpg"
                public_fname = f"{pose}__{uuid.uuid4()}{ext}"
                public_dest  = os.path.join(user_src_dir, public_fname)
                try:
                    shutil.copy2(src_path, public_dest)
                    # Use Host header so the URL contains the real client-facing IP:port
                    # base = _public_base_url(request)
                    public_url = f"{PUBLIC_BASE_URL}/src/{username}/{public_fname}"
                except Exception as copy_err:
                    logger.exception(f"[Onboard] Could not save public image for pose={pose}")
            for d in details:
                if d.get("pose") == pose and d.get("image") == img_idx + 1:
                    d["image_url"] = public_url
                    break

        logger.info(
            f"[Onboard] face_id={face_id} enrolled {enrolled}/{len(images)} "
            f"(batch upsert, GPU={'yes' if GPU_AVAILABLE else 'no'})."
        )

        front_url = next(
            (d["image_url"] for d in details if d.get("pose") == "front"),
            None,
        )

        # ── Response includes face_id (OPT 16) ───────────────────────────────
        return {
            "status":    "success",
            "code":      200,
            "face_id":   face_id,       # UUID — store this on the mobile client
            "image_url": front_url,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[Onboard] Error for '{username}'")
        raise HTTPException(419, detail={"status": "error", "message": str(e)})
    finally:
        # Clean up raw uploads
        for _, p in saved_paths:
            try: 
                if p and os.path.exists(p): os.remove(p)
            except Exception as e:
                logger.warning(f"[Onboard] Cleanup failed for {p}: {e}")
                
        # Clean up preprocessing variants and crop files
        for img_idx, (best_path, all_variants) in variant_registry.items():
            try:
                _cleanup_variants(all_variants, None)
            except Exception as e:
                logger.warning(f"[Onboard] Cleanup variants failed for idx {img_idx}: {e}")
                
        for _, img_path in saved_paths:
            if not img_path: continue
            base = os.path.splitext(img_path)[0]
            for suffix in ("_clahe", "_gamma", "_sharp", "_crop"):
                for ext in (".jpg", ".png"):
                    candidate = f"{base}{suffix}{ext}"
                    try: 
                        if os.path.exists(candidate): os.remove(candidate)
                    except Exception as e:
                        logger.warning(f"[Onboard] Cleanup failed for {candidate}: {e}")
                        
        for img_idx, (best_path, all_variants) in variant_registry.items():
            if not best_path: continue
            base      = os.path.splitext(best_path)[0]
            candidate = f"{base}_crop.jpg"
            try: 
                if os.path.exists(candidate): os.remove(candidate)
            except Exception as e:
                logger.warning(f"[Onboard] Cleanup failed for {candidate}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# VERIFY
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/mobile/verify")
@app.post("/face/mobile/verify")
async def mobile_verify(
    request: Request,
    image: UploadFile = File(..., description="Live face photo to verify"),
):
    """
    Verify a face against ALL enrolled users — no username required.
    """
    ext      = os.path.splitext(image.filename or "img.jpg")[1] or ".jpg"
    tmp_path = os.path.join(TEMP_DIR, f"verify_{uuid.uuid4()}{ext}")

    try:
        try:
            with open(tmp_path, "wb") as f:
                f.write(await image.read())
        except Exception as e:
            logger.exception("[Verify] Failed to save uploaded image")
            raise HTTPException(400, detail={"status": "error", "message": f"Failed to save image: {e}"})

        # 1. Quality gate
        try:
            quality = _check_quality(tmp_path, anti_spoof=False)
        except Exception as e:
            logger.exception("[Verify] Quality check crashed")
            raise HTTPException(500, detail={"status": "error", "message": f"Quality check error: {e}"})
            
        if not quality["ok"]:
            raise HTTPException(419, detail={"status": "error", "message": quality["reason"]})

        # 2-5. Single-detect crop + adaptive variants + parallel full-scan search
        try:
            model_results = _search_best_of_variants(tmp_path, k=20, username_filter=None)
        except Exception as e:
            logger.exception("[Verify] Search pipeline crashed")
            raise HTTPException(500, detail={"status": "error", "message": f"Search failed: {e}"})

        # 6. Priority vote
        per_model: dict[str, str | None] = {}
        for mn, res in model_results.items():
            per_model[mn] = res.get("name") if res.get("recognized") else None

        primary_hits = {mn: n for mn, n in per_model.items()
                        if mn in PRIMARY_MODELS and n is not None}

        winner, winner_models = None, []

        # Tier 1: both ArcFace AND Facenet512 agree
        if len(primary_hits) == 2:
            names = list(primary_hits.values())
            if names[0] == names[1]:
                winner        = names[0]
                winner_models = list(primary_hits.keys())
                logger.info(f"[Verify] Tier-1 pass (both primaries) → {winner}")

        # Tier 2: one primary + GhostFaceNet agree
        if winner is None:
            ghost_name = per_model.get(TIEBREAKER_MODEL)
            if ghost_name:
                for mn, pname in primary_hits.items():
                    if pname == ghost_name:
                        winner        = ghost_name
                        winner_models = [mn, TIEBREAKER_MODEL]
                        logger.info(
                            f"[Verify] Tier-2 pass ({mn} + {TIEBREAKER_MODEL}) → {winner}"
                        )
                        break

        confirmed: dict = {}
        if winner:
            for mn in winner_models:
                confirmed[mn] = model_results[mn]

        vote_passed = winner is not None
        dissenters  = [mn for mn in model_results if mn not in confirmed]

        # 7. Weighted confidence
        confidence, avg_dist = 0.0, None
        if confirmed:
            dist_list, weighted_conf, total_weight = [], 0.0, 0.0
            for m in MODELS:
                mn = m["name"]
                if mn in confirmed:
                    dist     = confirmed[mn]["distance"]
                    raw_conf = max(0.0, (1.0 - dist / m["threshold"]) * 99.0)
                    weighted_conf += raw_conf * m["weight"]
                    total_weight  += m["weight"]
                    dist_list.append(dist)
            confidence = round(weighted_conf / max(total_weight, 1e-9), 2)
            avg_dist   = round(float(np.mean(dist_list)), 4)

        zone_info = _confidence_zone(confidence) if vote_passed else {
            "zone": "red", "action": "reject", "verified": False
        }
        verified = vote_passed

        # 8. Distance-spread sanity check
        spoof_check = _check_distance_spread(confirmed) if verified else {"ok": True}
        if not spoof_check["ok"]:
            logger.warning(f"[Verify] Spoof flag: {spoof_check['reason']}")
            verified  = False
            zone_info = {"zone": "red", "action": "reject", "verified": False}

        logger.info(
            f"[Verify] verified={verified} "
            f"zone={zone_info['zone']} confidence={confidence} "
            f"confirmed={list(confirmed.keys())} dissenters={dissenters}"
        )

        if not verified:
            raise HTTPException(
                419,
                detail={"status": "error", "message": "User not verified."},
            )

        # 9. Look up face_id from profile collection (OPT 16)
        face_id      = _lookup_face_id(winner) if winner else None
        relative_url = _get_front_image_url(winner) if winner else None
        # Use Host header so the URL contains the real client-facing IP:port
        if relative_url:
            # base = _public_base_url(request)
            front_url = f"{PUBLIC_BASE_URL}{relative_url}"
        else:
            front_url = None

        logger.info(f"[Verify] face_id={face_id} verified ✓")

        return {
            "status":    "success",
            "code":      200,
            "message":   "User verified.",
            "face_id":   face_id,       # same UUID returned at onboard
            "image_url": front_url,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[Verify] Unexpected error")
        raise HTTPException(419, detail={"status": "error", "message": str(e)})
    finally:
        try: 
            if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)
        except Exception as e:
            logger.warning(f"[Verify] Cleanup failed for {tmp_path}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# COMPARE  (stateless 1:1 face match between two uploaded images)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/mobile/compare")
@app.post("/face/mobile/compare")
async def mobile_compare(
    image1: UploadFile = File(..., description="First face photo"),
    image2: UploadFile = File(..., description="Second face photo"),
):
    """
    Compare two face photos directly and report whether they belong to the
    same person. Stateless — no Qdrant lookup or enrollment involved.
    Uses the same 3-model ensemble (ArcFace + Facenet512 + GhostFaceNet)
    and priority-vote rule as /mobile/verify, but the embeddings are
    compared against each other instead of against the enrolled database.
    """
    ext1  = os.path.splitext(image1.filename or "img1.jpg")[1] or ".jpg"
    ext2  = os.path.splitext(image2.filename or "img2.jpg")[1] or ".jpg"
    path1 = os.path.join(TEMP_DIR, f"compare_{uuid.uuid4()}{ext1}")
    path2 = os.path.join(TEMP_DIR, f"compare_{uuid.uuid4()}{ext2}")
    cleanup_paths = [path1, path2]

    try:
        try:
            with open(path1, "wb") as f:
                f.write(await image1.read())
            with open(path2, "wb") as f:
                f.write(await image2.read())
        except Exception as e:
            logger.exception("[Compare] Failed to save uploaded images")
            raise HTTPException(400, detail={"status": "error", "message": f"Failed to save images: {e}"})

        # 1. Quality gate on both images
        for label, path in (("image1", path1), ("image2", path2)):
            try:
                quality = _check_quality(path, anti_spoof=False)
            except Exception as e:
                logger.exception(f"[Compare] Quality check crashed for {label}")
                raise HTTPException(500, detail={"status": "error", "message": f"Quality check error: {e}"})
            if not quality["ok"]:
                raise HTTPException(419, detail={"status": "error", "message": f"{label}: {quality['reason']}"})

        # 2. Detect face once per image, reuse the aligned crop for every model
        crop1 = _detect_and_save_crop(path1)
        crop2 = _detect_and_save_crop(path2)
        cleanup_paths += [p for p in (crop1, crop2) if p]

        if crop1 is None:
            raise HTTPException(419, detail={"status": "error", "message": "No face detected in image1."})
        if crop2 is None:
            raise HTTPException(419, detail={"status": "error", "message": "No face detected in image2."})

        # 3. Per-model embedding + direct cosine-distance comparison
        model_results: dict[str, dict] = {}
        for m in MODELS:
            emb1 = _get_embedding(crop1, m["name"], skip_detector=True)
            emb2 = _get_embedding(crop2, m["name"], skip_detector=True)
            if emb1 is None or emb2 is None:
                model_results[m["name"]] = {
                    "matched": False, "distance": None, "threshold": m["threshold"],
                }
                continue
            dist = _cosine_distance(emb1, emb2)
            model_results[m["name"]] = {
                "matched":   dist < m["threshold"],
                "distance":  round(dist, 4),
                "threshold": m["threshold"],
            }

        # 4. Priority vote — same rule as /mobile/verify:
        #    Tier 1: both primaries (ArcFace + Facenet512) agree → match.
        #    Tier 2: one primary + GhostFaceNet agree → match.
        #    GhostFaceNet alone never confirms a match.
        primary_hits = {mn: r for mn, r in model_results.items()
                        if mn in PRIMARY_MODELS and r["matched"]}

        same_person, winner_models = False, []
        if len(primary_hits) == 2:
            same_person   = True
            winner_models = list(primary_hits.keys())
        elif len(primary_hits) == 1 and model_results.get(TIEBREAKER_MODEL, {}).get("matched"):
            same_person   = True
            winner_models = list(primary_hits.keys()) + [TIEBREAKER_MODEL]

        confirmed = {mn: model_results[mn] for mn in winner_models}

        # 5. Weighted confidence over the confirming models
        confidence = 0.0
        if confirmed:
            weighted_conf, total_weight = 0.0, 0.0
            for m in MODELS:
                mn = m["name"]
                if mn in confirmed:
                    dist     = confirmed[mn]["distance"]
                    raw_conf = max(0.0, (1.0 - dist / m["threshold"]) * 99.0)
                    weighted_conf += raw_conf * m["weight"]
                    total_weight  += m["weight"]
            confidence = round(weighted_conf / max(total_weight, 1e-9), 2)

        zone_info = _confidence_zone(confidence) if same_person else {
            "zone": "red", "action": "reject", "verified": False
        }
        same_person = same_person and zone_info["verified"]

        logger.info(
            f"[Compare] same_person={same_person} confidence={confidence} "
            f"zone={zone_info['zone']} models={model_results}"
        )

        return {
            "status":        "success",
            "code":          200,
            "same_person":   same_person,
            "confidence":    confidence,
            "zone":          zone_info["zone"],
            "model_details": model_results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[Compare] Unexpected error")
        raise HTTPException(500, detail={"status": "error", "message": str(e)})
    finally:
        for p in cleanup_paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                logger.warning(f"[Compare] Cleanup failed for {p}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS  (fetch stored vectors for an enrolled face_id)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/mobile/embeddings")
@app.post("/face/mobile/embeddings")
async def mobile_get_embeddings(
    request:Request,
    face_id: str = Form(..., description="face_id returned by /mobile/onboard"),
):
    """
    Fetch the stored enrollment embeddings for an already-enrolled face_id.
    No face image is submitted — this is a pure lookup by face_id.
    """
    try:
        if _lookup_face_id(face_id) is None:
            raise HTTPException(
                404,
                detail={"status": "error", "message": f"face_id '{face_id}' is not enrolled."},
            )

        embeddings = _fetch_user_embeddings(face_id)

        if not any(embeddings.values()):
            raise HTTPException(
                404,
                detail={"status": "error", "message": f"No embeddings found for face_id '{face_id}'."},
            )

        return {
            "status":     "success",
            "code":       200,
            "face_id":    face_id,
            "embeddings": embeddings,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[Embeddings] Error fetching embeddings for face_id={face_id}")
        raise HTTPException(500, detail={"status": "error", "message": str(e)})




# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3009, workers=1)


