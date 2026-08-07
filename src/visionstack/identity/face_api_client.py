"""Client for the external face-embedding service used by employee onboarding.

Stands in for a local RetinaFace/ArcFace model (see face_detector.py / face_embedder.py TODOs
and the README's identity/ phase status) until that's built. Enrollment photos are sent to a
separately-hosted face API which does its own detection/pose validation and returns a face_id;
a second call exchanges that face_id for the actual embedding vectors, which callers persist into
`employee_face_embeddings` (pgvector) themselves — this module has no DB dependency.

Confirmed against a real successful onboard+embeddings call:
    POST /face/mobile/onboard   -> {"status": "success", "code": 200, "face_id": "...", ...}
    POST /face/mobile/embeddings -> {
        "status": "success", "code": 200, "face_id": "...",
        "embeddings": {
            "ArcFace":      [{"img_name": ..., "pose": "front", "embedding": [512 floats]}, ...],
            "Facenet512":   [...],
            "GhostFaceNet": [...],
        },
    }
The service runs 3 independent embedding models and returns one vector per pose per model; our
schema has a single 512-d `employee_face_embeddings.embedding` column (see db/vector.py), so we
store one model's vector, not all three. ArcFace is what identity/face_embedder.py's own TODO
already names as the target model, so it's the one picked here (EMBEDDING_MODEL below).
"""
from __future__ import annotations

import logging

import httpx

from visionstack.common.config import get_settings
from visionstack.common.errors import FaceApiError

logger = logging.getLogger("visionstack.identity.face_api_client")

TIMEOUT_SECONDS = 30.0
EMBEDDING_MODEL = "ArcFace"
PREFERRED_POSE = "front"


def onboard_face(front: bytes, right: bytes, left: bytes) -> str:
    """Submits the 3 enrollment photos and returns the service's face_id for this person.

    The server's third multipart field is literally named "blink" (POST /mobile/onboard
    signature), not "left" -- it's just a generic third-pose slot with no actual blink/liveness
    check behind it, so our left-turn photo is submitted under that field name. The `poses` form
    field is a separate, freeform label used only for display/error messages, so it's set to the
    photo's real pose ("left") rather than the server's default ("blink").
    """
    files = {
        "front": ("front.jpg", front, "image/jpeg"),
        "right": ("right.jpg", right, "image/jpeg"),
        "blink": ("left.jpg", left, "image/jpeg"),
    }
    response = _post("/face/mobile/onboard", files=files, data={"poses": "front,right,left"})
    return _extract_face_id(response)


def get_embedding(face_id: str) -> list[float]:
    """Exchanges a face_id (from onboard_face) for its ArcFace embedding vector."""
    response = _post("/face/mobile/embeddings", data={"face_id": face_id})
    return _extract_embedding(response)


def verify_face(image: bytes) -> str | None:
    """Verifies a live face crop against every enrolled face. Returns the matched face_id, or
    None if there's nothing we can attribute to an employee -- the service's own /verify returns
    HTTP 419 "User not verified" for a genuine no-match, which is a normal outcome here (an
    unrecognized person walking past a camera), not an error condition like the other calls'
    failures are.

    Confirmed live that a *successful* match can still come back with a null face_id --
    {"status": "success", "message": "User verified.", "face_id": None, ...} -- apparently the
    service's own face_id lookup for the matched profile can fail independently of the match
    itself. That's still not attributable to an employee on our end, so it's treated the same as
    "no match" rather than raised as an error (see _extract_face_id, which is stricter and used
    by onboard_face, where a missing face_id genuinely is a failure).
    """
    files = {"image": ("verify.jpg", image, "image/jpeg")}
    resp = _request("/face/mobile/verify", files=files)
    if resp.status_code == 419:
        return None
    response = _json_or_raise(resp)
    face_id = response.get("face_id")
    if isinstance(face_id, str):
        return face_id
    if response.get("status") == "success":
        logger.info("verify_face: service reported a match but returned no face_id: %r", response)
    return None


def _request(path: str, **kwargs) -> httpx.Response:
    base_url = get_settings().face_api_base_url
    try:
        with httpx.Client(base_url=base_url, timeout=TIMEOUT_SECONDS) as client:
            return client.post(path, **kwargs)
    except httpx.HTTPError as e:
        raise FaceApiError(f"could not reach face API at {base_url}{path}: {e}") from e


def _json_or_raise(resp: httpx.Response) -> dict:
    if resp.status_code >= 400:
        raise FaceApiError(_error_message(resp))
    try:
        return resp.json()
    except ValueError as e:
        raise FaceApiError(f"face API returned non-JSON response: {resp.text[:500]}") from e


def _post(path: str, **kwargs) -> dict:
    return _json_or_raise(_request(path, **kwargs))


def _error_message(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return f"face API returned {resp.status_code}: {resp.text[:500]}"

    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, dict) and "failed_poses" in detail:
        reasons = ", ".join(f"{p['pose']} ({p['reason']})" for p in detail["failed_poses"])
        return f"Face capture failed: {reasons}"
    return f"face API returned {resp.status_code}: {detail}"


def _extract_face_id(response: dict) -> str:
    face_id = response.get("face_id")
    if isinstance(face_id, str):
        return face_id
    raise FaceApiError(f"onboard response had no 'face_id' string field: {response!r}")


def _extract_embedding(response: dict) -> list[float]:
    embeddings = response.get("embeddings")
    if not isinstance(embeddings, dict):
        raise FaceApiError(f"embeddings response missing 'embeddings' object: {response!r}")

    items = embeddings.get(EMBEDDING_MODEL) or next(iter(embeddings.values()), None)
    if not items:
        raise FaceApiError(f"embeddings response had no vectors for any model: {response!r}")

    chosen = next((item for item in items if item.get("pose") == PREFERRED_POSE), items[0])
    vector = chosen.get("embedding")
    if not isinstance(vector, list):
        raise FaceApiError(f"embedding item had no 'embedding' array: {chosen!r}")
    return [float(x) for x in vector]
