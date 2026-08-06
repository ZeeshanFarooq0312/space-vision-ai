import numpy as np

from visionstack.identity.body_embedder import EMBEDDING_DIM as BODY_DIM
from visionstack.identity.body_embedder import NoOpBodyEmbedder
from visionstack.identity.face_detector import NoOpFaceDetector
from visionstack.identity.face_embedder import EMBEDDING_DIM as FACE_DIM
from visionstack.identity.face_embedder import NoOpFaceEmbedder
from visionstack.identity.gallery import NoOpEmployeeGallery
from visionstack.identity.matcher import NoOpIdentityMatcher


def test_noop_face_detector_finds_no_faces():
    assert NoOpFaceDetector().detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_noop_face_embedder_returns_zero_vector_of_correct_dim():
    embedding = NoOpFaceEmbedder().embed(np.zeros((10, 10, 3), dtype=np.uint8))
    assert embedding.shape == (FACE_DIM,)
    assert np.all(embedding == 0)


def test_noop_body_embedder_returns_zero_vector_of_correct_dim():
    embedding = NoOpBodyEmbedder().embed(np.zeros((10, 10, 3), dtype=np.uint8))
    assert embedding.shape == (BODY_DIM,)


def test_noop_gallery_query_returns_no_matches():
    gallery = NoOpEmployeeGallery()
    gallery.enroll("emp-1", np.zeros(512), modality="face")
    assert gallery.query(np.zeros(512), modality="face") == []


def test_noop_identity_matcher_always_unknown():
    result = NoOpIdentityMatcher().match(face_embedding=np.zeros(512), body_embedding=None)
    assert result.is_unknown is True
    assert result.employee_id is None
    assert result.modality == "none"
