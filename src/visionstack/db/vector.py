"""pgvector column helper, shared by employee_face_embeddings / employee_body_embeddings."""
from __future__ import annotations

from pgvector.sqlalchemy import Vector

FACE_EMBEDDING_DIM = 512
BODY_EMBEDDING_DIM = 512


def face_embedding_column() -> Vector:
    return Vector(FACE_EMBEDDING_DIM)


def body_embedding_column() -> Vector:
    return Vector(BODY_EMBEDDING_DIM)
