"""Geometry helpers shared by detection, tracking, and zone monitoring."""
from __future__ import annotations

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

from visionstack.common.types import BBox

Point = tuple[float, float]


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes, in [0, 1]."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0
    union_area = a.width * a.height + b.width * b.height - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """True if `point` lies inside `polygon` (list of (x, y) vertices)."""
    if len(polygon) < 3:
        return False
    return ShapelyPolygon(polygon).contains(ShapelyPoint(point))


def bbox_foot_point(bbox: BBox) -> Point:
    """Bottom-center of a bounding box — the usual proxy for a person's floor position."""
    cx, _ = bbox.center
    return cx, bbox.y2
