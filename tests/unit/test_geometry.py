from visionstack.common.geometry import bbox_foot_point, iou, point_in_polygon
from visionstack.common.types import BBox


def test_iou_identical_boxes_is_one():
    box = BBox(0, 0, 10, 10)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    a = BBox(0, 0, 10, 10)
    b = BBox(20, 20, 30, 30)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 5, 15, 15)
    # intersection = 5x5=25, union = 100+100-25=175
    assert abs(iou(a, b) - 25 / 175) < 1e-6


def test_point_in_polygon_inside():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), square) is True


def test_point_in_polygon_outside():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((50, 50), square) is False


def test_point_in_polygon_degenerate_polygon_is_false():
    assert point_in_polygon((5, 5), [(0, 0), (1, 1)]) is False


def test_bbox_foot_point_is_bottom_center():
    box = BBox(0, 0, 10, 20)
    assert bbox_foot_point(box) == (5.0, 20.0)
