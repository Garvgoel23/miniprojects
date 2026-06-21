"""
spatial_utils.py — Geometry utilities for bounding-box association.

All functions operate on boxes in [x1, y1, x2, y2] (top-left, bottom-right) format.
"""

from __future__ import annotations

import math
from typing import Sequence


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection-over-Union between two axis-aligned boxes.

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        IoU value in [0.0, 1.0].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def centroid(box: Sequence[float]) -> tuple[float, float]:
    """Return the center point (cx, cy) of a box [x1, y1, x2, y2]."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def centroid_distance(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Euclidean distance between the centroids of two boxes."""
    ca = centroid(box_a)
    cb = centroid(box_b)
    return math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)


def containment_ratio(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of *inner* box's area that lies inside *outer*.

    Useful for checking if a small detection (plate, helmet) is "inside"
    a larger detection (vehicle, rider).

    Returns:
        Value in [0.0, 1.0].
    """
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    inner_area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    if inner_area <= 0:
        return 0.0
    return inter_area / inner_area


def point_in_polygon(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    """Ray-casting algorithm to test if *point* is inside *polygon*.

    Args:
        point: (x, y) coordinate.
        polygon: List of (x, y) vertices defining a closed polygon.

    Returns:
        True if the point is inside the polygon.
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Check if the ray from (x, y) going right crosses edge (i, j)
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def is_past_stop_line(
    vehicle_box: Sequence[float],
    stop_line_box: Sequence[float],
    margin: float = 50.0,
) -> tuple[bool, float]:
    """Check whether a vehicle has crossed past a stop line.

    Logic: the vehicle's bottom edge (y2) must be below (greater than,
    in image coords) the stop line's vertical center + margin.

    Args:
        vehicle_box: [x1, y1, x2, y2] of the vehicle.
        stop_line_box: [x1, y1, x2, y2] of the stop-line detection.
        margin: Extra pixels the vehicle must extend past the line center.

    Returns:
        (is_past: bool, overshoot_px: float) — overshoot is how far past
        the threshold the vehicle's bottom edge extends (0 if not past).
    """
    # Stop line's vertical center
    line_y_center = (stop_line_box[1] + stop_line_box[3]) / 2.0
    threshold = line_y_center + margin

    vehicle_bottom = vehicle_box[3]
    overshoot = vehicle_bottom - threshold
    return (overshoot > 0, max(0.0, overshoot))


def box_area(box: Sequence[float]) -> float:
    """Area of a bounding box [x1, y1, x2, y2]."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
