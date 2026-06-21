"""
illegal_parking.py — Illegal parking violation analyzer.

Rule:
  A vehicle is illegally parked if:
    1. Its centroid falls inside a configured no-parking zone polygon.
    2. It has been stationary for at least PARKING_MIN_STATIONARY_FRAMES.

  This violation requires:
    - Pre-configured no-parking zone polygons in config.NO_PARKING_ZONES.
    - Temporal tracking data (video/stream mode only).

  If no zones are configured, this analyzer produces no violations.

  Spatial confidence = 1.0 if centroid is inside the polygon (binary).
  Detection confidence = vehicle detection confidence.
"""

from __future__ import annotations

from typing import Optional

import config
from engine.spatial_utils import centroid, point_in_polygon
from engine.tracker import CentroidTracker
from engine.violation_detector import DetectionResult
from violations.base import BaseViolationAnalyzer, ViolationResult


class IllegalParkingAnalyzer(BaseViolationAnalyzer):
    """Detect illegal parking violations in configured no-parking zones."""

    def __init__(self, tracker: Optional[CentroidTracker] = None):
        self.tracker = tracker

    def analyze(
        self,
        detections: DetectionResult,
        frame_height: int = 0,
        frame_width: int = 0,
    ) -> list[ViolationResult]:
        violations: list[ViolationResult] = []

        # No zones configured → no violations can be detected
        if not config.NO_PARKING_ZONES:
            return violations

        # Illegal parking requires temporal data (tracker)
        if self.tracker is None:
            return violations

        if not detections.vehicles:
            return violations

        for veh in detections.vehicles:
            veh_centroid = centroid(veh.bbox)

            # Check if vehicle centroid is inside any no-parking zone
            in_zone = False
            for zone_polygon in config.NO_PARKING_ZONES:
                if point_in_polygon(veh_centroid, zone_polygon):
                    in_zone = True
                    break

            if not in_zone:
                continue

            # Check stationarity via tracker
            track_id = self.tracker.get_track_for_bbox(veh.bbox)
            if track_id is None:
                continue

            if not self.tracker.is_stationary(track_id):
                continue

            # Vehicle is stationary in a no-parking zone → flag it
            # Spatial confidence is 1.0 (centroid clearly inside polygon)
            violations.append(
                ViolationResult(
                    type="illegal_parking",
                    vehicle_class=veh.class_name,
                    vehicle_bbox=veh.bbox,
                    vehicle_confidence=veh.confidence,
                    evidence_bboxes=[],  # No additional evidence boxes
                    detection_conf=veh.confidence,
                    spatial_conf=1.0,
                )
            )

        return violations
