"""
tripling.py — Triple-riding violation analyzer.

Rule:
  For each motorcycle / two-wheeler bbox, count the number of rider-class
  detections (rider_with_helmet, rider_without_helmet, with_helmet,
  without_helmet) that overlap with the vehicle by at least
  HELMET_RIDER_IOU (0.20).

  If count >= 3, flag triple riding.

  Spatial confidence = min(1.0, rider_count / 3).
  Detection confidence = average confidence of all overlapping rider detections.
"""

from __future__ import annotations

import config
from engine.spatial_utils import iou
from engine.violation_detector import DetectionResult
from violations.base import BaseViolationAnalyzer, ViolationResult


class TriplingViolationAnalyzer(BaseViolationAnalyzer):
    """Detect triple-riding violations on two-wheelers."""

    # Minimum number of riders to constitute triple riding
    MIN_RIDERS = 3

    def analyze(
        self,
        detections: DetectionResult,
        frame_height: int = 0,
        frame_width: int = 0,
    ) -> list[ViolationResult]:
        violations: list[ViolationResult] = []

        # Only two-wheelers can have triple-riding violations
        two_wheelers = [
            v
            for v in detections.vehicles
            if v.class_id in config.TWO_WHEELER_CLASS_IDS
        ]

        if not two_wheelers:
            return violations

        # All rider/helmet detections that could represent a person on the bike.
        # We include both rider_* and helmet-only classes because in some frames
        # the model detects helmets but not the full rider bbox.
        person_dets = detections.helmets  # IDs 10, 11, 12, 13

        if not person_dets:
            return violations

        for veh in two_wheelers:
            # Find all person detections overlapping this vehicle
            # IoU threshold for person ↔ vehicle: 0.20 (HELMET_RIDER_IOU)
            overlapping = []
            for p in person_dets:
                overlap = iou(p.bbox, veh.bbox)
                if overlap >= config.HELMET_RIDER_IOU:
                    overlapping.append(p)

            if len(overlapping) < self.MIN_RIDERS:
                continue

            # Average detection confidence of all overlapping riders
            avg_conf = sum(p.confidence for p in overlapping) / len(overlapping)

            # Spatial confidence: how clearly we see >= 3 riders
            # At exactly 3, spatial_conf = 1.0; more than 3 also 1.0
            spatial_conf = min(1.0, len(overlapping) / self.MIN_RIDERS)

            violations.append(
                ViolationResult(
                    type="triple_riding",
                    vehicle_class=veh.class_name,
                    vehicle_bbox=veh.bbox,
                    vehicle_confidence=veh.confidence,
                    evidence_bboxes=[
                        {
                            "class": p.class_name,
                            "bbox": p.bbox,
                            "confidence": p.confidence,
                        }
                        for p in overlapping
                    ],
                    detection_conf=avg_conf,
                    spatial_conf=spatial_conf,
                )
            )

        return violations
