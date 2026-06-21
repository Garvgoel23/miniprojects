"""
helmet.py — No-helmet violation analyzer.

Rule:
  For each `without_helmet` or `rider_without_helmet` detection, find the
  nearest two-wheeler or three-wheeler by IoU overlap.  If IoU >= threshold
  (RIDER_VEHICLE_IOU = 0.30), flag the vehicle for a no-helmet violation.

  Spatial confidence = the IoU value between the no-helmet detection and
  the matched vehicle (clamped to 1.0).

  If IoU matching fails, fall back to centroid distance within
  RIDER_VEHICLE_MAX_DIST (150 px).  Spatial confidence for distance-based
  matches is scaled as: 1.0 - (distance / max_distance).
"""

from __future__ import annotations

import config
from engine.spatial_utils import centroid_distance, iou
from engine.violation_detector import Detection, DetectionResult
from violations.base import BaseViolationAnalyzer, ViolationResult


class HelmetViolationAnalyzer(BaseViolationAnalyzer):
    """Detect no-helmet violations on two/three-wheelers."""

    def analyze(
        self,
        detections: DetectionResult,
        frame_height: int = 0,
        frame_width: int = 0,
    ) -> list[ViolationResult]:
        violations: list[ViolationResult] = []

        # Candidate vehicles: two-wheelers and three-wheelers
        target_vehicles = [
            v
            for v in detections.vehicles
            if v.class_id in config.TWO_WHEELER_CLASS_IDS
            or v.class_id in config.THREE_WHEELER_CLASS_IDS
        ]

        if not target_vehicles or not detections.no_helmets:
            return violations

        # Track which vehicles have already been flagged to avoid duplicates
        flagged_vehicle_bboxes: set[tuple[float, ...]] = set()

        for nh_det in detections.no_helmets:
            best_vehicle: Detection | None = None
            best_score: float = 0.0
            match_type: str = "none"

            # --- Primary: IoU matching ---
            # IoU threshold for rider/helmet ↔ vehicle association: 0.30
            for veh in target_vehicles:
                overlap = iou(nh_det.bbox, veh.bbox)
                if overlap >= config.RIDER_VEHICLE_IOU and overlap > best_score:
                    best_vehicle = veh
                    best_score = overlap
                    match_type = "iou"

            # --- Fallback: centroid distance ---
            # Max distance: 150 px (RIDER_VEHICLE_MAX_DIST)
            if best_vehicle is None:
                for veh in target_vehicles:
                    dist = centroid_distance(nh_det.bbox, veh.bbox)
                    if dist <= config.RIDER_VEHICLE_MAX_DIST:
                        # Convert distance to a score: closer = higher
                        score = 1.0 - (dist / config.RIDER_VEHICLE_MAX_DIST)
                        if score > best_score:
                            best_vehicle = veh
                            best_score = score
                            match_type = "centroid"

            if best_vehicle is None:
                continue

            veh_key = tuple(best_vehicle.bbox)
            if veh_key in flagged_vehicle_bboxes:
                continue
            flagged_vehicle_bboxes.add(veh_key)

            # Spatial confidence = match quality (IoU or distance-based score)
            spatial_conf = min(1.0, best_score)

            violations.append(
                ViolationResult(
                    type="no_helmet",
                    vehicle_class=best_vehicle.class_name,
                    vehicle_bbox=best_vehicle.bbox,
                    vehicle_confidence=best_vehicle.confidence,
                    evidence_bboxes=[
                        {
                            "class": nh_det.class_name,
                            "bbox": nh_det.bbox,
                            "confidence": nh_det.confidence,
                        }
                    ],
                    detection_conf=nh_det.confidence,
                    spatial_conf=spatial_conf,
                )
            )

        return violations
