"""
vehicle_mods.py — Modified vehicle violation analyzer.

Rule:
  If a `modified` class detection (class ID 20) exists, find the nearest
  vehicle by IoU overlap.  If IoU >= RIDER_VEHICLE_IOU (0.30), flag the
  vehicle as modified.

  This is the simplest violation: the model directly detects modifications.
  Association is needed only to link the modification to a specific vehicle
  for plate extraction.

  Spatial confidence = IoU between the modified detection and the vehicle.
  Detection confidence = modified detection confidence.
"""

from __future__ import annotations

import config
from engine.spatial_utils import centroid_distance, iou
from engine.violation_detector import Detection, DetectionResult
from violations.base import BaseViolationAnalyzer, ViolationResult


class ModifiedVehicleAnalyzer(BaseViolationAnalyzer):
    """Detect modified vehicle violations."""

    def analyze(
        self,
        detections: DetectionResult,
        frame_height: int = 0,
        frame_width: int = 0,
    ) -> list[ViolationResult]:
        violations: list[ViolationResult] = []

        if not detections.modified or not detections.vehicles:
            return violations

        # Track flagged vehicles to avoid duplicates
        flagged: set[tuple[float, ...]] = set()

        for mod_det in detections.modified:
            best_vehicle: Detection | None = None
            best_iou: float = 0.0

            # Primary: IoU matching with threshold 0.30
            for veh in detections.vehicles:
                overlap = iou(mod_det.bbox, veh.bbox)
                if overlap >= config.RIDER_VEHICLE_IOU and overlap > best_iou:
                    best_vehicle = veh
                    best_iou = overlap

            # Fallback: centroid distance (150 px max)
            if best_vehicle is None:
                best_dist = float("inf")
                for veh in detections.vehicles:
                    dist = centroid_distance(mod_det.bbox, veh.bbox)
                    if dist <= config.RIDER_VEHICLE_MAX_DIST and dist < best_dist:
                        best_vehicle = veh
                        best_dist = dist
                        # Distance-based spatial score
                        best_iou = 1.0 - (dist / config.RIDER_VEHICLE_MAX_DIST)

            if best_vehicle is None:
                continue

            veh_key = tuple(best_vehicle.bbox)
            if veh_key in flagged:
                continue
            flagged.add(veh_key)

            violations.append(
                ViolationResult(
                    type="modified_vehicle",
                    vehicle_class=best_vehicle.class_name,
                    vehicle_bbox=best_vehicle.bbox,
                    vehicle_confidence=best_vehicle.confidence,
                    evidence_bboxes=[
                        {
                            "class": mod_det.class_name,
                            "bbox": mod_det.bbox,
                            "confidence": mod_det.confidence,
                        }
                    ],
                    detection_conf=mod_det.confidence,
                    spatial_conf=min(1.0, best_iou),
                )
            )

        return violations
