"""
red_light.py — Red-light jumping violation analyzer.

Rule:
  If a `red_light` detection exists AND any vehicle bbox extends past the
  `stop_line` bbox (vehicle bottom > stop-line center + STOP_LINE_MARGIN_PX),
  flag the vehicle for red-light jumping.

  Spatial confidence = normalized overshoot distance:
    min(1.0, overshoot_px / frame_height × 10)
  This scales so that ~10% of frame height past the line gives full confidence.

  Detection confidence = minimum of (red_light confidence, vehicle confidence).
"""

from __future__ import annotations

import config
from engine.spatial_utils import is_past_stop_line
from engine.violation_detector import DetectionResult
from violations.base import BaseViolationAnalyzer, ViolationResult


class RedLightViolationAnalyzer(BaseViolationAnalyzer):
    """Detect red-light jumping violations."""

    def analyze(
        self,
        detections: DetectionResult,
        frame_height: int = 0,
        frame_width: int = 0,
    ) -> list[ViolationResult]:
        violations: list[ViolationResult] = []

        # Must have at least one red light and one stop line detected
        if not detections.red_lights or not detections.stop_lines:
            return violations

        if not detections.vehicles:
            return violations

        # Use the most confident red light and stop line
        red_light = max(detections.red_lights, key=lambda d: d.confidence)
        stop_line = max(detections.stop_lines, key=lambda d: d.confidence)

        for veh in detections.vehicles:
            # Check if vehicle extends past the stop line
            # Margin: STOP_LINE_MARGIN_PX = 50 px
            past, overshoot = is_past_stop_line(
                veh.bbox, stop_line.bbox, margin=config.STOP_LINE_MARGIN_PX
            )

            if not past:
                continue

            # Spatial confidence based on how far past the line the vehicle is.
            # Normalize by frame height: 10% of frame = full confidence.
            if frame_height > 0:
                spatial_conf = min(1.0, overshoot / (frame_height * 0.10))
            else:
                spatial_conf = min(1.0, overshoot / 100.0)

            detection_conf = min(red_light.confidence, veh.confidence)

            violations.append(
                ViolationResult(
                    type="red_light_jumping",
                    vehicle_class=veh.class_name,
                    vehicle_bbox=veh.bbox,
                    vehicle_confidence=veh.confidence,
                    evidence_bboxes=[
                        {
                            "class": red_light.class_name,
                            "bbox": red_light.bbox,
                            "confidence": red_light.confidence,
                        },
                        {
                            "class": stop_line.class_name,
                            "bbox": stop_line.bbox,
                            "confidence": stop_line.confidence,
                        },
                    ],
                    detection_conf=detection_conf,
                    spatial_conf=spatial_conf,
                )
            )

        return violations
