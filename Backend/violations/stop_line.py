"""
stop_line.py — Stop-line violation analyzer.

Rule:
  If any traffic light is detected that is NOT green (i.e., red_light,
  yellow_light, or generic traffic_light) AND a vehicle bbox extends past
  the stop_line, flag a stop-line violation.

  Difference from red_light.py:
    - red_light.py fires only on confirmed red lights.
    - stop_line.py fires on red, yellow, AND ambiguous traffic-light
      detections (any non-green signal state).

  Spatial confidence = normalized overshoot (same formula as red_light.py).
  Detection confidence = minimum of (signal confidence, vehicle confidence).
"""

from __future__ import annotations

import config
from engine.spatial_utils import is_past_stop_line
from engine.violation_detector import Detection, DetectionResult
from violations.base import BaseViolationAnalyzer, ViolationResult


class StopLineViolationAnalyzer(BaseViolationAnalyzer):
    """Detect stop-line violations on non-green signals."""

    def analyze(
        self,
        detections: DetectionResult,
        frame_height: int = 0,
        frame_width: int = 0,
    ) -> list[ViolationResult]:
        violations: list[ViolationResult] = []

        if not detections.stop_lines or not detections.vehicles:
            return violations

        # Collect non-green signal detections:
        #   red_light (14), yellow_light (16), traffic_light (17)
        # Exclude green_light (15) — green means legal to cross.
        non_green_signals: list[Detection] = []
        for sig in detections.signals:
            if sig.class_id != config.GREEN_LIGHT_ID:
                non_green_signals.append(sig)

        if not non_green_signals:
            return violations

        # Use the most confident non-green signal and stop line
        signal = max(non_green_signals, key=lambda d: d.confidence)
        stop_line = max(detections.stop_lines, key=lambda d: d.confidence)

        # Skip if this is a pure red-light scenario — let red_light.py handle it.
        # Stop-line analyzer fires only if the signal is yellow or ambiguous,
        # OR if there's no red light (in which case red_light.py won't fire).
        # To avoid double-counting, we check: if we have a red light AND this
        # is the same signal, we still flag (the pipeline deduplicates later).
        # Decision: flag here for ALL non-green states; pipeline handles dedup.

        for veh in detections.vehicles:
            # Check if vehicle is past the stop line
            # Margin: STOP_LINE_MARGIN_PX = 50 px
            past, overshoot = is_past_stop_line(
                veh.bbox, stop_line.bbox, margin=config.STOP_LINE_MARGIN_PX
            )

            if not past:
                continue

            if frame_height > 0:
                spatial_conf = min(1.0, overshoot / (frame_height * 0.10))
            else:
                spatial_conf = min(1.0, overshoot / 100.0)

            detection_conf = min(signal.confidence, veh.confidence)

            violations.append(
                ViolationResult(
                    type="stop_line_violation",
                    vehicle_class=veh.class_name,
                    vehicle_bbox=veh.bbox,
                    vehicle_confidence=veh.confidence,
                    evidence_bboxes=[
                        {
                            "class": signal.class_name,
                            "bbox": signal.bbox,
                            "confidence": signal.confidence,
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
