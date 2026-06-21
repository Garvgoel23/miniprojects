"""
pipeline.py — Dual-model parallel pipeline orchestrator.

Runs both YOLO models (violation + plate) in parallel via ThreadPoolExecutor,
then merges results through spatial association, violation analysis, plate
matching, and composite confidence scoring.

This is the central module that ties engine/ and violations/ together.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

import cv2
import numpy as np

import config
from engine.plate_reader import PlateReader, PlateResult
from engine.spatial_utils import iou
from engine.tracker import CentroidTracker
from engine.violation_detector import DetectionResult, ViolationDetector
from violations.base import (
    ViolationResult,
    compute_composite,
    compute_severity,
)
from violations.helmet import HelmetViolationAnalyzer
from violations.illegal_parking import IllegalParkingAnalyzer
from violations.red_light import RedLightViolationAnalyzer
from violations.stop_line import StopLineViolationAnalyzer
from violations.tripling import TriplingViolationAnalyzer
from violations.vehicle_mods import ModifiedVehicleAnalyzer

logger = logging.getLogger(__name__)


class Pipeline:
    """Dual-model, parallel-inference pipeline for traffic violation detection.

    Usage:
        pipe = Pipeline()
        result = pipe.process_image(image_bytes)
        # or
        for frame_result in pipe.process_video(video_bytes, stride=3):
            ...
    """

    def __init__(self):
        logger.info("Initializing pipeline...")

        # Load models
        self.violation_detector = ViolationDetector()
        self.plate_reader = PlateReader()

        # Tracker for video/stream temporal consistency
        self.tracker = CentroidTracker()

        # Initialize all 6 violation analyzers
        self.analyzers = [
            HelmetViolationAnalyzer(),
            TriplingViolationAnalyzer(),
            RedLightViolationAnalyzer(),
            IllegalParkingAnalyzer(tracker=self.tracker),
            StopLineViolationAnalyzer(),
            ModifiedVehicleAnalyzer(),
        ]

        # Thread pool for parallel model inference (2 models)
        self._executor = ThreadPoolExecutor(max_workers=2)

        logger.info("Pipeline ready — %d violation analyzers loaded", len(self.analyzers))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(self, image_bytes: bytes) -> dict[str, Any]:
        """Process a single image.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)

        Returns:
            Structured result dict (self-contained for frontend rendering).
        """
        frame = self._decode_image(image_bytes)
        if frame is None:
            return self._error_result("Failed to decode image")

        result = self._process_frame(frame, frame_id=0, is_video=False)
        return result

    def process_video(
        self,
        video_bytes: bytes,
        stride: int | None = None,
    ) -> list[dict[str, Any]]:
        """Process a video file, analyzing every Nth frame.

        Args:
            video_bytes: Raw video file bytes.
            stride: Process every Nth frame (default: config.VIDEO_STRIDE).

        Returns:
            List of per-frame result dicts.
        """
        stride = stride or config.VIDEO_STRIDE
        self.tracker = CentroidTracker()  # Reset tracker for new video

        # Write video bytes to a temp file for OpenCV
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()

        results = []
        try:
            cap = cv2.VideoCapture(tmp.name)
            if not cap.isOpened():
                return [self._error_result("Failed to open video")]

            frame_idx = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % stride == 0:
                    timestamp = frame_idx / fps
                    result = self._process_frame(
                        frame,
                        frame_id=frame_idx,
                        is_video=True,
                        timestamp=timestamp,
                    )
                    results.append(result)

                frame_idx += 1

            cap.release()
        finally:
            Path(tmp.name).unlink(missing_ok=True)

        return results

    def process_video_path(
        self,
        video_path: str,
        stride: int | None = None,
    ) -> list[dict[str, Any]]:
        """Process a video file from a file path (CLI mode).

        Args:
            video_path: Path to the video file.
            stride: Process every Nth frame.

        Returns:
            List of per-frame result dicts.
        """
        stride = stride or config.VIDEO_STRIDE
        self.tracker = CentroidTracker()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [self._error_result(f"Failed to open video: {video_path}")]

        results = []
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                timestamp = frame_idx / fps
                result = self._process_frame(
                    frame,
                    frame_id=frame_idx,
                    is_video=True,
                    timestamp=timestamp,
                )
                results.append(result)

            frame_idx += 1

        cap.release()
        return results

    def process_stream_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> dict[str, Any]:
        """Process a single frame from a live stream.

        The tracker persists across calls, providing temporal consistency.

        Args:
            frame: BGR image (H×W×3).
            frame_id: Sequential frame counter.
            timestamp: Seconds since stream start.

        Returns:
            Structured result dict.
        """
        return self._process_frame(
            frame, frame_id=frame_id, is_video=True, timestamp=timestamp
        )

    def get_health(self) -> dict[str, Any]:
        """Return health/status info for the /api/health endpoint."""
        return {
            "status": "healthy",
            "device": config.DEVICE,
            "models": {
                "violation_model": {
                    "path": config.VIOLATION_MODEL_PATH,
                    "loaded": self.violation_detector is not None,
                },
                "plate_model": {
                    "path": config.PLATE_MODEL_PATH,
                    "loaded": self.plate_reader is not None,
                },
            },
            "analyzers": [type(a).__name__ for a in self.analyzers],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Core frame processing
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        is_video: bool = False,
        timestamp: float = 0.0,
    ) -> dict[str, Any]:
        """Core processing for a single frame.

        Steps:
          1. Run both models in parallel (ThreadPoolExecutor)
          2. Feed detections into all violation analyzers
          3. Match plates to flagged vehicles (IoU >= PLATE_VEHICLE_IOU)
          4. Compute composite confidence + severity per violation
          5. Return structured JSON-ready dict
        """
        h, w = frame.shape[:2]

        # --- Step 1: Parallel model inference ---
        future_violations = self._executor.submit(self.violation_detector.detect, frame)
        future_plates = self._executor.submit(self.plate_reader.detect_and_read, frame)

        detection_result: DetectionResult = future_violations.result()
        plate_results: list[PlateResult] = future_plates.result()

        # --- Step 1.5: Update tracker (video/stream mode) ---
        if is_video:
            vehicle_bboxes = [v.bbox for v in detection_result.vehicles]
            self.tracker.update(vehicle_bboxes)

        # --- Step 2: Run all violation analyzers ---
        all_violations: list[ViolationResult] = []
        for analyzer in self.analyzers:
            try:
                found = analyzer.analyze(
                    detection_result,
                    frame_height=h,
                    frame_width=w,
                )
                all_violations.extend(found)
            except Exception as e:
                logger.error(
                    "Analyzer %s failed: %s", type(analyzer).__name__, e, exc_info=True
                )

        # --- Step 3: Match plates to flagged vehicles ---
        # For each violation, find the plate with the highest IoU overlap
        # with the vehicle bbox.  Threshold: PLATE_VEHICLE_IOU = 0.15
        violation_plates: dict[int, PlateResult] = {}  # violation_index → plate
        for vi, viol in enumerate(all_violations):
            best_plate: PlateResult | None = None
            best_overlap = 0.0
            for plate in plate_results:
                overlap = iou(plate.bbox, viol.vehicle_bbox)
                if overlap >= config.PLATE_VEHICLE_IOU and overlap > best_overlap:
                    best_plate = plate
                    best_overlap = overlap
            if best_plate is not None:
                violation_plates[vi] = best_plate

        # --- Step 4: Compute composite confidence + severity ---
        for vi, viol in enumerate(all_violations):
            temporal_conf = 0.0
            if is_video:
                track_id = self.tracker.get_track_for_bbox(viol.vehicle_bbox)
                if track_id is not None:
                    # Record this violation for temporal tracking
                    self.tracker.record_violation(track_id, viol.type, True)
                    temporal_conf = self.tracker.temporal_consistency(
                        track_id, viol.type
                    )

            viol.composite_conf = compute_composite(
                viol.detection_conf,
                viol.spatial_conf,
                temporal_conf,
                is_video=is_video,
            )
            viol.severity = compute_severity(viol.composite_conf)

        # --- Step 5: Build structured output ---
        violations_json = []
        for vi, viol in enumerate(all_violations):
            entry: dict[str, Any] = {
                "id": f"v_{vi:03d}",
                "type": viol.type,
                "confidence": round(viol.composite_conf, 4),
                "severity": viol.severity,
                "vehicle": {
                    "class": viol.vehicle_class,
                    "bbox": [round(c, 1) for c in viol.vehicle_bbox],
                    "confidence": round(viol.vehicle_confidence, 4),
                },
                "evidence": [
                    {
                        "class": e["class"],
                        "bbox": [round(c, 1) for c in e["bbox"]],
                        "confidence": round(e["confidence"], 4),
                    }
                    for e in viol.evidence_bboxes
                ],
                "scoring": {
                    "detection_conf": round(viol.detection_conf, 4),
                    "spatial_conf": round(viol.spatial_conf, 4),
                    "temporal_conf": round(
                        (
                            self.tracker.temporal_consistency(
                                self.tracker.get_track_for_bbox(viol.vehicle_bbox) or -1,
                                viol.type,
                            )
                            if is_video
                            else 0.0
                        ),
                        4,
                    ),
                    "composite": round(viol.composite_conf, 4),
                },
            }

            # Attach plate if matched
            plate = violation_plates.get(vi)
            if plate is not None:
                entry["plate"] = {
                    "text": plate.corrected_text,
                    "raw_text": plate.raw_text,
                    "bbox": [round(c, 1) for c in plate.bbox],
                    "ocr_confidence": round(plate.ocr_confidence, 4),
                }
            else:
                entry["plate"] = None

            violations_json.append(entry)

        # Summary
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for v in violations_json:
            by_type[v["type"]] = by_type.get(v["type"], 0) + 1
            by_severity[v["severity"]] = by_severity.get(v["severity"], 0) + 1

        return {
            "frame_id": frame_id,
            "timestamp": (
                timestamp if is_video
                else datetime.now(timezone.utc).isoformat()
            ),
            "dimensions": {"width": w, "height": h},
            "violations": violations_json,
            "summary": {
                "total_violations": len(violations_json),
                "by_type": by_type,
                "by_severity": by_severity,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray | None:
        """Decode raw bytes into a BGR numpy array."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame

    @staticmethod
    def _error_result(message: str) -> dict[str, Any]:
        """Return a minimal error result dict."""
        return {
            "frame_id": -1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dimensions": {"width": 0, "height": 0},
            "violations": [],
            "summary": {"total_violations": 0, "by_type": {}, "by_severity": {}},
            "error": message,
        }
