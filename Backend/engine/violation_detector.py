"""
violation_detector.py — YOLOv8 wrapper for the 21-class violation detection model.

Loads violation_model.pt and provides a clean detection interface that
returns structured Detection dataclass instances grouped by category.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection from the violation model."""

    class_id: int
    class_name: str
    bbox: list[float]       # [x1, y1, x2, y2] absolute pixel coords
    confidence: float

    @property
    def category(self) -> str:
        """Semantic category: 'vehicle', 'helmet', 'signal', or 'infra'."""
        if self.class_id in config.VEHICLE_CLASS_IDS:
            return "vehicle"
        if self.class_id in config.HELMET_CLASS_IDS:
            return "helmet"
        if self.class_id in config.SIGNAL_CLASS_IDS:
            return "signal"
        return "infra"


@dataclass
class DetectionResult:
    """All detections from a single frame, pre-grouped by category."""

    all_detections: list[Detection] = field(default_factory=list)
    vehicles: list[Detection] = field(default_factory=list)
    helmets: list[Detection] = field(default_factory=list)
    riders: list[Detection] = field(default_factory=list)
    no_helmets: list[Detection] = field(default_factory=list)
    signals: list[Detection] = field(default_factory=list)
    stop_lines: list[Detection] = field(default_factory=list)
    modified: list[Detection] = field(default_factory=list)
    red_lights: list[Detection] = field(default_factory=list)
    green_lights: list[Detection] = field(default_factory=list)
    yellow_lights: list[Detection] = field(default_factory=list)


class ViolationDetector:
    """Wrapper around the YOLOv8 violation detection model.

    Usage:
        detector = ViolationDetector()
        result = detector.detect(frame_bgr)
        for v in result.vehicles:
            print(v.class_name, v.bbox, v.confidence)
    """

    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.model_path = model_path or config.VIOLATION_MODEL_PATH
        self.device = device or config.DEVICE
        logger.info("Loading violation model from %s on %s", self.model_path, self.device)
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        logger.info("Violation model loaded successfully")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a single BGR frame.

        Args:
            frame: OpenCV BGR image (H×W×3 numpy array).

        Returns:
            DetectionResult with detections grouped by category.
        """
        results = self.model(
            frame,
            imgsz=config.IMG_SIZE,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_NMS_THRESHOLD,
            max_det=config.MAX_DETECTIONS,
            verbose=False,
        )

        det_result = DetectionResult()

        # results is a list of Results objects (one per image); we sent one frame
        if not results or len(results) == 0:
            return det_result

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return det_result

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            bbox = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
            class_name = config.CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")

            det = Detection(
                class_id=cls_id,
                class_name=class_name,
                bbox=bbox,
                confidence=conf,
            )

            det_result.all_detections.append(det)

            # Group into categories for quick access by violation analyzers
            if cls_id in config.VEHICLE_CLASS_IDS:
                det_result.vehicles.append(det)
            if cls_id in config.RIDER_CLASS_IDS:
                det_result.riders.append(det)
            if cls_id in config.NO_HELMET_CLASS_IDS:
                det_result.no_helmets.append(det)
            if cls_id in config.HELMET_CLASS_IDS:
                det_result.helmets.append(det)
            if cls_id == config.RED_LIGHT_ID:
                det_result.red_lights.append(det)
            if cls_id == config.GREEN_LIGHT_ID:
                det_result.green_lights.append(det)
            if cls_id == config.YELLOW_LIGHT_ID:
                det_result.yellow_lights.append(det)
            if cls_id in config.SIGNAL_CLASS_IDS:
                det_result.signals.append(det)
            if cls_id == config.STOP_LINE_ID:
                det_result.stop_lines.append(det)
            if cls_id == config.MODIFIED_ID:
                det_result.modified.append(det)

        logger.debug(
            "Detected %d objects: %d vehicles, %d helmets, %d signals",
            len(det_result.all_detections),
            len(det_result.vehicles),
            len(det_result.helmets),
            len(det_result.signals),
        )
        return det_result
