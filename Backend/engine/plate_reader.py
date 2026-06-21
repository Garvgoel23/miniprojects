"""
plate_reader.py — License plate detection (YOLOv8) + OCR (EasyOCR).

Pipeline per plate crop:
  1. YOLO plate_model.pt → bounding box
  2. Crop plate region from frame
  3. Preprocess: grayscale → CLAHE → bilateral filter
  4. EasyOCR → raw text
  5. Regex correction for Indian plate format with OCR confusion fixes
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


@dataclass
class PlateResult:
    """A detected license plate with its OCR-extracted text."""

    bbox: list[float]           # [x1, y1, x2, y2] in original frame coords
    detection_confidence: float  # YOLO confidence for the plate bbox
    raw_text: str               # Raw OCR output before correction
    corrected_text: str         # After regex/confusion correction
    ocr_confidence: float       # EasyOCR confidence score (0.0–1.0)


class PlateReader:
    """Detects license plates via YOLO and reads them via EasyOCR.

    Usage:
        reader = PlateReader()
        plates = reader.detect_and_read(frame_bgr)
        for p in plates:
            print(p.corrected_text, p.ocr_confidence)
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
    ):
        self.model_path = model_path or config.PLATE_MODEL_PATH
        self.device = device or config.DEVICE

        logger.info("Loading plate model from %s on %s", self.model_path, self.device)
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # EasyOCR reader — initialized once (heavy on first load).
        # gpu=True if CUDA is available.
        logger.info("Initializing EasyOCR reader")
        self.ocr_reader = easyocr.Reader(
            ["en"],
            gpu=(self.device == "cuda"),
            verbose=False,
        )
        logger.info("Plate reader ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_read(self, frame: np.ndarray) -> list[PlateResult]:
        """Detect plates in *frame* and OCR each crop.

        Args:
            frame: BGR image (H×W×3).

        Returns:
            List of PlateResult, one per detected plate.
        """
        results = self.model(
            frame,
            imgsz=config.IMG_SIZE,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_NMS_THRESHOLD,
            verbose=False,
        )

        plates: list[PlateResult] = []

        if not results or len(results) == 0:
            return plates

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return plates

        h, w = frame.shape[:2]

        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].tolist()
            det_conf = float(boxes.conf[i].item())

            # Clamp bbox to frame bounds
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))

            if x2 - x1 < 10 or y2 - y1 < 5:
                # Plate crop too small to OCR meaningfully
                continue

            crop = frame[y1:y2, x1:x2]
            preprocessed = self._preprocess(crop)
            raw_text, ocr_conf = self._ocr(preprocessed)
            corrected = self._correct_plate_text(raw_text)

            plates.append(
                PlateResult(
                    bbox=bbox,
                    detection_confidence=det_conf,
                    raw_text=raw_text,
                    corrected_text=corrected,
                    ocr_confidence=ocr_conf,
                )
            )

        logger.debug("Read %d plates", len(plates))
        return plates

    def detect_only(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        """Detect plate bounding boxes without running OCR.

        Returns:
            List of (bbox, confidence) tuples.
        """
        results = self.model(
            frame,
            imgsz=config.IMG_SIZE,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_NMS_THRESHOLD,
            verbose=False,
        )
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].tolist()
                    conf = float(boxes.conf[i].item())
                    detections.append((bbox, conf))
        return detections

    def read_crop(self, frame: np.ndarray, bbox: list[float]) -> Optional[PlateResult]:
        """OCR a specific plate region (used when plate bbox is already known).

        Args:
            frame: Full BGR frame.
            bbox: [x1, y1, x2, y2] of the plate in frame coords.

        Returns:
            PlateResult or None if crop is too small.
        """
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 - x1 < 10 or y2 - y1 < 5:
            return None

        crop = frame[y1:y2, x1:x2]
        preprocessed = self._preprocess(crop)
        raw_text, ocr_conf = self._ocr(preprocessed)
        corrected = self._correct_plate_text(raw_text)

        return PlateResult(
            bbox=bbox,
            detection_confidence=1.0,  # bbox was given, not detected
            raw_text=raw_text,
            corrected_text=corrected,
            ocr_confidence=ocr_conf,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        """Enhance a plate crop for better OCR accuracy.

        Pipeline:
          1. Convert to grayscale
          2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
             - clipLimit=2.0, tileGridSize=8×8
             - Improves contrast in unevenly lit plates
          3. Bilateral filter (d=11, σColor=17, σSpace=17)
             - Smooths noise while preserving character edges
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=config.OCR_CLAHE_CLIP_LIMIT,
            tileGridSize=config.OCR_CLAHE_TILE_GRID,
        )
        enhanced = clahe.apply(gray)

        filtered = cv2.bilateralFilter(
            enhanced,
            d=config.OCR_BILATERAL_D,
            sigmaColor=config.OCR_BILATERAL_SIGMA_COLOR,
            sigmaSpace=config.OCR_BILATERAL_SIGMA_SPACE,
        )
        return filtered

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _ocr(self, preprocessed: np.ndarray) -> tuple[str, float]:
        """Run EasyOCR on a preprocessed grayscale plate image.

        Returns:
            (text, avg_confidence) — text is uppercased, whitespace stripped.
        """
        try:
            ocr_results = self.ocr_reader.readtext(preprocessed)
        except Exception as e:
            logger.warning("EasyOCR failed: %s", e)
            return ("", 0.0)

        if not ocr_results:
            return ("", 0.0)

        # Concatenate all detected text blocks and average their confidence
        texts = []
        confs = []
        for (_bbox, text, conf) in ocr_results:
            texts.append(text)
            confs.append(conf)

        combined = "".join(texts).upper().strip()
        # Remove spaces and special characters — plates are alphanumeric
        combined = re.sub(r"[^A-Z0-9]", "", combined)
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        return (combined, avg_conf)

    # ------------------------------------------------------------------
    # Regex correction for Indian plates
    # ------------------------------------------------------------------

    @staticmethod
    def _correct_plate_text(raw: str) -> str:
        """Apply context-sensitive OCR confusion corrections.

        Indian plate format: <2 letters><2 digits><1-3 letters><4 digits>
        Example: KA01AB1234

        Strategy:
          - Positions 0-1: must be letters → apply letter corrections
          - Positions 2-3: must be digits → apply digit corrections
          - Positions 4 to -4: must be letters → apply letter corrections
          - Last 4 positions: must be digits → apply digit corrections
        """
        if len(raw) < 6:
            # Too short to be a valid plate; return as-is
            return raw

        corrected = list(raw)

        # Positions 0–1: state code (letters)
        for i in range(min(2, len(corrected))):
            ch = corrected[i]
            if ch in config.OCR_LETTER_CORRECTIONS:
                corrected[i] = config.OCR_LETTER_CORRECTIONS[ch]

        # Positions 2–3: RTO code (digits)
        for i in range(2, min(4, len(corrected))):
            ch = corrected[i]
            if ch in config.OCR_DIGIT_CORRECTIONS:
                corrected[i] = config.OCR_DIGIT_CORRECTIONS[ch]

        # Last 4 positions: vehicle number (digits)
        for i in range(max(4, len(corrected) - 4), len(corrected)):
            ch = corrected[i]
            if ch in config.OCR_DIGIT_CORRECTIONS:
                corrected[i] = config.OCR_DIGIT_CORRECTIONS[ch]

        # Middle positions (between pos 4 and len-4): series letters
        for i in range(4, max(4, len(corrected) - 4)):
            ch = corrected[i]
            if ch in config.OCR_LETTER_CORRECTIONS:
                corrected[i] = config.OCR_LETTER_CORRECTIONS[ch]

        return "".join(corrected)
