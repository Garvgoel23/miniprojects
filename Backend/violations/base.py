"""
base.py — Abstract base class and shared data structures for violation analyzers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import config
from engine.violation_detector import DetectionResult


@dataclass
class ViolationResult:
    """A single detected violation with full context for JSON output."""

    type: str                              # e.g. "no_helmet", "triple_riding"
    vehicle_class: str                     # e.g. "motorcycle", "car"
    vehicle_bbox: list[float]              # [x1, y1, x2, y2]
    vehicle_confidence: float              # Detection confidence of the vehicle
    evidence_bboxes: list[dict] = field(default_factory=list)
    # Each evidence dict: {"class": str, "bbox": [x1,y1,x2,y2], "confidence": float}
    detection_conf: float = 0.0            # Raw model confidence (best among evidence)
    spatial_conf: float = 0.0              # Spatial association quality [0.0–1.0]
    composite_conf: float = 0.0            # Weighted blend (computed by pipeline)
    severity: str = "LOW"                  # HIGH / MEDIUM / LOW


def compute_severity(composite: float) -> str:
    """Map a composite confidence score to a severity label.

    Thresholds (from config.py):
      - HIGH   : composite >= 0.80
      - MEDIUM : composite >= 0.50
      - LOW    : composite <  0.50
    """
    if composite >= config.SEVERITY_HIGH:
        return "HIGH"
    if composite >= config.SEVERITY_MEDIUM:
        return "MEDIUM"
    return "LOW"


def compute_composite(
    detection_conf: float,
    spatial_conf: float,
    temporal_conf: float = 0.0,
    is_video: bool = False,
) -> float:
    """Compute the weighted composite confidence score.

    For video/stream mode:
        composite = W_DETECTION × detection + W_SPATIAL × spatial + W_TEMPORAL × temporal

    For single-image mode (no temporal data):
        composite = W_DETECTION_IMG × detection + W_SPATIAL_IMG × spatial
    """
    if is_video:
        return (
            config.W_DETECTION * detection_conf
            + config.W_SPATIAL * spatial_conf
            + config.W_TEMPORAL * temporal_conf
        )
    else:
        return (
            config.W_DETECTION_IMG * detection_conf
            + config.W_SPATIAL_IMG * spatial_conf
        )


class BaseViolationAnalyzer(ABC):
    """Abstract base for all violation analyzers.

    Each subclass implements `analyze()` to inspect grouped detections
    and return a list of ViolationResult for its specific violation type.
    """

    @abstractmethod
    def analyze(
        self,
        detections: DetectionResult,
        frame_height: int = 0,
        frame_width: int = 0,
    ) -> list[ViolationResult]:
        """Analyze detections for a specific violation type.

        Args:
            detections: Grouped detection result from ViolationDetector.
            frame_height: Frame height in pixels (for normalization).
            frame_width: Frame width in pixels.

        Returns:
            List of ViolationResult, possibly empty.
        """
        ...
