"""
schemas.py — Pydantic models for API request/response validation.

These models define the JSON schema for all API responses.  The schemas
are designed to be self-contained: the frontend can render violation cards,
severity badges, and plate info directly from the response without any
additional backend calls.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Response sub-models
# ---------------------------------------------------------------------------


class BBoxModel(BaseModel):
    """A bounding box with class and confidence."""

    class_name: str = Field(..., alias="class", description="Detection class name")
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2] pixel coordinates")
    confidence: float = Field(..., ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class VehicleModel(BaseModel):
    """The vehicle associated with a violation."""

    class_name: str = Field(..., alias="class", description="Vehicle class name")
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class PlateModel(BaseModel):
    """License plate OCR result."""

    text: str = Field(..., description="Corrected plate text")
    raw_text: str = Field("", description="Raw OCR output before correction")
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2]")
    ocr_confidence: float = Field(..., ge=0.0, le=1.0)


class ScoringModel(BaseModel):
    """Breakdown of the composite confidence score."""

    detection_conf: float = Field(..., ge=0.0, le=1.0)
    spatial_conf: float = Field(..., ge=0.0, le=1.0)
    temporal_conf: float = Field(0.0, ge=0.0, le=1.0)
    composite: float = Field(..., ge=0.0, le=1.0)


class ViolationModel(BaseModel):
    """A single detected violation."""

    id: str = Field(..., description="Unique violation ID within this frame")
    type: str = Field(..., description="Violation type identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Composite confidence")
    severity: str = Field(..., description="HIGH / MEDIUM / LOW")
    vehicle: VehicleModel
    evidence: list[BBoxModel] = Field(default_factory=list)
    plate: Optional[PlateModel] = None
    scoring: ScoringModel


class SummaryModel(BaseModel):
    """Aggregated violation summary for a frame."""

    total_violations: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    by_severity: dict[str, int] = Field(default_factory=dict)


class DimensionsModel(BaseModel):
    """Frame dimensions."""

    width: int
    height: int


# ---------------------------------------------------------------------------
# Top-level response models
# ---------------------------------------------------------------------------


class FrameResultModel(BaseModel):
    """Result of processing a single frame (image or video frame)."""

    frame_id: int = 0
    timestamp: Any = Field(
        None,
        description="ISO timestamp (image) or seconds (video frame)",
    )
    dimensions: DimensionsModel
    violations: list[ViolationModel] = Field(default_factory=list)
    summary: SummaryModel = Field(default_factory=SummaryModel)
    error: Optional[str] = None


class ImageAnalysisResponse(BaseModel):
    """Response for POST /api/analyze/image."""

    report: FrameResultModel


class VideoAnalysisResponse(BaseModel):
    """Response for POST /api/analyze/video."""

    total_frames_analyzed: int = 0
    frames: list[FrameResultModel] = Field(default_factory=list)
    summary: SummaryModel = Field(default_factory=SummaryModel)


class HealthResponse(BaseModel):
    """Response for GET /api/health."""

    status: str = "healthy"
    device: str = ""
    models: dict[str, Any] = Field(default_factory=dict)
    analyzers: list[str] = Field(default_factory=list)
    timestamp: str = ""


class DatasetAnalysisResponse(BaseModel):
    """Response for POST /api/analyze/dataset."""

    dataset_path: str
    dataset_type: str
    total_images: int = 0
    total_labels: int = 0
    class_distribution: dict[str, int] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    status: str = "ok"
