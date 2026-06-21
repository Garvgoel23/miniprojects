"""
config.py — Central configuration for the Traffic Violation Detection backend.

All spatial thresholds, confidence weights, and model paths are defined here
with explicit comments so they are auditable for the hackathon metrics report.
"""

import os
from pathlib import Path

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths  (relative to this file's directory, i.e. /Backend)
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = _BASE_DIR / ".." / "Models"

# Model file paths — both models sit in /Models at the repo root
VIOLATION_MODEL_PATH = str(MODELS_DIR / "violation_model.pt")
PLATE_MODEL_PATH = str(MODELS_DIR / "plate_model.pt")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
# Auto-select CUDA if available; fall back to CPU
DEVICE = "cuda" if _CUDA_AVAILABLE else "cpu"

# ---------------------------------------------------------------------------
# YOLO inference settings
# ---------------------------------------------------------------------------
IMG_SIZE = 640              # Input image size for YOLO inference
CONF_THRESHOLD = 0.25       # Minimum confidence to keep a detection
IOU_NMS_THRESHOLD = 0.45    # IoU threshold for Non-Maximum Suppression
MAX_DETECTIONS = 300         # Maximum detections per frame

# ---------------------------------------------------------------------------
# Video / Stream
# ---------------------------------------------------------------------------
VIDEO_STRIDE = 3            # Process every Nth frame by default

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
MAX_UPLOAD_MB = 100          # Max upload file size in megabytes

# ---------------------------------------------------------------------------
# 21-class map for violation_model.pt  (Model 1)
#   Vehicles   : 0-9
#   Helmet     : 10-13
#   Signals    : 14-17
#   Infra      : 18-20
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0:  "bus",
    1:  "car",
    2:  "motorcycle",
    3:  "truck",
    4:  "three_wheeler",
    5:  "tractor",
    6:  "van",
    7:  "vikram",
    8:  "two_wheeler",
    9:  "bike",
    10: "with_helmet",
    11: "without_helmet",
    12: "rider_with_helmet",
    13: "rider_without_helmet",
    14: "red_light",
    15: "green_light",
    16: "yellow_light",
    17: "traffic_light",
    18: "stop_line",
    19: "fixed_obstacle",
    20: "modified",
}

# Semantic groupings used by violation analyzers
VEHICLE_CLASS_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
TWO_WHEELER_CLASS_IDS = {2, 8, 9}           # motorcycle, two_wheeler, bike
THREE_WHEELER_CLASS_IDS = {4, 7}             # three_wheeler, vikram
HELMET_CLASS_IDS = {10, 11, 12, 13}
RIDER_CLASS_IDS = {12, 13}                   # rider_with/without_helmet
NO_HELMET_CLASS_IDS = {11, 13}               # without_helmet, rider_without_helmet
SIGNAL_CLASS_IDS = {14, 15, 16, 17}
RED_LIGHT_ID = 14
GREEN_LIGHT_ID = 15
YELLOW_LIGHT_ID = 16
TRAFFIC_LIGHT_ID = 17
STOP_LINE_ID = 18
MODIFIED_ID = 20

# ---------------------------------------------------------------------------
# Spatial association thresholds
# Each threshold is tuned for the typical camera angles in Indian traffic
# footage.  All values are documented for auditability.
# ---------------------------------------------------------------------------

# IoU between a rider/helmet detection and a two-wheeler bbox.
# Riders usually overlap significantly with their vehicle.
RIDER_VEHICLE_IOU = 0.30

# IoU between a helmet detection and a rider bbox.
# Helmets are small relative to riders; a lower threshold is appropriate.
HELMET_RIDER_IOU = 0.20

# IoU between a license plate bbox and the parent vehicle bbox.
# Plates sit at the edge of vehicles, so overlap can be modest.
PLATE_VEHICLE_IOU = 0.15

# Maximum pixel distance (centroid-to-centroid) to associate a rider with
# a vehicle when IoU alone is insufficient (fallback heuristic).
RIDER_VEHICLE_MAX_DIST = 150  # pixels

# How many pixels below the stop-line's y-center a vehicle's bottom edge
# must extend to be considered "past" the line.
STOP_LINE_MARGIN_PX = 50

# ---------------------------------------------------------------------------
# Composite confidence weights
#
#   composite = W_DETECTION × detection_conf
#             + W_SPATIAL   × spatial_conf
#             + W_TEMPORAL  × temporal_conf
#
# For single-image mode, temporal_conf is unavailable.  Weights are
# redistributed: detection gets 62.5%, spatial gets 37.5%.
# ---------------------------------------------------------------------------
W_DETECTION = 0.50     # Weight for raw model detection confidence
W_SPATIAL = 0.30       # Weight for spatial association quality
W_TEMPORAL = 0.20      # Weight for temporal consistency (video/stream only)

# When temporal data is unavailable (single image), redistribute:
W_DETECTION_IMG = 0.625   # 0.50 / (0.50 + 0.30) * 1.0
W_SPATIAL_IMG = 0.375     # 0.30 / (0.50 + 0.30) * 1.0

# ---------------------------------------------------------------------------
# Severity thresholds  (applied to the composite confidence score)
# ---------------------------------------------------------------------------
SEVERITY_HIGH = 0.80     # composite >= 0.80 → HIGH
SEVERITY_MEDIUM = 0.50   # composite >= 0.50 → MEDIUM
                          # composite <  0.50 → LOW

# ---------------------------------------------------------------------------
# Tracker settings (centroid-based, for temporal consistency)
# ---------------------------------------------------------------------------
TRACKER_MAX_DISAPPEARED = 15   # Frames before a track is dropped
TRACKER_MAX_DISTANCE = 100     # Max centroid distance to match a track (px)
TRACKER_HISTORY_LEN = 10       # Number of recent frames to consider for
                                # temporal consistency ratio

# ---------------------------------------------------------------------------
# Illegal parking — no-parking zone polygons
#
# Each zone is a list of (x, y) tuples defining a polygon in pixel coords.
# These are scene-specific and must be configured per camera.
# Ship empty — no illegal-parking violations will fire until zones are set.
# ---------------------------------------------------------------------------
NO_PARKING_ZONES: list[list[tuple[int, int]]] = []

# Minimum frames a vehicle must be stationary in a zone to count as parked.
PARKING_MIN_STATIONARY_FRAMES = 30

# Maximum centroid movement (px) across frames to be considered stationary.
PARKING_MAX_MOVEMENT_PX = 10

# ---------------------------------------------------------------------------
# OCR settings
# ---------------------------------------------------------------------------
# CLAHE parameters for plate crop preprocessing
OCR_CLAHE_CLIP_LIMIT = 2.0
OCR_CLAHE_TILE_GRID = (8, 8)

# Bilateral filter parameters for noise reduction while preserving edges
OCR_BILATERAL_D = 11
OCR_BILATERAL_SIGMA_COLOR = 17
OCR_BILATERAL_SIGMA_SPACE = 17

# Indian license plate regex pattern:
#   <2 state letters> <2 RTO digits> <1-3 series letters> <4 digits>
# e.g., KA01AB1234, MH12DE5678
PLATE_REGEX = r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$"

# Common OCR character confusions  (char_seen → correct_replacement)
# Applied context-sensitively: digit positions get digit corrections,
# letter positions get letter corrections.
OCR_DIGIT_CORRECTIONS = {
    "O": "0",
    "I": "1",
    "B": "8",
    "S": "5",
    "Z": "2",
    "G": "6",
    "T": "7",
    "l": "1",
    "o": "0",
    "s": "5",
    "z": "2",
}

OCR_LETTER_CORRECTIONS = {
    "0": "O",
    "1": "I",
    "8": "B",
    "5": "S",
    "2": "Z",
    "6": "G",
    "7": "T",
}
