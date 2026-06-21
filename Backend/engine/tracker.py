"""
tracker.py — Simple centroid-based object tracker for temporal consistency.

Assigns persistent IDs to detections across frames by nearest-centroid
matching.  No deep features or re-identification — keeps dependencies minimal.

Used to:
  1. Track vehicles across video frames for temporal violation consistency.
  2. Detect stationary vehicles (for illegal parking).
  3. Provide temporal_consistency scores (fraction of recent frames where a
     track was flagged for a given violation type).
"""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import numpy as np

import config
from engine.spatial_utils import centroid, centroid_distance

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """State for a single tracked object."""

    track_id: int
    centroid: tuple[float, float]
    bbox: list[float]
    disappeared: int = 0
    # History of centroids for stationarity detection
    centroid_history: list[tuple[float, float]] = field(default_factory=list)
    # Per-violation-type history: {violation_type: [bool, bool, ...]}
    # Each entry records whether this track was flagged in a recent frame.
    violation_history: dict[str, list[bool]] = field(
        default_factory=lambda: defaultdict(list)
    )


class CentroidTracker:
    """Track objects across frames via centroid proximity.

    Args:
        max_disappeared: Frames before a track is dropped.
        max_distance: Max centroid distance (px) for matching.
        history_len: Number of recent frames for temporal consistency.
    """

    def __init__(
        self,
        max_disappeared: int = config.TRACKER_MAX_DISAPPEARED,
        max_distance: float = config.TRACKER_MAX_DISTANCE,
        history_len: int = config.TRACKER_HISTORY_LEN,
    ):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.history_len = history_len
        self._next_id = 0
        self.tracks: OrderedDict[int, Track] = OrderedDict()

    def update(self, bboxes: list[list[float]]) -> dict[int, list[float]]:
        """Update tracks with new detections.

        Args:
            bboxes: List of [x1, y1, x2, y2] for current-frame detections.

        Returns:
            Dict mapping track_id → bbox for all active tracks.
        """
        # No detections — mark all existing tracks as disappeared
        if len(bboxes) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid].disappeared += 1
                if self.tracks[tid].disappeared > self.max_disappeared:
                    del self.tracks[tid]
            return {tid: t.bbox for tid, t in self.tracks.items()}

        input_centroids = [centroid(b) for b in bboxes]

        # No existing tracks — register all detections
        if len(self.tracks) == 0:
            for i, bbox in enumerate(bboxes):
                self._register(bbox, input_centroids[i])
            return {tid: t.bbox for tid, t in self.tracks.items()}

        # Match existing tracks to new detections by centroid distance
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]

        # Compute pairwise distances
        D = np.zeros((len(track_ids), len(input_centroids)), dtype=np.float64)
        for r, tc in enumerate(track_centroids):
            for c, ic in enumerate(input_centroids):
                D[r, c] = np.sqrt((tc[0] - ic[0]) ** 2 + (tc[1] - ic[1]) ** 2)

        # Greedy matching: smallest distance first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)

        used_rows = set()
        used_cols = set()

        for row in rows:
            col = cols[row]
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            tid = track_ids[row]
            self.tracks[tid].centroid = input_centroids[col]
            self.tracks[tid].bbox = bboxes[col]
            self.tracks[tid].disappeared = 0

            # Update centroid history for stationarity detection
            hist = self.tracks[tid].centroid_history
            hist.append(input_centroids[col])
            if len(hist) > self.history_len:
                hist.pop(0)

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched tracks (disappeared) and new detections (register)
        for row in range(len(track_ids)):
            if row not in used_rows:
                tid = track_ids[row]
                self.tracks[tid].disappeared += 1
                if self.tracks[tid].disappeared > self.max_disappeared:
                    del self.tracks[tid]

        for col in range(len(input_centroids)):
            if col not in used_cols:
                self._register(bboxes[col], input_centroids[col])

        return {tid: t.bbox for tid, t in self.tracks.items()}

    def record_violation(self, track_id: int, violation_type: str, flagged: bool):
        """Record whether *track_id* was flagged for *violation_type* this frame."""
        if track_id not in self.tracks:
            return
        hist = self.tracks[track_id].violation_history[violation_type]
        hist.append(flagged)
        if len(hist) > self.history_len:
            hist.pop(0)

    def temporal_consistency(self, track_id: int, violation_type: str) -> float:
        """Fraction of recent frames where *track_id* was flagged for *violation_type*.

        Returns 0.0 if the track doesn't exist or has no history.
        """
        if track_id not in self.tracks:
            return 0.0
        hist = self.tracks[track_id].violation_history.get(violation_type, [])
        if not hist:
            return 0.0
        return sum(1 for v in hist if v) / len(hist)

    def is_stationary(
        self,
        track_id: int,
        max_movement: float = config.PARKING_MAX_MOVEMENT_PX,
    ) -> bool:
        """Check if a tracked object has remained approximately stationary.

        Compares the centroid at the start and end of its history window.
        """
        if track_id not in self.tracks:
            return False
        hist = self.tracks[track_id].centroid_history
        if len(hist) < 2:
            return False

        # Max displacement from first centroid in history
        first = hist[0]
        max_disp = max(
            np.sqrt((c[0] - first[0]) ** 2 + (c[1] - first[1]) ** 2) for c in hist
        )
        return max_disp <= max_movement

    def get_track_for_bbox(
        self, bbox: list[float], max_dist: float | None = None
    ) -> int | None:
        """Find the track_id closest to *bbox*'s centroid, if within max_dist."""
        if max_dist is None:
            max_dist = self.max_distance
        c = centroid(bbox)
        best_id = None
        best_dist = float("inf")
        for tid, track in self.tracks.items():
            d = np.sqrt((track.centroid[0] - c[0]) ** 2 + (track.centroid[1] - c[1]) ** 2)
            if d < best_dist and d <= max_dist:
                best_dist = d
                best_id = tid
        return best_id

    def _register(self, bbox: list[float], c: tuple[float, float]):
        """Register a new track."""
        self.tracks[self._next_id] = Track(
            track_id=self._next_id,
            centroid=c,
            bbox=bbox,
            centroid_history=[c],
        )
        self._next_id += 1
