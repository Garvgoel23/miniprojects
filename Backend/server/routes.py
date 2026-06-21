"""
routes.py — Thin API route handlers.

All real processing is delegated to pipeline.py.  Route handlers are
responsible only for:
  1. Parsing/validating request inputs
  2. Calling the pipeline
  3. Formatting the response
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    File,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse

from server.app import get_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------


@router.get("/health")
async def health():
    """Server health check + model status."""
    pipe = get_pipeline()
    return pipe.get_health()


# ---------------------------------------------------------------------------
# POST /api/analyze/image
# ---------------------------------------------------------------------------


@router.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    return_annotated: bool = Query(False, description="Return annotated image (not implemented)"),
):
    """Upload an image and receive violation analysis.

    Accepts JPEG/PNG images via multipart form upload.
    """
    pipe = get_pipeline()
    image_bytes = await file.read()

    if not image_bytes:
        return JSONResponse(
            status_code=400,
            content={"error": "Empty file uploaded"},
        )

    # Run pipeline (synchronous, but wrapped in executor to not block event loop)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, pipe.process_image, image_bytes)

    return {"report": result}


# ---------------------------------------------------------------------------
# POST /api/analyze/video
# ---------------------------------------------------------------------------


@router.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    stride: int = Query(None, description="Process every Nth frame"),
):
    """Upload a video and receive per-frame violation analysis.

    The video is processed frame-by-frame with the given stride (default: 3).
    """
    pipe = get_pipeline()
    video_bytes = await file.read()

    if not video_bytes:
        return JSONResponse(
            status_code=400,
            content={"error": "Empty file uploaded"},
        )

    # Run pipeline in executor
    loop = asyncio.get_event_loop()
    frames = await loop.run_in_executor(
        None, pipe.process_video, video_bytes, stride
    )

    # Aggregate summary across all frames
    total_by_type: dict[str, int] = {}
    total_by_severity: dict[str, int] = {}
    total_violations = 0
    for f in frames:
        total_violations += f["summary"]["total_violations"]
        for vtype, count in f["summary"]["by_type"].items():
            total_by_type[vtype] = total_by_type.get(vtype, 0) + count
        for sev, count in f["summary"]["by_severity"].items():
            total_by_severity[sev] = total_by_severity.get(sev, 0) + count

    return {
        "total_frames_analyzed": len(frames),
        "frames": frames,
        "summary": {
            "total_violations": total_violations,
            "by_type": total_by_type,
            "by_severity": total_by_severity,
        },
    }


# ---------------------------------------------------------------------------
# WS /api/analyze/stream
# ---------------------------------------------------------------------------


@router.websocket("/analyze/stream")
async def analyze_stream(ws: WebSocket):
    """Live stream analysis via WebSocket.

    Client sends a JSON config message first:
        {"stream_url": "rtsp://...", "stride": 5}

    Server then sends back per-frame JSON results continuously.
    Send {"command": "stop"} to end the stream.
    """
    await ws.accept()
    pipe = get_pipeline()

    try:
        # Wait for initial config message
        config_msg = await ws.receive_text()
        cfg = json.loads(config_msg)
        stream_url = cfg.get("stream_url", "0")
        stride = cfg.get("stride", 3)

        # Try to parse webcam index
        try:
            stream_source = int(stream_url)
        except (ValueError, TypeError):
            stream_source = stream_url

        logger.info("Starting stream analysis: source=%s, stride=%d", stream_source, stride)

        # Open video capture in a separate thread
        loop = asyncio.get_event_loop()

        cap = cv2.VideoCapture(stream_source)
        if not cap.isOpened():
            await ws.send_json({"error": f"Cannot open stream: {stream_url}"})
            await ws.close()
            return

        frame_idx = 0
        pipe.tracker = __import__("engine.tracker", fromlist=["CentroidTracker"]).CentroidTracker()

        try:
            while True:
                # Check for stop command (non-blocking)
                try:
                    msg = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                    data = json.loads(msg)
                    if data.get("command") == "stop":
                        logger.info("Stream stop requested")
                        break
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    break

                ret, frame = cap.read()
                if not ret:
                    await ws.send_json({"error": "Stream ended", "frame_id": frame_idx})
                    break

                if frame_idx % stride == 0:
                    result = await loop.run_in_executor(
                        None,
                        pipe.process_stream_frame,
                        frame,
                        frame_idx,
                        float(frame_idx),
                    )
                    await ws.send_json(result)

                frame_idx += 1

        finally:
            cap.release()

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("Stream error: %s", e, exc_info=True)
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# POST /api/analyze/dataset
# ---------------------------------------------------------------------------


@router.post("/analyze/dataset")
async def analyze_dataset(
    dataset_path: str = Query(..., description="Absolute path to dataset directory"),
    dataset_type: str = Query("violation", description="Dataset type: 'violation' or 'plate'"),
):
    """Validate dataset quality — checks label files, class distribution, etc.

    This is a utility endpoint for development/training workflow.
    """
    path = Path(dataset_path)

    if not path.exists():
        return JSONResponse(
            status_code=400,
            content={"error": f"Dataset path does not exist: {dataset_path}"},
        )

    issues: list[str] = []
    class_distribution: dict[str, int] = {}
    total_images = 0
    total_labels = 0

    # Look for images and labels in common YOLO directory structures
    image_dirs = []
    label_dirs = []
    for sub in ["train", "val", "test", ""]:
        img_dir = path / sub / "images" if sub else path / "images"
        lbl_dir = path / sub / "labels" if sub else path / "labels"
        if img_dir.exists():
            image_dirs.append(img_dir)
        if lbl_dir.exists():
            label_dirs.append(lbl_dir)

    if not image_dirs:
        issues.append("No 'images' directory found")
    if not label_dirs:
        issues.append("No 'labels' directory found")

    # Count images
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for img_dir in image_dirs:
        for f in img_dir.iterdir():
            if f.suffix.lower() in image_exts:
                total_images += 1

    # Parse labels and build class distribution
    for lbl_dir in label_dirs:
        for f in lbl_dir.iterdir():
            if f.suffix == ".txt":
                total_labels += 1
                try:
                    with open(f) as fh:
                        for line in fh:
                            parts = line.strip().split()
                            if parts:
                                cls_id = parts[0]
                                cls_name = f"class_{cls_id}"
                                class_distribution[cls_name] = (
                                    class_distribution.get(cls_name, 0) + 1
                                )
                except Exception as e:
                    issues.append(f"Error reading {f.name}: {e}")

    if total_images > 0 and total_labels > 0:
        ratio = total_labels / total_images
        if ratio < 0.9:
            issues.append(
                f"Label coverage is low: {total_labels}/{total_images} "
                f"({ratio:.0%}) — some images may be missing labels"
            )

    return {
        "dataset_path": str(path),
        "dataset_type": dataset_type,
        "total_images": total_images,
        "total_labels": total_labels,
        "class_distribution": class_distribution,
        "issues": issues,
        "status": "ok" if not issues else "warnings",
    }
