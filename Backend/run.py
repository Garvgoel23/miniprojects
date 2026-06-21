"""
run.py — CLI entry point for the Traffic Violation Detection System.

Usage:
    # Start the FastAPI server
    python run.py --server

    # Analyze a single image
    python run.py --source path/to/image.jpg

    # Analyze a video (every 3rd frame)
    python run.py --source path/to/video.mp4 --stride 3

    # Analyze webcam feed
    python run.py --source 0

    # Validate a dataset
    python run.py --analyze-dataset ../datasets/combined-full --type violation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the Backend directory is on sys.path so imports work
# regardless of CWD
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import config


def setup_logging(verbose: bool = False):
    """Configure root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def run_server(host: str, port: int):
    """Start the FastAPI server via uvicorn."""
    import uvicorn

    from server.app import create_app

    app = create_app()
    print(f"\n🚦 Traffic Violation Detection API")
    print(f"   Server:  http://{host}:{port}")
    print(f"   Docs:    http://{host}:{port}/docs")
    print(f"   Device:  {config.DEVICE}\n")

    uvicorn.run(app, host=host, port=port)


def run_source(source: str, stride: int, output: str | None):
    """Run detection on an image, video, or webcam feed."""
    from pipeline import Pipeline

    pipe = Pipeline()

    # Determine source type
    source_path = Path(source) if not source.isdigit() else None

    if source.isdigit():
        # Webcam
        print(f"📷 Opening webcam {source} (press Ctrl+C to stop)...")
        import cv2

        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            print(f"❌ Cannot open webcam {source}")
            sys.exit(1)

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % stride == 0:
                    result = pipe.process_stream_frame(
                        frame, frame_id=frame_idx, timestamp=float(frame_idx)
                    )
                    n = result["summary"]["total_violations"]
                    if n > 0:
                        print(
                            f"  Frame {frame_idx}: {n} violation(s) — "
                            + ", ".join(
                                f"{v['type']} ({v['severity']})"
                                for v in result["violations"]
                            )
                        )
                frame_idx += 1
        except KeyboardInterrupt:
            print("\n⏹ Stopped")
        finally:
            cap.release()

    elif source_path and source_path.exists():
        ext = source_path.suffix.lower()
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

        if ext in image_exts:
            # Single image
            print(f"🖼 Analyzing image: {source_path.name}")
            image_bytes = source_path.read_bytes()
            result = pipe.process_image(image_bytes)
            _print_result(result, output)

        elif ext in video_exts:
            # Video file
            print(f"🎬 Analyzing video: {source_path.name} (stride={stride})")
            results = pipe.process_video_path(str(source_path), stride=stride)
            aggregate = {
                "total_frames_analyzed": len(results),
                "frames": results,
            }
            _print_result(aggregate, output)
            # Print summary
            total_v = sum(f["summary"]["total_violations"] for f in results)
            print(f"\n📊 {total_v} total violation(s) across {len(results)} frames")

        else:
            print(f"❌ Unsupported file type: {ext}")
            sys.exit(1)
    else:
        print(f"❌ Source not found: {source}")
        sys.exit(1)


def run_dataset_analysis(dataset_path: str, dataset_type: str):
    """Run dataset quality validation."""
    path = Path(dataset_path)
    if not path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    print(f"📂 Analyzing {dataset_type} dataset: {path}")

    # Reuse the dataset analysis logic from routes (but run directly)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    issues: list[str] = []
    class_dist: dict[str, int] = {}
    total_images = 0
    total_labels = 0

    for sub in ["train", "val", "test", ""]:
        img_dir = path / sub / "images" if sub else path / "images"
        lbl_dir = path / sub / "labels" if sub else path / "labels"

        if img_dir.exists():
            for f in img_dir.iterdir():
                if f.suffix.lower() in image_exts:
                    total_images += 1

        if lbl_dir.exists():
            for f in lbl_dir.iterdir():
                if f.suffix == ".txt":
                    total_labels += 1
                    try:
                        with open(f) as fh:
                            for line in fh:
                                parts = line.strip().split()
                                if parts:
                                    cls_name = config.CLASS_NAMES.get(
                                        int(parts[0]), f"class_{parts[0]}"
                                    )
                                    class_dist[cls_name] = (
                                        class_dist.get(cls_name, 0) + 1
                                    )
                    except Exception as e:
                        issues.append(f"Error reading {f.name}: {e}")

    print(f"  Images: {total_images}")
    print(f"  Labels: {total_labels}")
    if class_dist:
        print("  Class distribution:")
        for cls, count in sorted(class_dist.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {count}")
    if issues:
        print(f"  ⚠ {len(issues)} issue(s):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ No issues found")


def _print_result(result: dict, output: str | None):
    """Print or save a result dict as formatted JSON."""
    json_str = json.dumps(result, indent=2, default=str)
    if output:
        Path(output).write_text(json_str, encoding="utf-8")
        print(f"💾 Results saved to {output}")
    else:
        print(json_str)


def main():
    parser = argparse.ArgumentParser(
        description="🚦 Traffic Violation Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--server",
        action="store_true",
        help="Start the FastAPI server",
    )
    group.add_argument(
        "--source",
        type=str,
        help="Path to image/video or webcam index (0, 1, ...)",
    )
    group.add_argument(
        "--analyze-dataset",
        type=str,
        metavar="PATH",
        help="Path to dataset directory for quality validation",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=config.VIDEO_STRIDE,
        help=f"Process every Nth video frame (default: {config.VIDEO_STRIDE})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=config.SERVER_HOST,
        help=f"Server host (default: {config.SERVER_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.SERVER_PORT,
        help=f"Server port (default: {config.SERVER_PORT})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save JSON results to file instead of stdout",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="violation",
        choices=["violation", "plate"],
        help="Dataset type for --analyze-dataset (default: violation)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.server:
        run_server(args.host, args.port)
    elif args.source:
        run_source(args.source, args.stride, args.output)
    elif args.analyze_dataset:
        run_dataset_analysis(args.analyze_dataset, args.type)


if __name__ == "__main__":
    main()
