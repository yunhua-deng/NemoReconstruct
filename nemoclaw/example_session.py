#!/usr/bin/env python3
"""
Example: Drive a NemoReconstruct session programmatically via the Python SDK.

This shows the same workflow a NemoClaw agent would perform — upload a video,
poll for completion, and download the PLY output. Useful for verifying the
backend is working before connecting NemoClaw.

Usage:
    python example_session.py /path/to/video.mov "My Scene"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Assumes the SDK is installed: pip install -e sdk/python
from nemo_reconstruct_client import NemoReconstructClient

POLL_INTERVAL = 15  # seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a NemoReconstruct job end-to-end")
    parser.add_argument("video", help="Path to .MOV / .MP4 / .M4V file")
    parser.add_argument("name", help="Human-readable name for the reconstruction")
    parser.add_argument("--api-url", default="http://127.0.0.1:8010", help="Backend base URL")
    parser.add_argument("--output-dir", default=".", help="Where to save the PLY output")
    parser.add_argument("--frame-rate", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--downsample", type=int, default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    client = NemoReconstructClient(base_url=args.api_url)

    # 1. Health check
    health = client.health()
    print(f"Backend status: {health['status']}")

    # 2. Upload
    params = {}
    if args.frame_rate is not None:
        params["frame_rate"] = args.frame_rate
    if args.max_epochs is not None:
        params["fvdb_max_epochs"] = args.max_epochs
    if args.downsample is not None:
        params["fvdb_image_downsample_factor"] = args.downsample

    job = client.upload_video(video_path, args.name, params=params or None)
    print(f"Uploaded — reconstruction ID: {job.id}")

    # 3. Poll until done
    while True:
        status = client.get_status(job.id)
        step = status.get("processing_step", "unknown")
        pct = status.get("processing_pct", 0)
        state = status["status"]
        print(f"  [{pct:3d}%] {step} — {state}")

        if state in ("completed", "failed"):
            break
        time.sleep(POLL_INTERVAL)

    if state == "failed":
        error = status.get("error_message", "unknown error")
        print(f"Job failed: {error}", file=sys.stderr)
        sys.exit(1)

    # 4. Download PLY
    output_path = Path(args.output_dir) / f"{job.id}.ply"
    client.download_artifact(job.id, "splat_ply", output_path)
    print(f"PLY saved to: {output_path}")


if __name__ == "__main__":
    main()
