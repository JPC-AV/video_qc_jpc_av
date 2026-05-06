"""
Standalone CLAMS SMPTE color bars detector (experimental).

Detection algorithm vendored from clamsproject/app-barsdetection (Apache-2.0).
See LICENSE in this directory for the upstream license. The MMIF / clams-python
framework wrapping has been stripped — this script reads a video file directly
with OpenCV and writes a JSON sidecar.

Dependencies (install in your active Python env):
    pip install opencv-python scikit-image numpy

Usage:
    python run_clams_bars.py /path/to/video.mkv
    python run_clams_bars.py /path/to/video.mkv -o /tmp/results.json
    python run_clams_bars.py /path/to/video.mkv --threshold 0.6 --no-stop-after-one

Security note: grey.p is a Python pickle. It will execute arbitrary code on
load if tampered with. Only use the grey.p shipped alongside this script (from
the upstream CLAMS repo).
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
from skimage.metrics import structural_similarity


DEFAULTS = {
    "threshold": 0.7,
    "minFrameCount": 10,
    "sampleRatio": 30,
    "stopAt": 5 * 60 * 30,  # ~5 minutes at 30 fps
    "stopAfterOne": True,
}

GREY_P_PATH = Path(__file__).parent / "grey.p"


def run_detection(video_path, threshold, minFrameCount, sampleRatio, stopAt, stopAfterOne, verbose=False):
    with open(GREY_P_PATH, "rb") as p:
        grey = pickle.load(p)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    upper = min(stopAt, total_frames) if total_frames > 0 else stopAt
    frames_to_test = list(range(0, upper, sampleRatio))

    def frame_in_range(frame_):
        f = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
        if f.shape != grey.shape:
            f = cv2.resize(f, (grey.shape[1], grey.shape[0]))
        score = structural_similarity(f, grey)
        if verbose:
            print(f"  frame {cur_frame}: score={score:.4f}  bars={score > threshold}")
        return score > threshold

    bars_found = []
    in_slate = False
    start_frame = None
    cur_frame = frames_to_test[0] if frames_to_test else 0
    for cur_frame in frames_to_test:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame - 1)
        ret, frame = cap.read()
        if not ret:
            break
        if frame_in_range(frame):
            if not in_slate:
                in_slate = True
                start_frame = cur_frame
        elif in_slate:
            in_slate = False
            if cur_frame - start_frame > minFrameCount:
                bars_found.append((start_frame, cur_frame))
            if stopAfterOne:
                cap.release()
                return bars_found, fps, total_frames
    if in_slate:
        if cur_frame - start_frame > minFrameCount:
            bars_found.append((start_frame, cur_frame))

    cap.release()
    return bars_found, fps, total_frames


def main():
    parser = argparse.ArgumentParser(
        description="Standalone CLAMS SMPTE color bars detector (experimental).",
    )
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output JSON path. Default: <video_stem>_clams_bars.json next to the video.")
    parser.add_argument("--threshold", type=float, default=DEFAULTS["threshold"],
                        help=f"SSIM threshold for a frame to count as bars (default: {DEFAULTS['threshold']})")
    parser.add_argument("--min-frame-count", type=int, default=DEFAULTS["minFrameCount"],
                        help=f"Minimum run-length to register a bars region (default: {DEFAULTS['minFrameCount']})")
    parser.add_argument("--sample-ratio", type=int, default=DEFAULTS["sampleRatio"],
                        help=f"Sample one frame every N frames (default: {DEFAULTS['sampleRatio']})")
    parser.add_argument("--stop-at", type=int, default=DEFAULTS["stopAt"],
                        help=f"Stop scanning after this frame index (default: {DEFAULTS['stopAt']} = ~5min @ 30fps)")
    parser.add_argument("--no-stop-after-one", action="store_true",
                        help="Detect all bar regions; default stops after the first.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-frame SSIM scores.")
    args = parser.parse_args()

    if not args.video.exists():
        parser.error(f"Video not found: {args.video}")

    output_path = args.output or args.video.with_name(f"{args.video.stem}_clams_bars.json")
    stop_after_one = not args.no_stop_after_one

    print(f"Running CLAMS bars detection on: {args.video}")
    print(f"  threshold={args.threshold}, sample_ratio={args.sample_ratio}, "
          f"stop_at={args.stop_at}, stop_after_one={stop_after_one}")

    t0 = time.perf_counter()
    bars, fps, total_frames = run_detection(
        args.video,
        threshold=args.threshold,
        minFrameCount=args.min_frame_count,
        sampleRatio=args.sample_ratio,
        stopAt=args.stop_at,
        stopAfterOne=stop_after_one,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0

    results = {
        "video": str(args.video),
        "fps": fps,
        "total_frames": total_frames,
        "elapsed_seconds": round(elapsed, 3),
        "params": {
            "threshold": args.threshold,
            "minFrameCount": args.min_frame_count,
            "sampleRatio": args.sample_ratio,
            "stopAt": args.stop_at,
            "stopAfterOne": stop_after_one,
        },
        "bars_found": [
            {
                "start_frame": int(s),
                "end_frame": int(e),
                "start_time": round(s / fps, 3) if fps else None,
                "end_time": round(e / fps, 3) if fps else None,
            }
            for s, e in bars
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nElapsed: {elapsed:.2f}s  ({total_frames} total frames, fps={fps:.3f})")
    print(f"Bars regions detected: {len(bars)}")
    for region in results["bars_found"]:
        print(f"  frames {region['start_frame']}-{region['end_frame']}  "
              f"({region['start_time']}s-{region['end_time']}s)")
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
