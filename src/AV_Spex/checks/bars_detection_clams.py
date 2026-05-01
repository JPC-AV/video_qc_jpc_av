"""
CLAMS-style SMPTE color bars detector.

Adapted from clamsproject/app-barsdetection
(https://github.com/clamsproject/app-barsdetection). The upstream project is
distributed under the Apache License 2.0; the full upstream license text lives
at src/AV_Spex/config/clams_bars/LICENSE.

Modifications from the upstream:
  - CLAMS / MMIF I/O removed; only the SSIM-based detection core is ported.
  - The pickled reference array (grey.p) is replaced by a PNG asset
    (config/clams_bars/smpte_bars_reference.png). Pixel values are byte-equal
    to the original, so SSIM scores match.
  - Source FPS is read via ffprobe instead of an MMIF VideoDocument property.
  - Per-frame SSIM scores are written to their own CSV for review.
  - The durations CSV mirrors AV Spex's qct-parse format so a single parser
    handles both detectors' output.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from skimage.metrics import structural_similarity

from AV_Spex.utils.log_setup import logger


REFERENCE_FILENAME = "smpte_bars_reference.png"
DURATIONS_CSV_NAME = "clams_bars_colorbars_durations.csv"
SSIM_SCORES_CSV_SUFFIX = "_clams_bars_ssim_scores.csv"


def _bundle_dir() -> str:
    """Return the AV_Spex package root, in dev or PyInstaller-frozen mode."""
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "AV_Spex")
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_reference_path() -> str:
    path = os.path.join(_bundle_dir(), "config", "clams_bars", REFERENCE_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CLAMS bars reference image not found at {path}. "
            "Expected a bundled grayscale PNG."
        )
    return path


def _get_video_fps(video_path: str) -> Optional[float]:
    """Read the video stream's frame rate via ffprobe. Returns None on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        for key in ("r_frame_rate", "avg_frame_rate"):
            value = stream.get(key, "")
            if "/" in value:
                num, den = value.split("/")
                num, den = float(num), float(den)
                if den > 0:
                    return num / den
            elif value:
                return float(value)
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, IndexError, ValueError):
        pass
    return None


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - (hours * 3600) - (minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def write_durations_csv(report_directory: str, runs: List[Tuple[str, float, float]]) -> None:
    """
    Write the bars-detection durations CSV from a combined list of runs.

    Each entry is (pass_label, start_seconds, end_seconds). The pass label is
    written as the first column so the report can distinguish primary detections
    from cross-validation second-pass hits.
    """
    path = Path(report_directory) / DURATIONS_CSV_NAME
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if runs:
            writer.writerow(["clams bars detection color bars found:"])
            for pass_label, start_seconds, end_seconds in runs:
                writer.writerow([
                    pass_label,
                    _format_timestamp(start_seconds),
                    _format_timestamp(end_seconds),
                ])
        else:
            writer.writerow(["clams bars detection found no color bars"])


def run_clams_bars_detection(
    video_path: str,
    report_directory: str,
    video_id: str,
    threshold: float = 0.7,
    sample_ratio: int = 30,
    stop_at_frame: int = 9000,
    min_frame_count: int = 10,
    stop_after_one: bool = True,
    fps: Optional[float] = None,
    check_cancelled=None,
    signals=None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Detect SMPTE color bars by SSIM comparison against a bundled reference image.

    Parameters mirror the upstream CLAMS app's runtime parameters.

    Returns:
        (start_seconds, end_seconds) for the first detected run, or (None, None)
        if no run met the minimum frame count.

    Always writes a durations CSV in the report directory; also writes a
    per-sampled-frame SSIM scores CSV alongside it.
    """
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    durations_csv = report_path / DURATIONS_CSV_NAME
    ssim_csv = report_path / f"{video_id}{SSIM_SCORES_CSV_SUFFIX}"

    if fps is None:
        fps = _get_video_fps(video_path)
    if not fps or fps <= 0:
        logger.warning(f"CLAMS bars detection: could not determine fps for {video_path}; skipping.")
        _write_durations_csv(durations_csv, None, None)
        return None, None

    reference = cv2.imread(_resolve_reference_path(), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        logger.warning("CLAMS bars detection: failed to load reference PNG; skipping.")
        _write_durations_csv(durations_csv, None, None)
        return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"CLAMS bars detection: could not open {video_path}; skipping.")
        _write_durations_csv(durations_csv, None, None)
        return None, None

    sampled_frames = list(range(0, max(stop_at_frame, 0), max(sample_ratio, 1)))
    bars_runs = []
    in_run = False
    start_frame = None
    cur_frame = sampled_frames[0] if sampled_frames else 0

    total_samples = len(sampled_frames)
    progress_interval = max(1, total_samples // 100)
    last_progress_pct = 0
    emit_progress = signals is not None and hasattr(signals, 'clams_detection_progress') and total_samples > 0

    try:
        with open(ssim_csv, "w", newline="") as ssim_file:
            ssim_writer = csv.writer(ssim_file)
            ssim_writer.writerow(["frame", "timestamp", "ssim_score", "exceeds_threshold"])

            for sample_idx, cur_frame in enumerate(sampled_frames):
                if check_cancelled and check_cancelled():
                    break

                if emit_progress and sample_idx % progress_interval == 0:
                    pct = int((sample_idx / total_samples) * 100)
                    pct = min(99, max(0, pct))
                    if pct > last_progress_pct:
                        signals.clams_detection_progress.emit(pct)
                        last_progress_pct = pct

                seek_target = max(cur_frame - 1, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray.shape != reference.shape:
                    gray = cv2.resize(gray, (reference.shape[1], reference.shape[0]))
                score = float(structural_similarity(gray, reference))
                exceeds = score > threshold

                ssim_writer.writerow([
                    cur_frame,
                    _format_timestamp(cur_frame / fps),
                    f"{score:.6f}",
                    "true" if exceeds else "false",
                ])

                if exceeds:
                    if not in_run:
                        in_run = True
                        start_frame = cur_frame
                elif in_run:
                    in_run = False
                    if cur_frame - start_frame > min_frame_count:
                        bars_runs.append((start_frame, cur_frame))
                    if stop_after_one:
                        break
    finally:
        cap.release()

    if in_run and start_frame is not None and cur_frame - start_frame > min_frame_count:
        bars_runs.append((start_frame, cur_frame))

    if emit_progress:
        signals.clams_detection_progress.emit(100)

    if bars_runs:
        s_frame, e_frame = bars_runs[0]
        s_seconds = s_frame / fps
        e_seconds = e_frame / fps
        _write_durations_csv(durations_csv, s_seconds, e_seconds)
        logger.debug(
            f"CLAMS bars detection: bars from {_format_timestamp(s_seconds)} "
            f"to {_format_timestamp(e_seconds)}"
        )
        return s_seconds, e_seconds

    _write_durations_csv(durations_csv, None, None)
    logger.debug("CLAMS bars detection: no color bars run met the minimum frame count.\n")
    return None, None
