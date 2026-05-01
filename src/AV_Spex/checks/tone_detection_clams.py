"""
CLAMS-style monotonic-tone detector.

Adapted from clamsproject/app-tonedetection
(https://github.com/clamsproject/app-tonedetection). The upstream project is
distributed under the Apache License 2.0; the full upstream license text lives
at src/AV_Spex/config/clams_tone/LICENSE.

Modifications from the upstream:
  - CLAMS / MMIF I/O removed; only the cross-correlation detection core is ported.
  - pydub is replaced by an ffmpeg subprocess that decodes to mono 16 kHz s16le
    PCM, which is then read into numpy. This avoids a new Python dependency
    while preserving sample-for-sample equivalence with the upstream loader.
  - The "stopAt" parameter is exposed in seconds (instead of the upstream's
    sample-count value documented as ms), and converted internally.
  - Per-tone results are written to a durations CSV in the same format used by
    qct-parse / clams bars detection, so the report parser can read both.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from AV_Spex.utils.log_setup import logger


DURATIONS_CSV_NAME = "clams_tone_detection_durations.csv"
SAMPLE_RATE = 16000
SAMPLE_SIZE = 4000  # 250 ms at 16 kHz


def _bundle_dir() -> str:
    """Return the AV_Spex package root, in dev or PyInstaller-frozen mode."""
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "AV_Spex")
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - (hours * 3600) - (minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _load_audio_mono_16k(filepath: str) -> Optional[np.ndarray]:
    """
    Decode the audio track of `filepath` to mono 16 kHz s16le PCM via ffmpeg
    and return a float32 numpy array normalized to [-1, 1].

    Returns None on failure (no audio stream, ffmpeg error, etc.).
    """
    cmd = [
        "ffmpeg", "-v", "error", "-i", filepath,
        "-vn",
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.warning(f"CLAMS tone detection: ffmpeg invocation failed: {exc}")
        return None
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        logger.warning(f"CLAMS tone detection: ffmpeg returned {proc.returncode}: {err}")
        return None
    if not proc.stdout:
        logger.warning("CLAMS tone detection: ffmpeg produced no audio samples.")
        return None
    raw = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32)
    return raw / 32768.0


def _write_durations_csv(path: Path, tones: List[Tuple[float, float]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if tones:
            writer.writerow(["clams tone detection tones found:"])
            for start_seconds, end_seconds in tones:
                writer.writerow([
                    _format_timestamp(start_seconds),
                    _format_timestamp(end_seconds),
                ])
        else:
            writer.writerow(["clams tone detection found no tones"])


def _detect_tones(
    samples: np.ndarray,
    tolerance: float,
    min_tone_duration_ms: int,
    stop_at_samples: int,
    check_cancelled=None,
    signals=None,
) -> List[Tuple[float, float]]:
    """
    Cross-correlation tone detection over consecutive 250 ms chunks.

    Faithful port of the upstream Tonedetection._detect_tones. Returns a list
    of (start_seconds, end_seconds) tones whose duration meets the threshold.
    """
    total_samples = len(samples)
    endpoint = min(stop_at_samples, total_samples)
    out: List[Tuple[float, float]] = []

    pos = 0
    ref_chunk = samples[pos:pos + SAMPLE_SIZE]
    pos += SAMPLE_SIZE
    curr_chunk = samples[pos:pos + SAMPLE_SIZE]
    chunk_len = len(curr_chunk)
    pos += SAMPLE_SIZE

    start_sample = 0
    duration = SAMPLE_SIZE

    progress_total = max(endpoint, 1)
    last_progress_pct = 0
    emit_progress = signals is not None and hasattr(signals, "clams_detection_progress")

    while chunk_len >= duration and start_sample < endpoint:
        if check_cancelled and check_cancelled():
            break

        if emit_progress:
            pct = int((start_sample / progress_total) * 100)
            pct = min(99, max(0, pct))
            if pct > last_progress_pct:
                signals.clams_detection_progress.emit(pct)
                last_progress_pct = pct

        similarity = float(np.average(np.correlate(ref_chunk, curr_chunk, mode="valid")))
        sim_count = 0
        while similarity >= tolerance:
            sim_count += 1
            duration += SAMPLE_SIZE
            ref_chunk = curr_chunk
            curr_chunk = samples[pos:pos + SAMPLE_SIZE]
            pos += SAMPLE_SIZE
            if len(curr_chunk) < SAMPLE_SIZE:
                break
            similarity = float(np.average(np.correlate(ref_chunk, curr_chunk, mode="valid")))

        if sim_count > 0:
            tone_start = start_sample / SAMPLE_RATE
            tone_end = (start_sample + duration) / SAMPLE_RATE
            out.append((tone_start, tone_end))

        start_sample += duration
        ref_chunk = curr_chunk
        curr_chunk = samples[pos:pos + SAMPLE_SIZE]
        chunk_len = len(curr_chunk)
        pos += SAMPLE_SIZE
        duration = SAMPLE_SIZE

    if emit_progress:
        signals.clams_detection_progress.emit(100)

    threshold_seconds = min_tone_duration_ms / 1000.0
    filtered = [(s, e) for s, e in out if (e - s) >= threshold_seconds]
    return filtered


def run_clams_tone_detection(
    video_path: str,
    report_directory: str,
    video_id: str,
    tolerance: float = 1.0,
    min_tone_duration_ms: int = 2000,
    stop_at_seconds: int = 3600,
    check_cancelled=None,
    signals=None,
) -> List[Tuple[float, float]]:
    """
    Detect spans of monotonic audio in `video_path`.

    Parameters mirror the upstream CLAMS app's runtime parameters, with
    `stop_at_seconds` exposed in seconds for clarity.

    Returns:
        List of (start_seconds, end_seconds) tones; empty when none qualify.

    Always writes a durations CSV in the report directory.
    """
    report_path = Path(report_directory)
    report_path.mkdir(parents=True, exist_ok=True)
    durations_csv = report_path / DURATIONS_CSV_NAME

    samples = _load_audio_mono_16k(video_path)
    if samples is None or len(samples) == 0:
        logger.warning(
            f"CLAMS tone detection: no decodable audio in {video_path}; skipping."
        )
        _write_durations_csv(durations_csv, [])
        return []

    stop_at_samples = max(int(stop_at_seconds), 0) * SAMPLE_RATE

    tones = _detect_tones(
        samples,
        tolerance=tolerance,
        min_tone_duration_ms=min_tone_duration_ms,
        stop_at_samples=stop_at_samples,
        check_cancelled=check_cancelled,
        signals=signals,
    )

    _write_durations_csv(durations_csv, tones)

    if tones:
        logger.debug(
            f"CLAMS tone detection: {len(tones)} tone(s) detected; "
            f"first at {_format_timestamp(tones[0][0])}–{_format_timestamp(tones[0][1])}"
        )
    else:
        logger.debug("CLAMS tone detection: no tones met the minimum duration threshold.\n")

    return tones
