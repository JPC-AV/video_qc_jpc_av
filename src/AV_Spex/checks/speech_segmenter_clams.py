"""
CLAMS-style audio speech and content segmenter.

Adapted from clamsproject/app-inaspeechsegmenter-wrapper
(https://github.com/clamsproject/app-inaspeechsegmenter-wrapper). The upstream
project is distributed under the MIT License.

Modifications from the upstream:
  - CLAMS / MMIF I/O removed; only the Segmenter call is used.
  - Results are written to a CSV file following the AV Spex report CSV format.
  - inaSpeechSegmenter is a soft import so a missing library warns rather than
    aborting startup.
"""

import csv
from pathlib import Path
from typing import List, Optional, Tuple

from AV_Spex.utils.log_setup import logger

# inaSpeechSegmenter requires TensorFlow/Keras; treat as optional so a missing
# install only warns rather than crashing the whole application.
try:
    from inaSpeechSegmenter import Segmenter as _Segmenter
    _INA_AVAILABLE = True
except ImportError:
    _INA_AVAILABLE = False

SEGMENTS_CSV_NAME = "ina_speech_segments.csv"

# Maps the five original inaSpeechSegmenter labels to the four-way
# high-level categories used by the CLAMS wrapper (and this module).
_LABEL_MAP = {
    'male': 'speech',
    'female': 'speech',
    'noEnergy': 'silence',
    'noise': 'noise',
    'music': 'music',
}


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - (hours * 3600) - (minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def write_segments_csv(
    report_directory: str,
    segments: List[Tuple[str, float, float]],
) -> None:
    """Write the speech-segmenter results CSV.

    Each entry is (original_label, start_seconds, end_seconds). The mapped
    high-level label is computed here from _LABEL_MAP and written as the first
    column; the original label (including gender) is written in the second.
    """
    path = Path(report_directory) / SEGMENTS_CSV_NAME
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if segments:
            writer.writerow(["label", "original_label", "start", "end"])
            for orig_label, start_sec, end_sec in segments:
                label = _LABEL_MAP.get(orig_label, orig_label)
                writer.writerow([
                    label,
                    orig_label,
                    _format_timestamp(start_sec),
                    _format_timestamp(end_sec),
                ])
        else:
            writer.writerow(["inaSpeechSegmenter produced no segments"])


def run_ina_speech_segmenter(
    video_path: str,
    report_directory: str,
    video_id: str,
    silence_ratio: int = 3,
    min_segment_duration_ms: int = 0,
    check_cancelled=None,
    signals=None,
) -> List[Tuple[str, float, float]]:
    """
    Classify audio content in `video_path` using inaSpeechSegmenter.

    Returns a list of (original_label, start_seconds, end_seconds) tuples.
    Labels are the raw inaSpeechSegmenter values: 'male', 'female', 'noEnergy',
    'noise', 'music'. Use _LABEL_MAP to convert to high-level categories.

    Does NOT write the segments CSV — call write_segments_csv after this so the
    caller can compose results before writing.

    Returns an empty list when the library is unavailable or analysis fails.
    """
    if not _INA_AVAILABLE:
        logger.warning(
            "inaSpeechSegmenter: library not installed — skipping. "
            "Install with: pip install inaSpeechSegmenter"
        )
        return []

    if check_cancelled and check_cancelled():
        return []

    logger.info(f"inaSpeechSegmenter: analysing audio in {video_path!r}")
    if signals and hasattr(signals, 'ina_segmenter_progress'):
        signals.ina_segmenter_progress.emit(0)

    try:
        segmenter = _Segmenter()
        segmenter.energy_ratio = silence_ratio
        raw_segments = segmenter(video_path)
    except Exception as exc:
        logger.warning(f"inaSpeechSegmenter: analysis failed: {exc}")
        if signals and hasattr(signals, 'ina_segmenter_progress'):
            signals.ina_segmenter_progress.emit(100)
        return []

    min_duration_s = min_segment_duration_ms / 1000.0
    out: List[Tuple[str, float, float]] = [
        (orig_label, float(start_sec), float(end_sec))
        for orig_label, start_sec, end_sec in raw_segments
        if (float(end_sec) - float(start_sec)) >= min_duration_s
    ]

    if signals and hasattr(signals, 'ina_segmenter_progress'):
        signals.ina_segmenter_progress.emit(100)

    counts: dict = {}
    for orig_label, _, _ in out:
        label = _LABEL_MAP.get(orig_label, orig_label)
        counts[label] = counts.get(label, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    if out:
        logger.debug(f"inaSpeechSegmenter: {len(out)} segment(s): {summary}\n")
    else:
        logger.debug("inaSpeechSegmenter: no segments met the minimum duration threshold.\n")

    return out
