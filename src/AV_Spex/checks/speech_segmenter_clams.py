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
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from AV_Spex.utils.log_setup import logger

# inaSpeechSegmenter is intentionally NOT imported at module level.
# Importing it triggers TensorFlow initialisation (model converter messages,
# TPU client warnings) even when the check is disabled. The import is deferred
# to run_ina_speech_segmenter() so it only happens when the check actually runs.

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


def parse_timestamp_string(ts: str) -> Optional[float]:
    """Parse 'HH:MM:SS.fff' (or 'MM:SS.fff' / plain seconds) into seconds.

    Returns None on unparseable input — callers should treat that as 'unknown'.
    """
    if ts is None:
        return None
    s = str(ts).strip()
    if not s:
        return None
    try:
        parts = s.split(':')
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
        return float(s)
    except (ValueError, TypeError):
        return None


class SpeechContextLookup:
    """Fast 'what label is at this timestamp?' lookup over segmenter output.

    Segments are (original_label, start_seconds, end_seconds) tuples, as
    returned by run_ina_speech_segmenter. The lookup returns the high-level
    mapped label ('speech', 'silence', 'music', 'noise') so callers can use
    it directly without re-applying _LABEL_MAP.
    """

    UNKNOWN = "unknown"

    def __init__(self, segments: Sequence[Tuple[str, float, float]]):
        # Keep segments sorted by start; segmenter output is already monotonic
        # but sort defensively in case a caller hands in unordered data.
        self._segments = sorted(
            ((float(s), float(e), _LABEL_MAP.get(lab, lab))
             for lab, s, e in segments),
            key=lambda row: row[0],
        )
        self._starts = [s for s, _, _ in self._segments]

    def __bool__(self) -> bool:
        return bool(self._segments)

    def label_at(self, timestamp_seconds: Optional[float]) -> str:
        """Return the mapped label covering `timestamp_seconds`.

        If the timestamp falls outside any segment (e.g. past the end of the
        analyzed audio, or before the first segment), returns UNKNOWN.
        """
        if timestamp_seconds is None or not self._segments:
            return self.UNKNOWN
        idx = bisect_right(self._starts, timestamp_seconds) - 1
        if idx < 0:
            return self.UNKNOWN
        start, end, label = self._segments[idx]
        if timestamp_seconds <= end:
            return label
        return self.UNKNOWN

    def label_for_event(
        self, start_seconds: Optional[float], end_seconds: Optional[float]
    ) -> str:
        """Return the label at the midpoint of an event span.

        Audio detection events are typically very brief (single frames or
        sub-second windows) so the midpoint is a fair representative of the
        whole event. If both endpoints are None the result is UNKNOWN.
        """
        if start_seconds is None and end_seconds is None:
            return self.UNKNOWN
        if start_seconds is None:
            mid = end_seconds
        elif end_seconds is None:
            mid = start_seconds
        else:
            mid = (start_seconds + end_seconds) / 2.0
        return self.label_at(mid)

    def breakdown(self) -> Dict[str, float]:
        """Return total seconds covered by each mapped label.

        Useful for context-summarising statistics (e.g. for channel imbalance,
        which has no per-event timestamps).
        """
        out: Dict[str, float] = {}
        for start, end, label in self._segments:
            out[label] = out.get(label, 0.0) + max(0.0, end - start)
        return out

    def total_duration(self) -> float:
        """Total seconds of audio covered by all segments."""
        return sum(self.breakdown().values())


def load_speech_segments_csv(
    report_directory: str,
) -> Optional[List[Tuple[str, float, float]]]:
    """Load `ina_speech_segments.csv` back into the in-memory segment shape.

    Returns the list of (original_label, start_seconds, end_seconds) tuples,
    or None if the CSV is absent or empty (segmenter did not run, or produced
    no segments). Caller should treat None as "no context available".
    """
    path = Path(report_directory) / SEGMENTS_CSV_NAME
    if not path.is_file():
        return None
    segments: List[Tuple[str, float, float]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or header[0] != "label":
            # First-row sentinel ('inaSpeechSegmenter produced no segments') or
            # an unrecognised file shape — treat as no segments.
            return None
        for row in reader:
            if len(row) < 4:
                continue
            _, orig_label, start_str, end_str = row[:4]
            start_sec = parse_timestamp_string(start_str)
            end_sec = parse_timestamp_string(end_str)
            if start_sec is None or end_sec is None:
                continue
            segments.append((orig_label, start_sec, end_sec))
    return segments or None


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
    if check_cancelled and check_cancelled():
        return []

    logger.info(f"inaSpeechSegmenter: analysing audio in {video_path!r}")
    if signals and hasattr(signals, 'ina_segmenter_progress'):
        signals.ina_segmenter_progress.emit(0)

    # pyannote.algorithms.utils.viterbi._update_emission passes a generator
    # expression to np.vstack. NumPy 1.26+ rejects this with a TypeError
    # ('arrays to stack must be passed as a "sequence" type'). Wrap the
    # stack-family functions to materialise generators to a list first.
    import numpy as _np
    _stack_funcs = ('stack', 'vstack', 'hstack', 'dstack', 'column_stack', 'concatenate')
    _originals = {name: getattr(_np, name) for name in _stack_funcs}

    def _make_compat(orig):
        def _compat(arrays, *args, **kwargs):
            if not isinstance(arrays, (list, tuple, _np.ndarray)):
                arrays = list(arrays)
            return orig(arrays, *args, **kwargs)
        return _compat

    for name, orig in _originals.items():
        setattr(_np, name, _make_compat(orig))

    raw_segments = None
    try:
        try:
            from inaSpeechSegmenter import Segmenter as _Segmenter
        except ImportError:
            logger.warning(
                "inaSpeechSegmenter: library not installed — skipping. "
                "Install with: pip install inaSpeechSegmenter"
            )
            return []

        segmenter = _Segmenter()
        segmenter.energy_ratio = silence_ratio
        raw_segments = segmenter(video_path)
    except Exception as exc:
        logger.warning(f"inaSpeechSegmenter: analysis failed: {exc}")
        return []
    finally:
        for name, orig in _originals.items():
            setattr(_np, name, orig)
        if signals and hasattr(signals, 'ina_segmenter_progress'):
            if raw_segments is None:
                signals.ina_segmenter_progress.emit(100)

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
