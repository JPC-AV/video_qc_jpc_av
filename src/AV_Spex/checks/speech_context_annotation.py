"""
Annotate qct-parse audio analysis CSVs with speech-context labels from
inaSpeechSegmenter output.

Runs after the segmenter completes. Reads each existing audio CSV in the
report directory, joins per-event timestamps against the segmenter timeline,
and rewrites the CSV with an extra "Speech context" column (per-event) or a
"Context breakdown" summary block (where there are no per-event rows).

Annotation is best-effort: if the segmenter CSV is missing or a CSV doesn't
exist yet (because the corresponding qct-parse step was disabled), the
annotator skips it silently.
"""

import csv
import os
from typing import List, Optional, Sequence, Tuple

from AV_Spex.utils.log_setup import logger
from AV_Spex.checks.speech_segmenter_clams import (
    SpeechContextLookup,
    load_speech_segments_csv,
    parse_timestamp_string,
)

DROPOUT_CSV_NAME = "qct-parse_audio_dropout.csv"
CLIPPING_CSV_NAME = "qct-parse_audio_clipping.csv"
IMBALANCE_CSV_NAME = "qct-parse_channel_imbalance.csv"

CONTEXT_COL = "Speech context"


def _read_all_rows(path: str) -> List[List[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


def _write_all_rows(path: str, rows: Sequence[Sequence[str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def _split_summary_and_table(
    rows: List[List[str]], data_header_first_cell: str
) -> Tuple[List[List[str]], Optional[int], List[List[str]]]:
    """Find the per-event data table inside a qct-parse audio CSV.

    Each audio CSV starts with summary key/value rows, then a blank line, then
    a header row beginning with `data_header_first_cell` (e.g. "Timestamp Start"),
    followed by event rows. Returns (summary_rows, header_index, table_rows).
    If no table is present, header_index is None and table_rows is [].
    """
    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0] == data_header_first_cell:
            header_idx = i
            break
    if header_idx is None:
        return list(rows), None, []
    summary = rows[:header_idx]
    table = rows[header_idx:]
    return summary, header_idx, table


def _context_summary_rows(
    contexts: Sequence[str], lookup: SpeechContextLookup
) -> List[List[str]]:
    """Build a "Context breakdown" block summarising per-event labels."""
    counts: dict = {}
    for c in contexts:
        counts[c] = counts.get(c, 0) + 1
    total = len(contexts)
    out = [
        [],
        ["Speech Context Breakdown"],
        ["Total events", total],
    ]
    # Sort labels by count desc for readability, with 'unknown' last.
    def _sort_key(item):
        label, count = item
        return (label == SpeechContextLookup.UNKNOWN, -count, label)
    for label, count in sorted(counts.items(), key=_sort_key):
        pct = (count / total * 100.0) if total else 0.0
        out.append([f"  {label}", f"{count} ({pct:.1f}%)"])
    return out


def annotate_audio_dropout_csv(
    report_directory: str, lookup: SpeechContextLookup
) -> bool:
    """Add a Speech context column + summary block to qct-parse_audio_dropout.csv.

    Returns True if the file was rewritten, False otherwise.
    """
    path = os.path.join(report_directory, DROPOUT_CSV_NAME)
    if not os.path.isfile(path):
        return False
    rows = _read_all_rows(path)
    summary, header_idx, table = _split_summary_and_table(rows, "Timestamp Start")
    if header_idx is None or len(table) < 2:
        return False  # no events table present

    header = list(table[0])
    if CONTEXT_COL in header:
        # Already annotated — re-running on the same CSV; skip to avoid
        # appending duplicate columns/blocks.
        return False
    header.append(CONTEXT_COL)

    new_table: List[List[str]] = [header]
    contexts: List[str] = []
    for row in table[1:]:
        start_sec = parse_timestamp_string(row[0]) if len(row) > 0 else None
        end_sec = parse_timestamp_string(row[1]) if len(row) > 1 else None
        ctx = lookup.label_for_event(start_sec, end_sec)
        contexts.append(ctx)
        new_table.append(list(row) + [ctx])

    out_rows = list(summary) + new_table + _context_summary_rows(contexts, lookup)
    _write_all_rows(path, out_rows)
    logger.debug(f"Annotated {DROPOUT_CSV_NAME} with speech context for {len(contexts)} event(s)")
    return True


def annotate_audio_clipping_csv(
    report_directory: str, lookup: SpeechContextLookup
) -> bool:
    """Add a Speech context column + summary block to qct-parse_audio_clipping.csv.

    Returns True if the file was rewritten, False otherwise.
    """
    path = os.path.join(report_directory, CLIPPING_CSV_NAME)
    if not os.path.isfile(path):
        return False
    rows = _read_all_rows(path)
    summary, header_idx, table = _split_summary_and_table(rows, "Timestamp")
    if header_idx is None or len(table) < 2:
        return False

    header = list(table[0])
    if CONTEXT_COL in header:
        return False
    header.append(CONTEXT_COL)

    new_table: List[List[str]] = [header]
    contexts: List[str] = []
    for row in table[1:]:
        ts_sec = parse_timestamp_string(row[0]) if len(row) > 0 else None
        ctx = lookup.label_at(ts_sec)
        contexts.append(ctx)
        new_table.append(list(row) + [ctx])

    out_rows = list(summary) + new_table + _context_summary_rows(contexts, lookup)
    _write_all_rows(path, out_rows)
    logger.debug(f"Annotated {CLIPPING_CSV_NAME} with speech context for {len(contexts)} event(s)")
    return True


def annotate_channel_imbalance_csv(
    report_directory: str, lookup: SpeechContextLookup
) -> bool:
    """Append a Context breakdown block to qct-parse_channel_imbalance.csv.

    Channel imbalance has no per-event rows, so we append an overall
    breakdown of how the analyzed audio splits across speech/silence/music/noise
    by total duration. Returns True if the file was rewritten.
    """
    path = os.path.join(report_directory, IMBALANCE_CSV_NAME)
    if not os.path.isfile(path):
        return False
    rows = _read_all_rows(path)
    # Detect prior annotation.
    for row in rows:
        if row and row[0] == "Speech Context Breakdown (by duration)":
            return False

    breakdown = lookup.breakdown()
    total_seconds = sum(breakdown.values())
    if total_seconds <= 0:
        return False

    extra: List[List[str]] = [
        [],
        ["Speech Context Breakdown (by duration)"],
        ["Total analyzed seconds", f"{total_seconds:.1f}"],
    ]
    def _sort_key(item):
        label, secs = item
        return (label == SpeechContextLookup.UNKNOWN, -secs, label)
    for label, secs in sorted(breakdown.items(), key=_sort_key):
        pct = secs / total_seconds * 100.0
        extra.append([f"  {label}", f"{secs:.1f}s ({pct:.1f}%)"])

    _write_all_rows(path, list(rows) + extra)
    logger.debug(
        f"Appended speech context breakdown to {IMBALANCE_CSV_NAME} "
        f"({total_seconds:.1f}s analyzed)"
    )
    return True


def annotate_audio_csvs(
    report_directory: str,
    segments: Optional[Sequence[Tuple[str, float, float]]] = None,
) -> dict:
    """Annotate all audio analysis CSVs with speech context.

    `segments` is the in-memory segmenter output (preferred). When omitted,
    the segmenter CSV is loaded from `report_directory` instead. If neither
    is available, no annotations are performed.

    Returns a dict of {csv_name: bool} indicating which files were rewritten.
    """
    if segments is None:
        segments = load_speech_segments_csv(report_directory)
    if not segments:
        logger.debug(
            "Speech context annotation skipped: no segmenter output available "
            f"in {report_directory}"
        )
        return {}

    lookup = SpeechContextLookup(segments)
    results = {
        DROPOUT_CSV_NAME: annotate_audio_dropout_csv(report_directory, lookup),
        CLIPPING_CSV_NAME: annotate_audio_clipping_csv(report_directory, lookup),
        IMBALANCE_CSV_NAME: annotate_channel_imbalance_csv(report_directory, lookup),
    }
    rewritten = [name for name, was_written in results.items() if was_written]
    if rewritten:
        logger.info(f"Speech context annotation: updated {len(rewritten)} CSV(s) — {', '.join(rewritten)}")
    return results
