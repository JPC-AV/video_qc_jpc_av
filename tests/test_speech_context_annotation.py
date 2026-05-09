"""Tests for checks.speech_context_annotation.

Each annotator rewrites a qct-parse audio CSV in place to add a Speech
context column (per-event annotators) or a Speech Context Breakdown block
(channel imbalance). The tests construct minimal CSVs in tmp dirs to avoid
depending on real qct-parse output.

Coverage:
* annotate_audio_dropout_csv — events get the column, summary block is added,
  re-running is a no-op, missing files / empty tables are handled.
* annotate_audio_clipping_csv — same shape.
* annotate_channel_imbalance_csv — breakdown block is appended; idempotent.
* annotate_audio_csvs orchestrator — uses passed segments preferentially,
  falls back to loading the segmenter CSV from disk, and returns an empty
  dict when neither is available.
"""

import csv

import pytest

from AV_Spex.checks.speech_segmenter_clams import SEGMENTS_CSV_NAME
from AV_Spex.checks.speech_context_annotation import (
    CLIPPING_CSV_NAME,
    DROPOUT_CSV_NAME,
    IMBALANCE_CSV_NAME,
    annotate_audio_clipping_csv,
    annotate_audio_csvs,
    annotate_audio_dropout_csv,
    annotate_channel_imbalance_csv,
)
from AV_Spex.checks.speech_segmenter_clams import SpeechContextLookup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Three contiguous segments covering 0-30s. Used by all per-event tests so
# the speech context for any event in [0,10) is 'speech', [10,20) is 'silence',
# [20,30) is 'music'.
SEGMENTS = [
    ("male", 0.0, 10.0),
    ("noEnergy", 10.0, 20.0),
    ("music", 20.0, 30.0),
]


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def _write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def _write_segments_csv(report_dir, segments):
    """Write an ina_speech_segments.csv that load_speech_segments_csv accepts.

    Mirrors write_segments_csv's output shape so the orchestrator's
    "load from disk when no segments passed" path can be exercised.
    """
    from AV_Spex.checks.speech_segmenter_clams import _LABEL_MAP, _format_timestamp

    rows = [["label", "original_label", "start", "end"]]
    for orig, start, end in segments:
        rows.append([
            _LABEL_MAP.get(orig, orig),
            orig,
            _format_timestamp(start),
            _format_timestamp(end),
        ])
    _write_rows(report_dir / SEGMENTS_CSV_NAME, rows)


def _make_dropout_csv(path, events):
    """Build a minimal qct-parse_audio_dropout.csv with the given event rows.

    Each event is (start_str, end_str, channel). The header summary fields are
    placeholders — the annotator only cares about the per-event table.
    """
    rows = [
        ["Audio Dropout Detection Results"],
        ["Total Audio Frames", "1000"],
        ["Dropout Events Detected", str(len(events))],
        ["Dropout Detected", "Yes" if events else "No"],
        [],
    ]
    if events:
        rows.append([
            "Timestamp Start", "Timestamp End", "Channel",
            "Worst RMS (dBFS)", "Median RMS (dBFS)", "Drop (dB)",
            "Confidence", "Corroborating Metrics",
        ])
        for start, end, ch in events:
            rows.append([start, end, str(ch), "-70", "-15", "55", "high", "x"])
    _write_rows(path, rows)


def _make_clipping_csv(path, events):
    """Build a minimal qct-parse_audio_clipping.csv. Each event is (ts_str,)."""
    rows = [
        ["Audio Clipping Detection Results"],
        ["Total Audio Frames", "1000"],
        ["Clipped Frames", str(len(events))],
        ["Clipping Detected", "Yes" if events else "No"],
        [],
    ]
    if events:
        rows.append(["Timestamp", "Peak Level (dBFS)", "Flat Factor"])
        for (ts,) in events:
            rows.append([ts, "-0.4", "3"])
    _write_rows(path, rows)


def _make_imbalance_csv(path):
    rows = [
        ["Channel Imbalance Analysis Results"],
        ["Total Audio Frames", "1000"],
        ["Number of Channels", "2"],
        ["Channel 1 Mean RMS (dBFS)", "-43.2"],
        ["Channel 2 Mean RMS (dBFS)", "-43.0"],
        [],
        ["Mean Difference (dB)", "-0.2"],
        ["Overall Characterization", "Balanced"],
    ]
    _write_rows(path, rows)


# ---------------------------------------------------------------------------
# annotate_audio_dropout_csv
# ---------------------------------------------------------------------------

def test_annotate_dropout_missing_file_returns_false(tmp_path):
    lookup = SpeechContextLookup(SEGMENTS)
    assert annotate_audio_dropout_csv(str(tmp_path), lookup) is False


def test_annotate_dropout_no_events_returns_false(tmp_path):
    """When qct-parse found no dropouts the CSV has no per-event table; the
    annotator should leave the file untouched (no summary block to add either)."""
    path = tmp_path / DROPOUT_CSV_NAME
    _make_dropout_csv(path, [])
    lookup = SpeechContextLookup(SEGMENTS)

    assert annotate_audio_dropout_csv(str(tmp_path), lookup) is False
    # File unchanged — no Speech context anywhere.
    text = path.read_text()
    assert "Speech context" not in text


def test_annotate_dropout_adds_column_and_summary(tmp_path):
    path = tmp_path / DROPOUT_CSV_NAME
    _make_dropout_csv(path, [
        ("00:00:05.000", "00:00:05.500", 1),   # midpoint 5.25 → speech
        ("00:00:14.000", "00:00:15.000", 1),   # midpoint 14.5 → silence
        ("00:00:25.000", "00:00:25.500", 1),   # midpoint 25.25 → music
    ])
    lookup = SpeechContextLookup(SEGMENTS)

    assert annotate_audio_dropout_csv(str(tmp_path), lookup) is True

    rows = _read_rows(path)
    # Locate the data header — it now ends with the new column.
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "Timestamp Start")
    assert rows[header_idx][-1] == "Speech context"

    contexts = [rows[header_idx + i + 1][-1] for i in range(3)]
    assert contexts == ["speech", "silence", "music"]

    # Summary block appears after the table.
    assert ["Speech Context Breakdown"] in rows
    breakdown_start = rows.index(["Speech Context Breakdown"])
    summary_text = "\n".join(",".join(r) for r in rows[breakdown_start:])
    assert "Total events,3" in summary_text
    assert "speech,1 (33.3%)" in summary_text
    assert "silence,1 (33.3%)" in summary_text
    assert "music,1 (33.3%)" in summary_text


def test_annotate_dropout_event_outside_segments_is_unknown(tmp_path):
    """An event past the end of all segments gets the UNKNOWN label rather
    than crashing or being assigned to the last segment."""
    path = tmp_path / DROPOUT_CSV_NAME
    _make_dropout_csv(path, [("00:01:40.000", "00:01:40.500", 1)])  # 100s, past 30s end
    lookup = SpeechContextLookup(SEGMENTS)

    assert annotate_audio_dropout_csv(str(tmp_path), lookup) is True

    rows = _read_rows(path)
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "Timestamp Start")
    assert rows[header_idx + 1][-1] == SpeechContextLookup.UNKNOWN


def test_annotate_dropout_is_idempotent(tmp_path):
    """Re-running the annotator on an already-annotated CSV must not append
    a second column or a second summary block."""
    path = tmp_path / DROPOUT_CSV_NAME
    _make_dropout_csv(path, [("00:00:05.000", "00:00:05.500", 1)])
    lookup = SpeechContextLookup(SEGMENTS)

    assert annotate_audio_dropout_csv(str(tmp_path), lookup) is True
    first_text = path.read_text()
    # Second call is a no-op.
    assert annotate_audio_dropout_csv(str(tmp_path), lookup) is False
    assert path.read_text() == first_text
    # Sanity: only one Speech context column header in the file.
    assert first_text.count("Speech context") == 1


# ---------------------------------------------------------------------------
# annotate_audio_clipping_csv
# ---------------------------------------------------------------------------

def test_annotate_clipping_missing_file_returns_false(tmp_path):
    lookup = SpeechContextLookup(SEGMENTS)
    assert annotate_audio_clipping_csv(str(tmp_path), lookup) is False


def test_annotate_clipping_no_events_returns_false(tmp_path):
    path = tmp_path / CLIPPING_CSV_NAME
    _make_clipping_csv(path, [])
    lookup = SpeechContextLookup(SEGMENTS)
    assert annotate_audio_clipping_csv(str(tmp_path), lookup) is False
    assert "Speech context" not in path.read_text()


def test_annotate_clipping_adds_column(tmp_path):
    path = tmp_path / CLIPPING_CSV_NAME
    _make_clipping_csv(path, [
        ("00:00:05.000",),   # speech
        ("00:00:25.500",),   # music
    ])
    lookup = SpeechContextLookup(SEGMENTS)

    assert annotate_audio_clipping_csv(str(tmp_path), lookup) is True

    rows = _read_rows(path)
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "Timestamp")
    assert rows[header_idx][-1] == "Speech context"
    contexts = [rows[header_idx + 1][-1], rows[header_idx + 2][-1]]
    assert contexts == ["speech", "music"]


def test_annotate_clipping_is_idempotent(tmp_path):
    path = tmp_path / CLIPPING_CSV_NAME
    _make_clipping_csv(path, [("00:00:05.000",)])
    lookup = SpeechContextLookup(SEGMENTS)

    assert annotate_audio_clipping_csv(str(tmp_path), lookup) is True
    first = path.read_text()
    assert annotate_audio_clipping_csv(str(tmp_path), lookup) is False
    assert path.read_text() == first


# ---------------------------------------------------------------------------
# annotate_channel_imbalance_csv
# ---------------------------------------------------------------------------

def test_annotate_imbalance_missing_file_returns_false(tmp_path):
    lookup = SpeechContextLookup(SEGMENTS)
    assert annotate_channel_imbalance_csv(str(tmp_path), lookup) is False


def test_annotate_imbalance_appends_breakdown_block(tmp_path):
    path = tmp_path / IMBALANCE_CSV_NAME
    _make_imbalance_csv(path)
    lookup = SpeechContextLookup(SEGMENTS)  # 30s total, 10s each label

    assert annotate_channel_imbalance_csv(str(tmp_path), lookup) is True

    rows = _read_rows(path)
    # Original content preserved at the top.
    assert rows[0] == ["Channel Imbalance Analysis Results"]
    # New block appended at the end.
    assert ["Speech Context Breakdown (by duration)"] in rows
    text = "\n".join(",".join(r) for r in rows)
    assert "Total analyzed seconds,30.0" in text
    assert "speech,10.0s (33.3%)" in text
    assert "silence,10.0s (33.3%)" in text
    assert "music,10.0s (33.3%)" in text


def test_annotate_imbalance_is_idempotent(tmp_path):
    path = tmp_path / IMBALANCE_CSV_NAME
    _make_imbalance_csv(path)
    lookup = SpeechContextLookup(SEGMENTS)

    assert annotate_channel_imbalance_csv(str(tmp_path), lookup) is True
    first = path.read_text()
    assert annotate_channel_imbalance_csv(str(tmp_path), lookup) is False
    assert path.read_text() == first


def test_annotate_imbalance_zero_segment_duration_returns_false(tmp_path):
    """Empty lookup → no analyzed seconds → nothing meaningful to append."""
    path = tmp_path / IMBALANCE_CSV_NAME
    _make_imbalance_csv(path)
    empty_lookup = SpeechContextLookup([])

    assert annotate_channel_imbalance_csv(str(tmp_path), empty_lookup) is False
    assert "Speech Context Breakdown" not in path.read_text()


# ---------------------------------------------------------------------------
# annotate_audio_csvs (orchestrator)
# ---------------------------------------------------------------------------

def test_orchestrator_skips_when_no_segments_or_csv(tmp_path):
    """No segments passed and no segmenter CSV on disk → empty results dict."""
    # Even if audio CSVs exist, annotation can't run without context.
    _make_dropout_csv(tmp_path / DROPOUT_CSV_NAME, [("00:00:05.000", "00:00:05.500", 1)])
    assert annotate_audio_csvs(str(tmp_path)) == {}


def test_orchestrator_uses_passed_segments(tmp_path):
    _make_dropout_csv(tmp_path / DROPOUT_CSV_NAME, [
        ("00:00:05.000", "00:00:05.500", 1),
    ])
    _make_imbalance_csv(tmp_path / IMBALANCE_CSV_NAME)

    results = annotate_audio_csvs(str(tmp_path), segments=SEGMENTS)
    assert results == {
        DROPOUT_CSV_NAME: True,
        CLIPPING_CSV_NAME: False,   # no file present
        IMBALANCE_CSV_NAME: True,
    }
    # Both files were rewritten.
    assert "Speech context" in (tmp_path / DROPOUT_CSV_NAME).read_text()
    assert "Speech Context Breakdown (by duration)" in (tmp_path / IMBALANCE_CSV_NAME).read_text()


def test_orchestrator_loads_segmenter_csv_from_disk(tmp_path):
    """When `segments` is omitted the orchestrator should read the segmenter
    CSV alongside the audio CSVs."""
    _make_dropout_csv(tmp_path / DROPOUT_CSV_NAME, [
        ("00:00:14.000", "00:00:15.000", 1),  # silence segment
    ])
    _write_segments_csv(tmp_path, SEGMENTS)

    results = annotate_audio_csvs(str(tmp_path))   # no segments arg
    assert results[DROPOUT_CSV_NAME] is True

    rows = _read_rows(tmp_path / DROPOUT_CSV_NAME)
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "Timestamp Start")
    assert rows[header_idx + 1][-1] == "silence"


def test_orchestrator_skips_when_segments_empty(tmp_path):
    """Empty segments list (e.g. segmenter ran but returned nothing) — same
    as no segments at all."""
    _make_dropout_csv(tmp_path / DROPOUT_CSV_NAME, [
        ("00:00:05.000", "00:00:05.500", 1),
    ])
    assert annotate_audio_csvs(str(tmp_path), segments=[]) == {}
    assert "Speech context" not in (tmp_path / DROPOUT_CSV_NAME).read_text()
