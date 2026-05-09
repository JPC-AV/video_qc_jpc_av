"""Tests for the speech-context lookup helpers in checks.speech_segmenter_clams.

Covers the post-segmenter consumer surface — `parse_timestamp_string`,
`SpeechContextLookup`, and `load_speech_segments_csv`. The Segmenter itself is
not exercised here (it requires inaSpeechSegmenter / TensorFlow at runtime).
"""

import csv

import pytest

from AV_Spex.checks.speech_segmenter_clams import (
    SEGMENTS_CSV_NAME,
    SpeechContextLookup,
    load_speech_segments_csv,
    parse_timestamp_string,
)


# ---------------------------------------------------------------------------
# parse_timestamp_string
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("00:00:00.000", 0.0),
    ("00:00:01.500", 1.5),
    ("00:01:01.250", 61.25),
    ("01:01:01.001", 3661.001),
    ("00:00:46.0000", 46.0),
])
def test_parse_timestamp_string_hms(text, expected):
    assert parse_timestamp_string(text) == pytest.approx(expected)


@pytest.mark.parametrize("text,expected", [
    ("01:30.5", 90.5),
    ("00:00", 0.0),
])
def test_parse_timestamp_string_ms(text, expected):
    assert parse_timestamp_string(text) == pytest.approx(expected)


@pytest.mark.parametrize("text,expected", [
    ("12.5", 12.5),
    ("0", 0.0),
])
def test_parse_timestamp_string_plain_seconds(text, expected):
    assert parse_timestamp_string(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", None, "not a time", "1:2:3:4"])
def test_parse_timestamp_string_invalid(text):
    assert parse_timestamp_string(text) is None


# ---------------------------------------------------------------------------
# SpeechContextLookup
# ---------------------------------------------------------------------------

def _basic_segments():
    """Three contiguous segments covering 0-30s with mapped labels:
    male→speech (0-10), noEnergy→silence (10-20), music→music (20-30)."""
    return [
        ("male", 0.0, 10.0),
        ("noEnergy", 10.0, 20.0),
        ("music", 20.0, 30.0),
    ]


def test_lookup_empty_is_falsy_and_returns_unknown():
    lookup = SpeechContextLookup([])
    assert not lookup
    assert lookup.label_at(5.0) == SpeechContextLookup.UNKNOWN
    assert lookup.label_for_event(0.0, 1.0) == SpeechContextLookup.UNKNOWN
    assert lookup.breakdown() == {}
    assert lookup.total_duration() == 0.0


def test_lookup_label_at_within_segments():
    lookup = SpeechContextLookup(_basic_segments())
    assert lookup.label_at(0.0) == "speech"      # exact start of first segment
    assert lookup.label_at(5.0) == "speech"      # mid-segment
    assert lookup.label_at(15.0) == "silence"
    assert lookup.label_at(25.0) == "music"


def test_lookup_label_at_boundary_belongs_to_next_segment():
    """At a segment boundary (start of next == end of previous) the timestamp
    is assigned to the segment whose start equals it — bisect_right places the
    value past the equal entry, so the lookup picks the half-open [start, end)
    convention common to interval lookups."""
    lookup = SpeechContextLookup(_basic_segments())
    assert lookup.label_at(10.0) == "silence"     # start of silence segment
    assert lookup.label_at(9.999) == "speech"     # still inside the speech segment


def test_lookup_label_at_outside_segments():
    lookup = SpeechContextLookup(_basic_segments())
    # Past the end of every segment → UNKNOWN
    assert lookup.label_at(100.0) == SpeechContextLookup.UNKNOWN
    # Negative timestamp (before first segment start of 0.0) — bisect_right
    # places -1 before any segment, so idx becomes -1 → UNKNOWN
    assert lookup.label_at(-1.0) == SpeechContextLookup.UNKNOWN


def test_lookup_label_at_with_gap_returns_unknown():
    """If segments don't fully cover the timeline, gaps return UNKNOWN."""
    segments = [
        ("male", 0.0, 5.0),
        ("music", 20.0, 30.0),
    ]
    lookup = SpeechContextLookup(segments)
    assert lookup.label_at(2.0) == "speech"
    # Gap between 5.0 and 20.0
    assert lookup.label_at(10.0) == SpeechContextLookup.UNKNOWN
    assert lookup.label_at(25.0) == "music"


def test_lookup_label_at_none_timestamp():
    lookup = SpeechContextLookup(_basic_segments())
    assert lookup.label_at(None) == SpeechContextLookup.UNKNOWN


def test_lookup_label_for_event_uses_midpoint():
    lookup = SpeechContextLookup(_basic_segments())
    # Event spans 4-6s — midpoint is 5.0 → speech
    assert lookup.label_for_event(4.0, 6.0) == "speech"
    # Event spans 19-21s — midpoint 20.0 falls into music
    assert lookup.label_for_event(19.0, 21.0) == "music"


def test_lookup_label_for_event_with_one_endpoint():
    lookup = SpeechContextLookup(_basic_segments())
    # Only start known
    assert lookup.label_for_event(15.0, None) == "silence"
    # Only end known
    assert lookup.label_for_event(None, 25.0) == "music"


def test_lookup_label_for_event_both_none():
    lookup = SpeechContextLookup(_basic_segments())
    assert lookup.label_for_event(None, None) == SpeechContextLookup.UNKNOWN


def test_lookup_female_maps_to_speech():
    """Both 'male' and 'female' map to the same 'speech' label."""
    lookup = SpeechContextLookup([("female", 0.0, 5.0)])
    assert lookup.label_at(2.0) == "speech"


def test_lookup_unknown_original_label_passes_through():
    """If the segmenter ever returned a label outside _LABEL_MAP, the
    lookup should just hand it back rather than synthesising 'unknown'."""
    lookup = SpeechContextLookup([("weird_label", 0.0, 5.0)])
    assert lookup.label_at(2.0) == "weird_label"


def test_lookup_breakdown_aggregates_by_mapped_label():
    """Both 'male' and 'female' should aggregate under 'speech'."""
    segments = [
        ("male", 0.0, 5.0),         # 5s speech
        ("female", 10.0, 20.0),     # 10s speech
        ("noEnergy", 20.0, 22.5),   # 2.5s silence
    ]
    lookup = SpeechContextLookup(segments)
    breakdown = lookup.breakdown()
    assert breakdown["speech"] == pytest.approx(15.0)
    assert breakdown["silence"] == pytest.approx(2.5)
    assert lookup.total_duration() == pytest.approx(17.5)


def test_lookup_sorts_unsorted_segments_defensively():
    """Caller may hand in segments out of order; lookup should sort them."""
    segments = [
        ("music", 20.0, 30.0),
        ("male", 0.0, 10.0),
        ("noEnergy", 10.0, 20.0),
    ]
    lookup = SpeechContextLookup(segments)
    assert lookup.label_at(5.0) == "speech"
    assert lookup.label_at(15.0) == "silence"
    assert lookup.label_at(25.0) == "music"


# ---------------------------------------------------------------------------
# load_speech_segments_csv
# ---------------------------------------------------------------------------

def _write_segments_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def test_load_segments_csv_missing_file_returns_none(tmp_path):
    assert load_speech_segments_csv(str(tmp_path)) is None


def test_load_segments_csv_sentinel_returns_none(tmp_path):
    """A segmenter that produced no segments writes a single sentinel line.
    The loader should treat that as 'no segments available'."""
    _write_segments_csv(tmp_path / SEGMENTS_CSV_NAME, [
        ["inaSpeechSegmenter produced no segments"],
    ])
    assert load_speech_segments_csv(str(tmp_path)) is None


def test_load_segments_csv_round_trip(tmp_path):
    """A normal CSV is parsed back into the in-memory shape and times come
    out as floats matching the timestamp strings."""
    _write_segments_csv(tmp_path / SEGMENTS_CSV_NAME, [
        ["label", "original_label", "start", "end"],
        ["speech", "male", "00:00:00.000", "00:00:10.000"],
        ["silence", "noEnergy", "00:00:10.000", "00:00:20.500"],
    ])
    out = load_speech_segments_csv(str(tmp_path))
    assert out == [
        ("male", 0.0, 10.0),
        ("noEnergy", 10.0, 20.5),
    ]


def test_load_segments_csv_skips_malformed_rows(tmp_path):
    """Rows with too few fields or unparseable timestamps are silently
    dropped — the loader returns the parseable subset."""
    _write_segments_csv(tmp_path / SEGMENTS_CSV_NAME, [
        ["label", "original_label", "start", "end"],
        ["speech", "male", "00:00:00.000", "00:00:10.000"],
        ["short row"],                                         # too few cols
        ["silence", "noEnergy", "garbage", "00:00:20.000"],   # unparseable start
        ["music", "music", "00:00:30.000", "00:00:40.000"],
    ])
    out = load_speech_segments_csv(str(tmp_path))
    assert out == [
        ("male", 0.0, 10.0),
        ("music", 30.0, 40.0),
    ]


def test_load_segments_csv_unrecognised_header_returns_none(tmp_path):
    """A CSV that doesn't start with the expected header is treated as
    'no segments' rather than crashing."""
    _write_segments_csv(tmp_path / SEGMENTS_CSV_NAME, [
        ["something else entirely"],
        ["speech", "male", "00:00:00.000", "00:00:10.000"],
    ])
    assert load_speech_segments_csv(str(tmp_path)) is None
