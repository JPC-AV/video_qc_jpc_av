"""Tests for utils.generate_report.

Module is large (4993 LOC) and most of it is HTML/CSV scaffolding for the
final report. This file covers the pure-logic helpers and small subprocess
wrappers; the big section-renderers (make_*_html, generate_*_html) are not
directly tested because they're long string-builders that depend on heavy
fixture inputs and are exercised end-to-end in real runs.

Coverage:
* csv_to_html_table — header + body, mismatch styling, fail/pass styling, error fallback
* read_text_file — UTF-8 + latin-1 fallback + missing-file
* prepare_file_section — empty-path branch + delegation
* image_to_data_uri — base64 encoding + missing-file empty fallback
* parse_timestamp — valid + malformed → placeholder tuple
* parse_profile — prefix mapping + default
* find_qct_thumbs — filename parsing into sorted dict
* find_report_csvs — directory scanning into the 15-tuple
* read_xml_file — UTF-8 + latin-1 fallback + missing-file
* _get_video_duration / _get_audio_channel_count — ffprobe wrappers
* _build_waveform_filter — mono / stereo / multi-channel filter graph
* _imbalance_status_colors — characterization → (text, bg, border)
* _parse_bars_durations_csv — qct-parse + CLAMS row schemas
* _parse_tone_detection_csv — present + empty + malformed
* _seconds_to_display — sub-hour + multi-hour
* _extract_frame_at — ffmpeg seek-then-decode command construction
"""

import csv
import os
import subprocess
from base64 import b64decode
from unittest.mock import MagicMock, patch

import pytest

from AV_Spex.utils import generate_report as gr


# ===========================================================================
# Section 1 — csv_to_html_table
# ===========================================================================

def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def test_csv_to_html_table_renders_header_and_body(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, [
        ["A", "B"],
        ["1", "2"],
        ["3", "4"],
    ])
    html = gr.csv_to_html_table(str(csv_path))
    assert "<table>" in html and "</table>" in html
    assert "<th>A</th>" in html and "<th>B</th>" in html
    assert "<td>1</td>" in html and "<td>2</td>" in html


def test_csv_to_html_table_check_fail_styles_pass_and_fail(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, [
        ["Field", "Result"],
        ["x", "pass"],
        ["y", "fail"],
    ])
    html = gr.csv_to_html_table(str(csv_path), check_fail=True)
    assert 'class="cell-match">pass' in html
    assert 'class="cell-mismatch">fail' in html


def test_csv_to_html_table_style_mismatched_marks_diff_columns(tmp_path):
    """When cols 1 and 2 differ, col 2 gets cell-match and col 3 gets cell-mismatch."""
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, [
        ["Field", "Expected", "Actual", "Note"],
        ["fps",   "29.97",   "29.97",  "ok"],   # match → no styling on cols 2/3
        ["fmt",   "FFV1",    "H264",   "diff"], # mismatch → col 2 is match-color, col 3 is mismatch-color
    ])
    html = gr.csv_to_html_table(str(csv_path), style_mismatched=True)
    # The "ok" row should have plain cells
    assert "<td>29.97</td>" in html
    # The mismatched row uses the match/mismatch CSS classes
    assert 'class="cell-match">H264' in html
    assert 'class="cell-mismatch">diff' in html


def test_csv_to_html_table_empty_value_in_actual_skips_mismatch_styling(tmp_path):
    """If row[2] is empty, the row is treated as 'no actual value' and skips the diff styling."""
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, [
        ["Field", "Expected", "Actual", "Note"],
        ["fmt",   "FFV1",    "",       "missing"],
    ])
    html = gr.csv_to_html_table(str(csv_path), style_mismatched=True)
    # Plain <td> for the missing-actual row
    assert 'class="cell-mismatch"' not in html
    assert 'class="cell-match"' not in html


def test_csv_to_html_table_falls_back_to_latin1_on_decode_error(tmp_path):
    """latin-1 fallback should be used when UTF-8 fails."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_bytes(b"Header\n\xe9\n")  # \xe9 is invalid UTF-8 start byte for a single byte
    html = gr.csv_to_html_table(str(csv_path))
    # Header rendered; é-character (latin-1 0xe9) shows up in the body
    assert "<th>Header</th>" in html


def test_csv_to_html_table_missing_file_returns_error_paragraph():
    html = gr.csv_to_html_table("/no/such/file.csv")
    assert "<p>Error processing CSV file:" in html


# ===========================================================================
# Section 2 — read_text_file / prepare_file_section / image_to_data_uri
# ===========================================================================

def test_read_text_file_utf8(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello world")
    assert gr.read_text_file(str(p)) == "hello world"


def test_read_text_file_falls_back_to_latin1(tmp_path):
    """Non-UTF-8 byte should be decoded with latin-1 fallback."""
    p = tmp_path / "x.txt"
    p.write_bytes(b"\xe9 caf\xe9")  # latin-1 for "é café"
    text = gr.read_text_file(str(p))
    assert "é" in text


def test_read_text_file_missing_file_raises(tmp_path):
    """Function only catches UnicodeDecodeError; FileNotFoundError bubbles up."""
    with pytest.raises(FileNotFoundError):
        gr.read_text_file(str(tmp_path / "no_such.txt"))


def test_prepare_file_section_with_path(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("body")
    content, name = gr.prepare_file_section(str(p))
    assert content == "body"
    assert name == "doc.txt"


def test_prepare_file_section_no_path_returns_empty_pair():
    assert gr.prepare_file_section(None) == ("", "")
    assert gr.prepare_file_section("") == ("", "")


def test_prepare_file_section_with_custom_processor(tmp_path):
    """The processor takes precedence over read_text_file."""
    p = tmp_path / "doc.xml"
    p.write_text("<root/>")
    content, name = gr.prepare_file_section(str(p), process_function=lambda _: "processed")
    assert content == "processed"
    assert name == "doc.xml"


def test_image_to_data_uri_encodes_bytes(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"PNGDATA")
    uri = gr.image_to_data_uri(str(img))
    assert uri.startswith("data:image/png;base64,")
    decoded = b64decode(uri.split(",", 1)[1])
    assert decoded == b"PNGDATA"


def test_image_to_data_uri_custom_mime_type(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"JPEG")
    uri = gr.image_to_data_uri(str(img), mime_type="image/jpeg")
    assert uri.startswith("data:image/jpeg;base64,")


def test_image_to_data_uri_missing_file_returns_empty():
    assert gr.image_to_data_uri("/no/such/file.png") == ""


# ===========================================================================
# Section 3 — parse_timestamp / parse_profile
# ===========================================================================

def test_parse_timestamp_valid_format():
    assert gr.parse_timestamp("01:02:03.4500") == (1, 2, 3, 4500)


def test_parse_timestamp_no_milliseconds_defaults_to_zero():
    assert gr.parse_timestamp("00:00:05") == (0, 0, 5, 0)


def test_parse_timestamp_pads_short_milliseconds_to_four_digits():
    """'.5' → 5000 (left-justified to 4 digits) for sortability."""
    assert gr.parse_timestamp("00:00:05.5") == (0, 0, 5, 5000)


def test_parse_timestamp_truncates_long_milliseconds():
    assert gr.parse_timestamp("00:00:05.12345") == (0, 0, 5, 1234)


@pytest.mark.parametrize("bad", ["", None, "5", "1:2", "abc", "00:00:00:abc"])
def test_parse_timestamp_invalid_returns_placeholder_tuple(bad):
    assert gr.parse_timestamp(bad) == (9999, 99, 99, 99, 9999)


@pytest.mark.parametrize("name,expected", [
    ("color_bars_detection.YMAX.940", 0),
    ("color_bars_evaluation.YMIN.28", 1),
    ("threshold_profile.BRNG.0.5", 2),
    ("tag_check.YMAX.940", 3),
    ("anything_else", 99),
    ("", 99),
])
def test_parse_profile_prefix_mapping(name, expected):
    assert gr.parse_profile(name) == expected


# ===========================================================================
# Section 4 — find_qct_thumbs
# ===========================================================================

def test_find_qct_thumbs_returns_empty_dict_when_no_thumbexports(tmp_path):
    assert gr.find_qct_thumbs(str(tmp_path)) == {}


def test_find_qct_thumbs_parses_color_bars_detection_filename(tmp_path):
    thumbs = tmp_path / "ThumbExports"
    thumbs.mkdir()
    (thumbs / "JPC_AV_05000.color_bars_detection.bars_found.first.0.00.00.03.1030.png").write_text("x")
    out = gr.find_qct_thumbs(str(tmp_path))
    assert len(out) == 1
    key = next(iter(out))
    # First-frame-of-color-bars uses the special label
    assert key.startswith("First frame of color bars")
    path, tag, ts = out[key]
    assert path.endswith(".png")
    assert tag == "bars_found"


def test_find_qct_thumbs_parses_failure_filename(tmp_path):
    thumbs = tmp_path / "ThumbExports"
    thumbs.mkdir()
    (thumbs / "JPC_AV_05000.color_bars_evaluation.YMAX.940.0.00.00.53.7870.jpg").write_text("x")
    out = gr.find_qct_thumbs(str(tmp_path))
    assert len(out) == 1
    key = next(iter(out))
    assert "Failed frame" in key
    assert "YMAX" in key
    _, tag, _ = out[key]
    assert tag == "YMAX"


def test_find_qct_thumbs_skips_dotfiles(tmp_path):
    thumbs = tmp_path / "ThumbExports"
    thumbs.mkdir()
    (thumbs / ".DS_Store").write_text("x")
    assert gr.find_qct_thumbs(str(tmp_path)) == {}


def test_find_qct_thumbs_sorts_by_timestamp_when_tag_names_unknown(tmp_path):
    """Sort key is (parse_profile(tag_name), parse_timestamp(timestamp)). Since
    tag_name is YMAX/bars (not a profile name), parse_profile returns 99 for
    each and the secondary timestamp sort takes over."""
    thumbs = tmp_path / "ThumbExports"
    thumbs.mkdir()
    (thumbs / "v.color_bars_evaluation.YMAX.940.0.00.00.50.0000.png").write_text("x")
    (thumbs / "v.color_bars_detection.bars.first.0.00.00.10.0000.png").write_text("x")

    out = gr.find_qct_thumbs(str(tmp_path))
    keys = list(out.keys())
    # Earlier timestamp (00:00:10) sorts first
    assert "00:00:10" in keys[0]
    assert "00:00:50" in keys[1]


# ===========================================================================
# Section 5 — find_report_csvs
# ===========================================================================

def test_find_report_csvs_empty_directory_returns_all_none(tmp_path):
    out = gr.find_report_csvs(str(tmp_path))
    # 15-tuple, all None except qctools_content_check_outputs which is []
    assert isinstance(out, tuple)
    assert len(out) == 15
    none_count = sum(1 for x in out if x is None)
    list_count = sum(1 for x in out if x == [])
    assert none_count == 14
    assert list_count == 1


def test_find_report_csvs_picks_up_known_filenames(tmp_path):
    """Each well-named CSV lands in the right slot in the tuple."""
    files = {
        "qct-parse_colorbars_durations.csv":   "colorbars_duration_output",
        "qct-parse_colorbars_eval_summary.csv": "bars_eval_check_output",
        "qct-parse_colorbars_values.csv":       "colorbars_values_output",
        "qct-parse_profile_summary.csv":        "profile_check_output",
        "qct-parse_profile_failures.csv":       "profile_fails_csv",
        "qct-parse_tags_summary.csv":           "tags_check_output",
        "qct-parse_tags_failures.csv":          "tag_fails_csv",
        "qct-parse_audio_clipping.csv":         "audio_clipping_csv",
        "qct-parse_channel_imbalance.csv":      "channel_imbalance_csv",
        "qct-parse_audible_timecode.csv":       "audible_timecode_csv",
        "qct-parse_audio_dropout.csv":          "audio_dropout_csv",
        "qct-parse_clamped_levels.csv":         "clamped_levels_csv",
        "JPC_AV_metadata_difference.csv":       "difference_csv",
    }
    for name in files:
        (tmp_path / name).write_text("")

    out = gr.find_report_csvs(str(tmp_path))
    # Unpack the 15-tuple in the order the function returns it
    (
        colorbars_duration_output, bars_eval_check_output, colorbars_values_output,
        content_check_outputs, profile_check_output, profile_fails_csv,
        tags_check_output, tag_fails_csv, colorbars_eval_fails_csv,
        audio_clipping_csv, channel_imbalance_csv, audible_timecode_csv,
        audio_dropout_csv, clamped_levels_csv, difference_csv,
    ) = out
    assert colorbars_duration_output.endswith("qct-parse_colorbars_durations.csv")
    assert bars_eval_check_output.endswith("qct-parse_colorbars_eval_summary.csv")
    assert colorbars_values_output.endswith("qct-parse_colorbars_values.csv")
    assert profile_check_output.endswith("qct-parse_profile_summary.csv")
    assert profile_fails_csv.endswith("qct-parse_profile_failures.csv")
    assert tags_check_output.endswith("qct-parse_tags_summary.csv")
    assert tag_fails_csv.endswith("qct-parse_tags_failures.csv")
    assert audio_clipping_csv.endswith("qct-parse_audio_clipping.csv")
    assert channel_imbalance_csv.endswith("qct-parse_channel_imbalance.csv")
    assert audible_timecode_csv.endswith("qct-parse_audible_timecode.csv")
    assert audio_dropout_csv.endswith("qct-parse_audio_dropout.csv")
    assert clamped_levels_csv.endswith("qct-parse_clamped_levels.csv")
    assert difference_csv.endswith("metadata_difference.csv")


def test_find_report_csvs_collects_content_filter_csvs_into_list(tmp_path):
    """contentFilter CSVs accumulate into a list (multiple per video)."""
    (tmp_path / "qct-parse_contentFilter_a.csv").write_text("")
    (tmp_path / "qct-parse_contentFilter_b.csv").write_text("")

    out = gr.find_report_csvs(str(tmp_path))
    content_check_outputs = out[3]
    assert len(content_check_outputs) == 2
    names = sorted(os.path.basename(p) for p in content_check_outputs)
    assert names == ["qct-parse_contentFilter_a.csv", "qct-parse_contentFilter_b.csv"]


def test_find_report_csvs_ignores_unknown_files(tmp_path):
    (tmp_path / ".DS_Store").write_text("")
    (tmp_path / "random_artifact.csv").write_text("")
    out = gr.find_report_csvs(str(tmp_path))
    assert all(x is None or x == [] for x in out)


# ===========================================================================
# Section 6 — read_xml_file
# ===========================================================================

def test_read_xml_file_utf8(tmp_path):
    p = tmp_path / "x.xml"
    p.write_text("<root/>")
    assert gr.read_xml_file(str(p)) == "<root/>"


def test_read_xml_file_falls_back_to_latin1(tmp_path):
    p = tmp_path / "x.xml"
    p.write_bytes(b"<root>\xe9</root>")
    out = gr.read_xml_file(str(p))
    assert "é" in out


def test_read_xml_file_missing_raises():
    """Function only catches UnicodeDecodeError; FileNotFoundError bubbles up."""
    with pytest.raises(FileNotFoundError):
        gr.read_xml_file("/no/such/file.xml")


# ===========================================================================
# Section 7 — ffprobe wrappers
# ===========================================================================

def test_get_video_duration_returns_float(monkeypatch):
    fake = MagicMock(returncode=0, stdout="3661.5\n", stderr="")
    monkeypatch.setattr(gr.subprocess, "run", lambda *a, **kw: fake)
    assert gr._get_video_duration("/v.mkv") == 3661.5


def test_get_video_duration_subprocess_error_returns_none(monkeypatch):
    monkeypatch.setattr(gr.subprocess, "run", MagicMock(side_effect=subprocess.CalledProcessError(1, "ffprobe")))
    assert gr._get_video_duration("/v.mkv") is None


def test_get_audio_channel_count_returns_int(monkeypatch):
    fake = MagicMock(returncode=0, stdout="2\n")
    monkeypatch.setattr(gr.subprocess, "run", lambda *a, **kw: fake)
    assert gr._get_audio_channel_count("/v.mkv") == 2


def test_get_audio_channel_count_failure_returns_none(monkeypatch):
    monkeypatch.setattr(gr.subprocess, "run", MagicMock(side_effect=subprocess.CalledProcessError(1, "ffprobe")))
    assert gr._get_audio_channel_count("/v.mkv") is None


# ===========================================================================
# Section 8 — _build_waveform_filter
# ===========================================================================

def test_build_waveform_filter_mono_no_channelsplit():
    f = gr._build_waveform_filter(num_channels=1, width=1200, height=80)
    assert "channelsplit" not in f
    assert "showwavespic" in f
    assert "1200x80" in f


def test_build_waveform_filter_stereo_uses_layout_stereo():
    f = gr._build_waveform_filter(num_channels=2, width=1200, height=80)
    assert "channelsplit=channel_layout=stereo" in f
    # vstack inputs = 2 waveforms + 1 separator = 3
    assert "vstack=inputs=3" in f


def test_build_waveform_filter_quad_uses_layout_quad():
    f = gr._build_waveform_filter(num_channels=4, width=800, height=60)
    assert "channelsplit=channel_layout=quad" in f
    # 4 channels: 4 waveforms + 3 separators = vstack=inputs=7
    assert "vstack=inputs=7" in f


def test_build_waveform_filter_unknown_count_uses_generic_layout():
    """Channel count not in the standard layout map → 'Nc' generic layout."""
    f = gr._build_waveform_filter(num_channels=5, width=800, height=60)
    assert "channelsplit=channel_layout=5c" in f


# ===========================================================================
# Section 9 — _imbalance_status_colors
# ===========================================================================

@pytest.mark.parametrize("characterization", ["Balanced", "Mono (single channel)"])
def test_imbalance_status_colors_balanced_green(characterization):
    text, bg, border = gr._imbalance_status_colors(characterization)
    # Green-family palette
    assert text == "#155724"
    assert bg == "#d4edda"
    assert border == "#c3e6cb"


@pytest.mark.parametrize("characterization", ["Slight imbalance", "Moderate imbalance"])
def test_imbalance_status_colors_warning_yellow(characterization):
    text, bg, border = gr._imbalance_status_colors(characterization)
    assert text == "#856404"
    assert bg == "#fff3cd"


def test_imbalance_status_colors_default_red_for_unknown():
    """Significant imbalance + anything unknown falls into red palette."""
    for c in ("Significant imbalance", "Unknown", "", "garbage"):
        text, bg, border = gr._imbalance_status_colors(c)
        assert text == "#721c24"
        assert bg == "#f8d7da"


# ===========================================================================
# Section 10 — _parse_bars_durations_csv
# ===========================================================================

def test_parse_bars_durations_qct_parse_two_column_format(tmp_path):
    """Legacy qct-parse single-row format reports pass_label='primary'."""
    csv_path = tmp_path / "bars.csv"
    _write_csv(csv_path, [
        ["qct-parse color bars found:"],
        ["00:00:00.000", "00:00:05.500"],
    ])
    runs = gr._parse_bars_durations_csv(str(csv_path))
    assert runs == [("primary", 0.0, 5.5)]


def test_parse_bars_durations_clams_three_column_format(tmp_path):
    csv_path = tmp_path / "bars.csv"
    _write_csv(csv_path, [
        ["clams bars detection color bars found:"],
        ["primary",    "00:00:00.000", "00:00:04.000"],
        ["second_pass", "00:01:00.000", "00:01:30.000"],
    ])
    runs = gr._parse_bars_durations_csv(str(csv_path))
    assert runs == [
        ("primary", 0.0, 4.0),
        ("second_pass", 60.0, 90.0),
    ]


def test_parse_bars_durations_no_bars_returns_empty(tmp_path):
    csv_path = tmp_path / "bars.csv"
    _write_csv(csv_path, [["clams bars detection found no color bars"]])
    assert gr._parse_bars_durations_csv(str(csv_path)) == []


def test_parse_bars_durations_missing_file_returns_empty():
    assert gr._parse_bars_durations_csv("/no/such/file.csv") == []
    assert gr._parse_bars_durations_csv(None) == []


def test_parse_bars_durations_skips_malformed_rows(tmp_path):
    csv_path = tmp_path / "bars.csv"
    _write_csv(csv_path, [
        ["color bars found"],
        ["malformed-timestamp"],   # 1 col — skipped (need ≥2)
        ["00:00:00.000", "BAD"],   # invalid end time → skipped
        ["primary", "00:00:00.000", "00:00:01.000"],   # valid
    ])
    runs = gr._parse_bars_durations_csv(str(csv_path))
    assert runs == [("primary", 0.0, 1.0)]


# ===========================================================================
# Section 11 — _parse_tone_detection_csv
# ===========================================================================

def test_parse_tone_detection_with_three_column_rows(tmp_path):
    csv_path = tmp_path / "tones.csv"
    _write_csv(csv_path, [
        ["clams tone detection tones found:"],
        ["primary",    "00:00:00.000", "00:00:02.000"],
        ["second_pass", "00:00:10.500", "00:00:15.500"],
    ])
    out = gr._parse_tone_detection_csv(str(csv_path))
    assert out == [
        ("primary", 0.0, 2.0),
        ("second_pass", 10.5, 15.5),
    ]


def test_parse_tone_detection_legacy_two_column_rows(tmp_path):
    csv_path = tmp_path / "tones.csv"
    _write_csv(csv_path, [
        ["tones found:"],
        ["00:00:00.000", "00:00:02.000"],
    ])
    out = gr._parse_tone_detection_csv(str(csv_path))
    assert out == [("primary", 0.0, 2.0)]


def test_parse_tone_detection_no_tones_returns_empty(tmp_path):
    csv_path = tmp_path / "tones.csv"
    _write_csv(csv_path, [["clams tone detection found no tones"]])
    assert gr._parse_tone_detection_csv(str(csv_path)) == []


def test_parse_tone_detection_missing_file_returns_empty():
    assert gr._parse_tone_detection_csv(None) == []
    assert gr._parse_tone_detection_csv("/no/such.csv") == []


# ===========================================================================
# Section 12 — _seconds_to_display
# ===========================================================================

@pytest.mark.parametrize("seconds,expected", [
    (None, "N/A"),
    (0.0, "0:00.0"),
    (5.5, "0:05.5"),
    (65.0, "1:05.0"),
    (3725.5, "1:02:05.5"),  # over 1 hour switches format
])
def test_seconds_to_display(seconds, expected):
    assert gr._seconds_to_display(seconds) == expected


# ===========================================================================
# Section 13 — _extract_frame_at
# ===========================================================================

def test_extract_frame_at_seeks_before_input_for_keyframe_seek(monkeypatch):
    """-ss MUST appear before -i to trigger fast keyframe seek."""
    run_mock = MagicMock(returncode=0)
    monkeypatch.setattr(gr.subprocess, "run", run_mock)

    gr._extract_frame_at("/v.mkv", timestamp=12.5, output_path="/out/frame.jpg", height=80)

    cmd = run_mock.call_args[0][0]
    ss_idx = cmd.index("-ss")
    i_idx = cmd.index("-i")
    assert ss_idx < i_idx, "-ss must appear before -i for fast keyframe seek"

    # Single frame, scaled to height
    assert "-frames:v" in cmd and cmd[cmd.index("-frames:v") + 1] == "1"
    vf_idx = cmd.index("-vf")
    assert cmd[vf_idx + 1] == "scale=-1:80"
    # Timestamp
    assert cmd[ss_idx + 1] == "12.5"


def test_extract_frame_at_propagates_subprocess_error(monkeypatch):
    """check=True means CalledProcessError bubbles up."""
    monkeypatch.setattr(
        gr.subprocess, "run",
        MagicMock(side_effect=subprocess.CalledProcessError(1, "ffmpeg")),
    )
    with pytest.raises(subprocess.CalledProcessError):
        gr._extract_frame_at("/v.mkv", 0.0, "/out.jpg", 80)
