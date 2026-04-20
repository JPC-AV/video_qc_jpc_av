import json
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any
from unittest.mock import MagicMock

from AV_Spex.checks.ffprobe_check import parse_ffprobe


# Minimal spex-like config object: parse_ffprobe reads spex_config.ffmpeg_values
# as a plain dict ({'video_stream': ..., 'audio_stream': ..., 'format': ...}),
# so we can hand it whatever nested dicts we want without pulling in the real
# SpexConfig dataclass.
class FakeSpex:
    def __init__(self, ffmpeg_values):
        self.ffmpeg_values = ffmpeg_values


def make_ffprobe_json(video_fields=None, audio_fields=None, format_fields=None):
    """Build a realistic ffprobe JSON payload."""
    default_video = {
        "codec_name": "ffv1",
        "codec_long_name": "FFmpeg video codec #1",
        "codec_type": "video",
        "codec_tag_string": "FFV1",
        "codec_tag": "0x31564646",
        "width": 720,
        "height": 486,
        "pix_fmt": "yuv422p10le",
        "color_space": "smpte170m",
        "color_transfer": "bt709",
        "color_primaries": "smpte170m",
        "field_order": "bt",
        "bits_per_raw_sample": "10",
        "display_aspect_ratio": "400:297",
    }
    default_audio = {
        "codec_name": "flac",
        "codec_long_name": "FLAC (Free Lossless Audio Codec)",
        "codec_type": "audio",
        "codec_tag": "0x0000",
        "sample_fmt": "s32",
        "sample_rate": "48000",
        "channels": 2,
        "channel_layout": "stereo",
        "bits_per_raw_sample": "24",
    }
    default_format = {
        "format_name": "matroska,webm",
        "format_long_name": "Matroska / WebM",
        "tags": {"ENCODER_SETTINGS": "something"},
    }

    if video_fields:
        default_video.update(video_fields)
    if audio_fields:
        default_audio.update(audio_fields)
    if format_fields is not None:
        # format_fields may include a sentinel None to clear tags
        for key, val in format_fields.items():
            if val is None:
                default_format.pop(key, None)
            else:
                default_format[key] = val

    return {
        "streams": [default_video, default_audio],
        "format": default_format,
    }


DEFAULT_EXPECTED = {
    "video_stream": {
        "codec_name": "ffv1",
        "codec_type": "video",
        "width": "720",
        "height": "486",
        "pix_fmt": "yuv422p10le",
        "color_space": "smpte170m",
        "bits_per_raw_sample": "10",
    },
    "audio_stream": {
        "codec_name": ["flac", "pcm_s24le"],
        "codec_type": "audio",
        "sample_rate": "48000",
        "channels": "2",
    },
    "format": {
        "format_name": "matroska webm",
        "format_long_name": "Matroska / WebM",
        "tags": {"ENCODER_SETTINGS": None},
    },
}


@pytest.fixture
def patch_config_mgr(monkeypatch):
    """Patch ConfigManager so parse_ffprobe gets a spex with controllable ffmpeg_values."""
    state = {"expected": json.loads(json.dumps(DEFAULT_EXPECTED))}

    def _configure(expected=None):
        if expected is not None:
            state["expected"] = expected

    mock_mgr = MagicMock()

    def fake_get_config(name, _cls):
        if name == "spex":
            return FakeSpex(state["expected"])
        return MagicMock()

    mock_mgr.get_config.side_effect = fake_get_config
    monkeypatch.setattr(
        "AV_Spex.checks.ffprobe_check.ConfigManager", lambda: mock_mgr
    )
    return _configure


def _write_ffprobe_json(tmp_path, payload, name="ffprobe.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


# ---------------------------------------------------------------------------
# format_name normalization (the core gotcha)
# ---------------------------------------------------------------------------

def test_format_name_comma_vs_space_is_considered_equal(tmp_path, patch_config_mgr, setup_logging):
    """FFprobe returns 'matroska,webm'; config has 'matroska webm' — should match."""
    patch_config_mgr()
    payload = make_ffprobe_json()  # format_name = "matroska,webm"
    path = _write_ffprobe_json(tmp_path, payload)

    diffs = parse_ffprobe(path)

    # No format_name difference should be flagged.
    assert "Encoder setting 'format_name'" not in diffs


def test_format_name_mismatch_is_flagged(tmp_path, patch_config_mgr, setup_logging):
    patch_config_mgr()
    payload = make_ffprobe_json(format_fields={"format_name": "mp4"})
    path = _write_ffprobe_json(tmp_path, payload)

    diffs = parse_ffprobe(path)

    assert "Encoder setting 'format_name'" in diffs
    actual, expected = diffs["Encoder setting 'format_name'"]
    assert actual == "mp4"
    assert expected == "matroska webm"


def test_format_long_name_substring_match(tmp_path, patch_config_mgr, setup_logging):
    """format_long_name is a substring check, not exact."""
    patch_config_mgr()
    payload = make_ffprobe_json(format_fields={"format_long_name": "Matroska / WebM (v4)"})
    path = _write_ffprobe_json(tmp_path, payload)

    diffs = parse_ffprobe(path)
    assert "Encoder setting 'format_long_name'" not in diffs


def test_format_long_name_mismatch_flagged(tmp_path, patch_config_mgr, setup_logging):
    patch_config_mgr()
    payload = make_ffprobe_json(format_fields={"format_long_name": "QuickTime"})
    path = _write_ffprobe_json(tmp_path, payload)

    diffs = parse_ffprobe(path)
    assert "Encoder setting 'format_long_name'" in diffs


# ---------------------------------------------------------------------------
# Happy path and mismatches
# ---------------------------------------------------------------------------

def test_matching_file_has_no_differences(tmp_path, patch_config_mgr, setup_logging):
    patch_config_mgr()
    path = _write_ffprobe_json(tmp_path, make_ffprobe_json())

    diffs = parse_ffprobe(path)

    assert diffs == {}


def test_video_stream_mismatch_detected(tmp_path, patch_config_mgr, setup_logging):
    patch_config_mgr()
    payload = make_ffprobe_json(video_fields={"codec_name": "h264", "width": 1920})
    path = _write_ffprobe_json(tmp_path, payload)

    diffs = parse_ffprobe(path)

    assert "codec_name" in diffs
    assert diffs["codec_name"][0] == "h264"
    assert diffs["codec_name"][1] == "ffv1"
    assert "width" in diffs
    assert diffs["width"][0] == "1920"


def test_audio_stream_list_expected_matches_any(tmp_path, patch_config_mgr, setup_logging):
    """Audio codec_name is a list of acceptable values."""
    patch_config_mgr()
    payload = make_ffprobe_json(audio_fields={"codec_name": "pcm_s24le"})
    path = _write_ffprobe_json(tmp_path, payload)

    diffs = parse_ffprobe(path)

    assert "codec_name" not in diffs


def test_audio_stream_list_expected_rejects_other(tmp_path, patch_config_mgr, setup_logging):
    patch_config_mgr()
    payload = make_ffprobe_json(audio_fields={"codec_name": "aac"})
    path = _write_ffprobe_json(tmp_path, payload)

    diffs = parse_ffprobe(path)

    assert "codec_name" in diffs
    assert diffs["codec_name"][0] == "aac"


def test_empty_or_skipped_expected_values_ignored(tmp_path, patch_config_mgr, setup_logging):
    """Fields with empty-string or empty-list expected values must be skipped."""
    expected = json.loads(json.dumps(DEFAULT_EXPECTED))
    expected["video_stream"]["codec_name"] = ""
    expected["video_stream"]["color_space"] = []
    patch_config_mgr(expected)

    # Even if actual differs, an empty expected means "don't check."
    payload = make_ffprobe_json(video_fields={"codec_name": "not_ffv1", "color_space": "bt709"})
    path = _write_ffprobe_json(tmp_path, payload)

    diffs = parse_ffprobe(path)

    assert "codec_name" not in diffs
    assert "color_space" not in diffs


def test_whitespace_stripped_from_actual(tmp_path, patch_config_mgr, setup_logging):
    """Actual values are .strip()'d before comparison."""
    patch_config_mgr()
    payload = make_ffprobe_json(video_fields={"codec_name": " ffv1 "})
    path = _write_ffprobe_json(tmp_path, payload)

    diffs = parse_ffprobe(path)
    assert "codec_name" not in diffs


# ---------------------------------------------------------------------------
# ENCODER_SETTINGS handling
# ---------------------------------------------------------------------------

def test_missing_encoder_settings_flagged(tmp_path, patch_config_mgr, setup_logging):
    patch_config_mgr()
    payload = make_ffprobe_json(format_fields={"tags": {}})
    path = _write_ffprobe_json(tmp_path, payload)

    diffs = parse_ffprobe(path)

    assert "Encoder Settings" in diffs
    assert "No Encoder Settings found" in diffs["Encoder Settings"][0]


def test_missing_tags_entirely_flagged(tmp_path, patch_config_mgr, setup_logging):
    """Format missing the 'tags' key at all should still flag Encoder Settings."""
    patch_config_mgr()
    payload = make_ffprobe_json(format_fields={"tags": None})  # removes the key
    path = _write_ffprobe_json(tmp_path, payload)

    diffs = parse_ffprobe(path)

    assert "Encoder Settings" in diffs


def test_encoder_settings_present_not_flagged(tmp_path, patch_config_mgr, setup_logging):
    patch_config_mgr()
    # Default payload has ENCODER_SETTINGS.
    path = _write_ffprobe_json(tmp_path, make_ffprobe_json())

    diffs = parse_ffprobe(path)

    assert "Encoder Settings" not in diffs


# ---------------------------------------------------------------------------
# Missing file behavior
# ---------------------------------------------------------------------------

def test_missing_file_returns_none(patch_config_mgr, setup_logging):
    patch_config_mgr()
    result = parse_ffprobe("/nonexistent/path/to/ffprobe.json")
    assert result is None
