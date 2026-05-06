"""Tests for ffprobe_import — parse FFprobe JSON and build FfprobeProfile."""

import json

import pytest

from AV_Spex.utils import ffprobe_import as fi
from AV_Spex.utils.config_setup import (
    FfprobeProfile,
    FFmpegVideoStream,
    FFmpegAudioStream,
    FFmpegFormat,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _default_video_stream():
    return {
        "codec_name": "ffv1",
        "codec_long_name": "FFmpeg video codec #1",
        "codec_type": "video",
        "codec_tag_string": "FFV1",
        "codec_tag": "0x31564646",
        "width": 720,
        "height": 486,
        "display_aspect_ratio": "400:297",
        "pix_fmt": "yuv422p10le",
        "color_space": "smpte170m",
        "color_transfer": "bt709",
        "color_primaries": "smpte170m",
        "field_order": "bt",
        "bits_per_raw_sample": "10",
    }


def _default_audio_stream():
    return {
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


def _default_format():
    return {
        "format_name": "matroska,webm",
        "format_long_name": "Matroska / WebM",
        "tags": {
            "TITLE": "JPC_AV_00001",
            "ENCODER_SETTINGS": "Source_VTR: SVO-5800",
            "creation_time": "2026-04-19T00:00:00.000000Z",
        },
    }


def _make_ffprobe_payload(
    video=None, audio_streams=None, fmt=None, include_video=True
):
    streams = []
    if include_video:
        streams.append(video if video is not None else _default_video_stream())
    if audio_streams is None:
        streams.append(_default_audio_stream())
    else:
        for audio in audio_streams:
            streams.append(audio)
    return {
        "streams": streams,
        "format": fmt if fmt is not None else _default_format(),
    }


def _write_json(tmp_path, payload, name="ffprobe.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


# ---------------------------------------------------------------------------
# parse_ffprobe_json_file
# ---------------------------------------------------------------------------

def test_parse_ffprobe_json_file_happy_path(tmp_path):
    path = _write_json(tmp_path, _make_ffprobe_payload())
    data = fi.parse_ffprobe_json_file(path)
    assert data is not None
    assert data["video_stream"]["codec_name"] == "ffv1"
    assert data["audio_stream"]["codec_name"] == "flac"
    assert data["format"]["format_name"] == "matroska,webm"


def test_parse_ffprobe_json_file_missing_file():
    assert fi.parse_ffprobe_json_file("/nonexistent/ffprobe.json") is None


def test_parse_ffprobe_json_file_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not really json")
    assert fi.parse_ffprobe_json_file(str(path)) is None


def test_parse_ffprobe_json_file_no_streams_key(tmp_path):
    path = _write_json(tmp_path, {"format": _default_format()})
    assert fi.parse_ffprobe_json_file(str(path)) is None


def test_parse_ffprobe_json_file_merges_multiple_audio_streams(tmp_path):
    audio1 = _default_audio_stream()
    audio2 = _default_audio_stream()
    audio2["codec_name"] = "pcm_s24le"
    audio2["codec_long_name"] = "PCM signed 24-bit LE"
    payload = _make_ffprobe_payload(audio_streams=[audio1, audio2])
    path = _write_json(tmp_path, payload)

    data = fi.parse_ffprobe_json_file(path)
    assert data is not None
    # First audio populates; second audio's codec_name should be merged into a list.
    audio = data["audio_stream"]
    assert isinstance(audio["codec_name"], list)
    assert "flac" in audio["codec_name"]
    assert "pcm_s24le" in audio["codec_name"]
    assert isinstance(audio["codec_long_name"], list)


def test_parse_ffprobe_json_file_latin1_fallback(tmp_path):
    payload = _make_ffprobe_payload()
    raw = json.dumps(payload).encode("utf-8")
    # Append a latin-1 byte at the beginning of a comment-free JSON: make UTF-8
    # decode fail by prepending a lone 0xe9 byte inside a string key.
    # Simpler: build JSON with a latin-1 byte inside a title value.
    payload["format"]["tags"]["TITLE"] = "JPC_AV_x"
    raw = json.dumps(payload).encode("utf-8")
    # Inject a 0xff byte at a neutral spot — inside an already-existing string field.
    # JSON needs to remain parseable after latin-1 decode, so splice a valid latin-1
    # char into the TITLE value.
    raw = raw.replace(b'"JPC_AV_x"', b'"JPC_AV_\xe9"')
    path = tmp_path / "latin1.json"
    path.write_bytes(raw)

    data = fi.parse_ffprobe_json_file(str(path))
    assert data is not None
    # Title should be decoded as latin-1 "é"
    assert "JPC_AV_" in data["format"]["tags"]["TITLE"]


# ---------------------------------------------------------------------------
# extract_* field extractors
# ---------------------------------------------------------------------------

def test_extract_video_stream_fields_picks_known_keys():
    stream = _default_video_stream()
    stream["unrelated_key"] = "ignored"
    extracted = fi.extract_video_stream_fields(stream)
    assert extracted["codec_name"] == "ffv1"
    assert "unrelated_key" not in extracted


def test_extract_audio_stream_fields_wraps_codec_name_in_list():
    audio = _default_audio_stream()
    # codec_name arrives as a string from single-stream ffprobe.
    extracted = fi.extract_audio_stream_fields(audio)
    assert isinstance(extracted["codec_name"], list)
    assert extracted["codec_name"] == ["flac"]
    assert isinstance(extracted["codec_long_name"], list)


def test_extract_audio_stream_fields_keeps_existing_list():
    audio = _default_audio_stream()
    audio["codec_name"] = ["flac", "pcm_s24le"]
    extracted = fi.extract_audio_stream_fields(audio)
    assert extracted["codec_name"] == ["flac", "pcm_s24le"]


def test_extract_format_fields_tags_subset():
    fmt = _default_format()
    extracted = fi.extract_format_fields(fmt)
    # Top-level format fields.
    assert extracted["format_name"] == "matroska,webm"
    # tags: known keys preserved, missing ones filled with None.
    assert extracted["tags"]["TITLE"] == "JPC_AV_00001"
    assert extracted["tags"]["ENCODER_SETTINGS"] == "Source_VTR: SVO-5800"
    assert extracted["tags"]["ENCODER"] is None
    # Unknown tag keys should not leak in.
    fmt["tags"]["UNKNOWN_TAG"] = "x"
    extracted2 = fi.extract_format_fields(fmt)
    assert "UNKNOWN_TAG" not in extracted2["tags"]


def test_extract_format_fields_no_tags_key():
    fmt = {"format_name": "matroska", "format_long_name": "Matroska"}
    extracted = fi.extract_format_fields(fmt)
    assert "tags" not in extracted
    assert extracted["format_name"] == "matroska"


# ---------------------------------------------------------------------------
# _apply_defaults
# ---------------------------------------------------------------------------

def test_apply_defaults_fills_missing_str_with_empty_string():
    fields = {"codec_name": "ffv1"}
    result = fi._apply_defaults(fields, FFmpegVideoStream)
    assert result["codec_name"] == "ffv1"
    assert result["width"] == ""
    assert result["codec_tag_string"] == ""


def test_apply_defaults_fills_missing_list_with_empty_list():
    fields = {}
    result = fi._apply_defaults(fields, FFmpegAudioStream)
    assert result["codec_name"] == []
    assert result["codec_long_name"] == []
    assert result["channels"] == ""


def test_apply_defaults_fills_missing_dict_with_empty_dict():
    fields = {"format_name": "matroska"}
    result = fi._apply_defaults(fields, FFmpegFormat)
    assert result["tags"] == {}
    assert result["format_long_name"] == ""


# ---------------------------------------------------------------------------
# import_ffprobe_file_to_profile
# ---------------------------------------------------------------------------

def test_import_ffprobe_file_to_profile_happy_path(tmp_path):
    path = _write_json(tmp_path, _make_ffprobe_payload())
    profile = fi.import_ffprobe_file_to_profile(path)
    assert profile is not None
    assert isinstance(profile, FfprobeProfile)
    assert profile.video_stream.codec_name == "ffv1"
    assert profile.audio_stream.codec_name == ["flac"]
    assert profile.format.format_name == "matroska,webm"
    assert profile.format.tags["TITLE"] == "JPC_AV_00001"


def test_import_ffprobe_file_to_profile_missing_file_returns_none():
    assert fi.import_ffprobe_file_to_profile("/nonexistent.json") is None


def test_import_ffprobe_file_to_profile_video_only_gets_audio_defaults(tmp_path):
    """Video-only file: audio fields default to empty list/str, no crash."""
    payload = _make_ffprobe_payload(audio_streams=[])  # no audio
    path = _write_json(tmp_path, payload)
    profile = fi.import_ffprobe_file_to_profile(path)
    assert profile is not None
    assert profile.audio_stream.codec_name == []
    assert profile.audio_stream.channels == ""


def test_import_ffprobe_file_to_profile_fills_missing_tags_defaults(tmp_path):
    fmt = {"format_name": "matroska", "format_long_name": "Matroska"}
    payload = _make_ffprobe_payload(fmt=fmt)
    path = _write_json(tmp_path, payload)
    profile = fi.import_ffprobe_file_to_profile(path)
    assert profile is not None
    # Missing tags should be filled with the known-key defaults.
    assert "TITLE" in profile.format.tags
    assert profile.format.tags["TITLE"] is None
    assert "ENCODER_SETTINGS" in profile.format.tags


# ---------------------------------------------------------------------------
# compare_with_expected
# ---------------------------------------------------------------------------

def _make_expected_profile(**overrides):
    video = FFmpegVideoStream(
        codec_name="ffv1",
        codec_long_name="FFmpeg video codec #1",
        codec_type="video",
        codec_tag_string="FFV1",
        codec_tag="0x31564646",
        width="720",
        height="486",
        display_aspect_ratio="400:297",
        pix_fmt="yuv422p10le",
        color_space="smpte170m",
        color_transfer="bt709",
        color_primaries="smpte170m",
        field_order="bt",
        bits_per_raw_sample="10",
    )
    audio = FFmpegAudioStream(
        codec_name=["flac", "pcm_s24le"],
        codec_long_name=["FLAC (Free Lossless Audio Codec)"],
        codec_type="audio",
        codec_tag="0x0000",
        sample_fmt="s32",
        sample_rate="48000",
        channels="2",
        channel_layout="stereo",
        bits_per_raw_sample="24",
    )
    fmt = FFmpegFormat(
        format_name="matroska,webm",
        format_long_name="Matroska / WebM",
        tags={},
    )
    return FfprobeProfile(video_stream=video, audio_stream=audio, format=fmt)


def test_compare_with_expected_all_match():
    expected = _make_expected_profile()
    imported = {
        "video_stream": fi.extract_video_stream_fields(_default_video_stream()),
        "audio_stream": fi.extract_audio_stream_fields(_default_audio_stream()),
        "format": fi.extract_format_fields(_default_format()),
    }
    # width in actual is int 720; expected is "720". str() comparison should match.
    results = fi.compare_with_expected(imported, expected)
    assert results["video_stream"]["mismatches"] == {}
    assert "codec_name" in results["video_stream"]["matches"]


def test_compare_with_expected_list_subset_matches():
    """If expected audio codec_name is a subset of actual, it's a match."""
    expected = _make_expected_profile()  # expects ["flac", "pcm_s24le"]
    # Actual has both codecs.
    imported_audio = {
        "codec_name": ["flac", "pcm_s24le"],
        "codec_long_name": ["FLAC (Free Lossless Audio Codec)"],
    }
    imported = {
        "video_stream": {},
        "audio_stream": imported_audio,
        "format": {},
    }
    results = fi.compare_with_expected(imported, expected)
    assert "codec_name" in results["audio_stream"]["matches"]


def test_compare_with_expected_list_not_subset_mismatches():
    expected = _make_expected_profile()  # expects ["flac", "pcm_s24le"]
    imported_audio = {"codec_name": ["aac"]}
    imported = {
        "video_stream": {},
        "audio_stream": imported_audio,
        "format": {},
    }
    results = fi.compare_with_expected(imported, expected)
    assert "codec_name" in results["audio_stream"]["mismatches"]


def test_compare_with_expected_missing_flagged_only_for_nonempty():
    expected = _make_expected_profile()
    imported = {"video_stream": {}, "audio_stream": {}, "format": {}}
    results = fi.compare_with_expected(imported, expected)
    # codec_name (non-empty expected) should be flagged missing.
    assert "codec_name" in results["video_stream"]["missing"]


def test_compare_with_expected_empty_expected_not_missing():
    """Empty-string and empty-list expected values should not be flagged."""
    expected = _make_expected_profile()
    # Replace a couple of fields with empty expected values.
    expected.video_stream.codec_name = ""
    expected.audio_stream.codec_name = []

    imported = {"video_stream": {}, "audio_stream": {}, "format": {}}
    results = fi.compare_with_expected(imported, expected)
    assert "codec_name" not in results["video_stream"]["missing"]
    assert "codec_name" not in results["audio_stream"]["missing"]


def test_compare_with_expected_skips_tags():
    expected = _make_expected_profile()
    expected.format.tags = {"TITLE": "some_value"}
    imported = {"video_stream": {}, "audio_stream": {}, "format": {"tags": {}}}
    results = fi.compare_with_expected(imported, expected)
    # Tags field should never appear in comparison results.
    assert "tags" not in results["format"]["matches"]
    assert "tags" not in results["format"]["mismatches"]
    assert "tags" not in results["format"]["missing"]


# ---------------------------------------------------------------------------
# validate_file_against_profile
# ---------------------------------------------------------------------------

def test_validate_file_against_profile_valid(tmp_path):
    path = _write_json(tmp_path, _make_ffprobe_payload())
    expected = _make_expected_profile()
    # Loosen audio expectation — the payload has only flac, expected asks for
    # flac OR pcm_s24le (subset check on single-value list).
    expected.audio_stream.codec_name = ["flac"]
    expected.audio_stream.codec_long_name = ["FLAC (Free Lossless Audio Codec)"]
    result = fi.validate_file_against_profile(path, expected)
    assert result["valid"] is True
    assert result["total_fields"] > 0
    assert result["matching_fields"] == result["total_fields"]
    assert "sections" in result


def test_validate_file_against_profile_mismatch(tmp_path):
    payload = _make_ffprobe_payload(
        video={**_default_video_stream(), "codec_name": "h264", "width": 1920}
    )
    path = _write_json(tmp_path, payload)
    expected = _make_expected_profile()
    result = fi.validate_file_against_profile(path, expected)
    assert result["valid"] is False
    assert result["matching_fields"] < result["total_fields"]
    assert "codec_name" in result["sections"]["video_stream"]["mismatches"]


def test_validate_file_against_profile_parse_failure_returns_invalid():
    result = fi.validate_file_against_profile("/nonexistent.json", _make_expected_profile())
    assert result["valid"] is False
    assert "error" in result
    assert result["sections"] == {}
