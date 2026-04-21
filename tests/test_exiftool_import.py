"""Tests for exiftool_import — parse exiftool JSON/text output and build ExiftoolProfile."""

import json

import pytest

from AV_Spex.utils import exiftool_import as ei
from AV_Spex.utils.config_setup import ExiftoolProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_exiftool_dict():
    """What a typical exiftool JSON output for an MKV/FFV1 looks like."""
    return {
        "SourceFile": "/path/to/JPC_AV_00001.mkv",
        "FileType": "MKV",
        "FileTypeExtension": "mkv",
        "MIMEType": "video/x-matroska",
        "VideoFrameRate": 29.97,
        "ImageWidth": 720,
        "ImageHeight": 486,
        "VideoScanType": "Progressive",
        "DisplayWidth": 4,
        "DisplayHeight": 3,
        "DisplayUnit": "Display Aspect Ratio",
        "AudioChannels": 2,
        "AudioSampleRate": 48000,
        "AudioBitsPerSample": 24,
        "CodecID": ["V_MS/VFW/FOURCC / FFV1", "A_FLAC"],
    }


def _write_json(tmp_path, payload, name="exiftool.json"):
    """Wrap in a list like exiftool -json does."""
    path = tmp_path / name
    path.write_text(json.dumps([payload]))
    return str(path)


def _write_text(tmp_path, pairs, name="exiftool.txt"):
    path = tmp_path / name
    path.write_text("\n".join(f"{k}: {v}" for k, v in pairs))
    return str(path)


# ---------------------------------------------------------------------------
# parse_exiftool_json
# ---------------------------------------------------------------------------

def test_parse_exiftool_json_array_format(tmp_path):
    path = _write_json(tmp_path, _default_exiftool_dict())
    data = ei.parse_exiftool_json(path)
    assert data is not None
    assert data["FileType"] == "MKV"


def test_parse_exiftool_json_object_format(tmp_path):
    path = tmp_path / "object.json"
    path.write_text(json.dumps(_default_exiftool_dict()))
    data = ei.parse_exiftool_json(str(path))
    assert data is not None
    assert data["FileType"] == "MKV"


def test_parse_exiftool_json_empty_array_returns_none(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]")
    # Empty list is falsy → returns None (tested via parse_exiftool_file).
    assert ei.parse_exiftool_json(str(path)) is None


def test_parse_exiftool_json_unexpected_structure_returns_none(tmp_path):
    path = tmp_path / "weird.json"
    path.write_text('"just a string"')
    assert ei.parse_exiftool_json(str(path)) is None


def test_parse_exiftool_json_malformed_returns_none(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json}")
    assert ei.parse_exiftool_json(str(path)) is None


# ---------------------------------------------------------------------------
# parse_exiftool_text
# ---------------------------------------------------------------------------

def test_parse_exiftool_text_basic(tmp_path):
    path = _write_text(tmp_path, [
        ("File Type", "MKV"),
        ("Image Width", "720"),
        ("Video Frame Rate", "29.97"),
    ])
    data = ei.parse_exiftool_text(path)
    assert data is not None
    # Spaces should be stripped from keys.
    assert data["FileType"] == "MKV"
    # Digit-only strings should be converted to int.
    assert data["ImageWidth"] == 720
    # Float-convertible strings become float.
    assert data["VideoFrameRate"] == 29.97


def test_parse_exiftool_text_ignores_non_colon_lines(tmp_path):
    path = tmp_path / "out.txt"
    path.write_text(
        "=== Exiftool Output ===\n"
        "FileType: MKV\n"
        "\n"
        "random text\n"
        "ImageWidth: 720\n"
    )
    data = ei.parse_exiftool_text(str(path))
    assert data is not None
    assert data["FileType"] == "MKV"
    assert data["ImageWidth"] == 720


def test_parse_exiftool_text_empty_returns_none(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    assert ei.parse_exiftool_text(str(path)) is None


# ---------------------------------------------------------------------------
# parse_exiftool_file (format auto-detect)
# ---------------------------------------------------------------------------

def test_parse_exiftool_file_json_extension(tmp_path):
    path = _write_json(tmp_path, _default_exiftool_dict())
    data = ei.parse_exiftool_file(path)
    assert data["FileType"] == "MKV"


def test_parse_exiftool_file_text_extension(tmp_path):
    path = _write_text(tmp_path, [("File Type", "MKV")])
    data = ei.parse_exiftool_file(path)
    assert data["FileType"] == "MKV"


def test_parse_exiftool_file_log_extension(tmp_path):
    path = tmp_path / "out.log"
    path.write_text("File Type: MKV\n")
    data = ei.parse_exiftool_file(str(path))
    assert data["FileType"] == "MKV"


def test_parse_exiftool_file_unknown_extension_tries_both(tmp_path):
    """Unknown extension — should try JSON first, fall back to text."""
    path = tmp_path / "out.xyz"
    path.write_text("File Type: MKV\n")
    data = ei.parse_exiftool_file(str(path))
    assert data is not None
    assert data["FileType"] == "MKV"


def test_parse_exiftool_file_missing_file_returns_none():
    assert ei.parse_exiftool_file("/nonexistent/exiftool.json") is None


# ---------------------------------------------------------------------------
# extract_profile_fields
# ---------------------------------------------------------------------------

def test_extract_profile_fields_maps_standard_keys():
    extracted = ei.extract_profile_fields(_default_exiftool_dict())
    assert extracted["FileType"] == "MKV"
    assert extracted["ImageWidth"] == "720"
    assert extracted["ImageHeight"] == "486"


def test_extract_profile_fields_maps_alternative_names():
    """Alternative names from text-format output should be picked up."""
    data = {
        "File Type": "MKV",
        "Image Width": 1920,
        "FrameRate": 29.97,
        "Channels": 2,
    }
    extracted = ei.extract_profile_fields(data)
    # "File Type" → FileType
    assert extracted["FileType"] == "MKV"
    # "Image Width" → ImageWidth
    assert extracted["ImageWidth"] == "1920"
    # "FrameRate" → VideoFrameRate (converted to string)
    assert extracted["VideoFrameRate"] == "29.97"
    # "Channels" → AudioChannels
    assert extracted["AudioChannels"] == "2"


def test_extract_profile_fields_handles_list_values():
    """CodecID often arrives as a list of codec IDs (video + audio)."""
    extracted = ei.extract_profile_fields(_default_exiftool_dict())
    assert isinstance(extracted["CodecID"], list)
    assert "V_MS/VFW/FOURCC / FFV1" in extracted["CodecID"]
    assert "A_FLAC" in extracted["CodecID"]


def test_extract_profile_fields_sample_rate_int_to_str():
    """AudioSampleRate: int → str of int."""
    extracted = ei.extract_profile_fields({"AudioSampleRate": 48000})
    assert extracted["AudioSampleRate"] == "48000"


def test_extract_profile_fields_deduplicates_list():
    data = {"CodecID": ["FFV1", "FFV1", "FLAC"]}
    extracted = ei.extract_profile_fields(data)
    # Duplicates removed while preserving order.
    assert extracted["CodecID"] == ["FFV1", "FLAC"]


def test_extract_profile_fields_single_value_unwrapped():
    """A single-value list should collapse to a scalar string."""
    data = {"CodecID": ["FFV1"]}
    extracted = ei.extract_profile_fields(data)
    assert extracted["CodecID"] == "FFV1"


def test_extract_profile_fields_ignores_unknown_keys():
    data = {"SomeRandomField": "value", "FileType": "MKV"}
    extracted = ei.extract_profile_fields(data)
    assert "SomeRandomField" not in extracted
    assert extracted["FileType"] == "MKV"


# ---------------------------------------------------------------------------
# import_exiftool_file_to_profile
# ---------------------------------------------------------------------------

def test_import_exiftool_file_to_profile_happy_path(tmp_path):
    path = _write_json(tmp_path, _default_exiftool_dict())
    profile = ei.import_exiftool_file_to_profile(path)
    assert profile is not None
    assert isinstance(profile, ExiftoolProfile)
    assert profile.FileType == "MKV"
    assert profile.ImageWidth == "720"
    assert profile.ImageHeight == "486"
    assert "V_MS/VFW/FOURCC / FFV1" in profile.CodecID


def test_import_exiftool_file_to_profile_missing_file_returns_none():
    assert ei.import_exiftool_file_to_profile("/nonexistent.json") is None


def test_import_exiftool_file_to_profile_fills_defaults_when_fields_missing(tmp_path):
    """If only FileType is provided, other fields should get their defaults."""
    path = _write_json(tmp_path, {"FileType": "MKV"})
    profile = ei.import_exiftool_file_to_profile(path)
    assert profile is not None
    assert profile.FileType == "MKV"
    # Defaults kicked in for missing fields.
    assert profile.FileTypeExtension == "unknown"
    assert profile.MIMEType == "application/octet-stream"
    assert profile.CodecID == []
    assert profile.ImageWidth == ""


def test_import_exiftool_file_to_profile_no_relevant_fields_returns_none(tmp_path):
    """Exiftool output with no mappable fields returns None."""
    path = _write_json(tmp_path, {"SomeUnknownKey": "value"})
    assert ei.import_exiftool_file_to_profile(path) is None


# ---------------------------------------------------------------------------
# compare_with_expected
# ---------------------------------------------------------------------------

def _make_expected_profile():
    return ExiftoolProfile(
        FileType="MKV",
        FileTypeExtension="mkv",
        MIMEType="video/x-matroska",
        VideoFrameRate="29.97",
        ImageWidth="720",
        ImageHeight="486",
        VideoScanType="Progressive",
        DisplayWidth="4",
        DisplayHeight="3",
        DisplayUnit="Display Aspect Ratio",
        CodecID=["V_MS/VFW/FOURCC / FFV1", "A_FLAC"],
        AudioChannels="2",
        AudioSampleRate="48000",
        AudioBitsPerSample="24",
    )


def test_compare_with_expected_all_match():
    expected = _make_expected_profile()
    result = ei.compare_with_expected(_default_exiftool_dict(), expected)
    assert result["mismatches"] == {}
    assert result["missing"] == {}
    assert "FileType" in result["matches"]


def test_compare_with_expected_scalar_mismatch():
    expected = _make_expected_profile()
    data = dict(_default_exiftool_dict())
    data["FileType"] = "MP4"
    result = ei.compare_with_expected(data, expected)
    assert "FileType" in result["mismatches"]
    assert result["mismatches"]["FileType"]["actual"] == "MP4"


def test_compare_with_expected_list_subset_matches():
    """CodecID: expected ⊆ actual should be a match."""
    expected = _make_expected_profile()
    expected.CodecID = ["V_MS/VFW/FOURCC / FFV1"]  # subset
    result = ei.compare_with_expected(_default_exiftool_dict(), expected)
    assert "CodecID" in result["matches"]


def test_compare_with_expected_list_not_subset_mismatches():
    expected = _make_expected_profile()
    expected.CodecID = ["SOMETHING_ELSE"]
    result = ei.compare_with_expected(_default_exiftool_dict(), expected)
    assert "CodecID" in result["mismatches"]


def test_compare_with_expected_missing_flagged_only_for_nonempty():
    expected = _make_expected_profile()
    # Empty exiftool data → everything non-empty expected should be missing.
    result = ei.compare_with_expected({}, expected)
    assert "FileType" in result["missing"]
    assert "CodecID" in result["missing"]


def test_compare_with_expected_empty_expected_not_flagged():
    expected = _make_expected_profile()
    expected.FileType = ""
    expected.CodecID = []
    result = ei.compare_with_expected({}, expected)
    assert "FileType" not in result["missing"]
    assert "CodecID" not in result["missing"]


# ---------------------------------------------------------------------------
# validate_file_against_profile
# ---------------------------------------------------------------------------

def test_validate_file_against_profile_valid(tmp_path):
    path = _write_json(tmp_path, _default_exiftool_dict())
    expected = _make_expected_profile()
    result = ei.validate_file_against_profile(path, expected)
    assert result["valid"] is True
    assert result["total_fields"] > 0
    assert result["matching_fields"] == result["total_fields"]


def test_validate_file_against_profile_mismatch(tmp_path):
    data = dict(_default_exiftool_dict())
    data["FileType"] = "MP4"
    path = _write_json(tmp_path, data)
    expected = _make_expected_profile()
    result = ei.validate_file_against_profile(path, expected)
    assert result["valid"] is False
    assert "FileType" in result["mismatches"]


def test_validate_file_against_profile_parse_failure_returns_invalid():
    result = ei.validate_file_against_profile("/nonexistent.json", _make_expected_profile())
    assert result["valid"] is False
    assert "error" in result
    assert result["matches"] == {}
    assert result["mismatches"] == {}
    assert result["missing"] == {}
