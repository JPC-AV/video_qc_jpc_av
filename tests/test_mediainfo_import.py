"""Tests for mediainfo_import — parse MediaInfo JSON and build MediainfoProfile."""

import json

import pytest

from AV_Spex.utils import mediainfo_import as mi
from AV_Spex.utils.config_setup import (
    MediainfoProfile,
    MediainfoGeneralValues,
    MediainfoVideoValues,
    MediainfoAudioValues,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_general_track():
    return {
        "@type": "General",
        "FileExtension": "mkv",
        "Format": "Matroska",
        "OverallBitRate_Mode": "VBR",
    }


def _default_video_track():
    return {
        "@type": "Video",
        "Format": "FFV1",
        "Format_Settings_GOP": "N=1",
        "CodecID": "V_MS/VFW/FOURCC / FFV1",
        "Width": "720",
        "Height": "486",
        "PixelAspectRatio": "0.900",
        "DisplayAspectRatio": "1.333",
        "FrameRate_Mode_String": "Constant",
        "FrameRate": "29.970",
        "Standard": "NTSC",
        "ColorSpace": "YUV",
        "ChromaSubsampling": "4:2:2",
        "BitDepth": "10",
        "ScanType": "Interlaced",
        "ScanOrder": "BFF",
        "Compression_Mode": "Lossless",
        "colour_primaries": "BT.601 NTSC",
        "colour_primaries_Source": "Container",
        "transfer_characteristics": "BT.709",
        "transfer_characteristics_Source": "Container",
        "matrix_coefficients": "BT.601",
        "extra": {
            "MaxSlicesCount": "24",
            "ErrorDetectionType": "Per slice",
        },
    }


def _default_audio_track():
    return {
        "@type": "Audio",
        "Format": "FLAC",
        "Channels": "2",
        "SamplingRate": "48000",
        "BitDepth": "24",
        "Compression_Mode": "Lossless",
    }


def _make_mediainfo_payload(general=None, video=None, audio=None):
    tracks = []
    if general is not None:
        tracks.append(general)
    else:
        tracks.append(_default_general_track())
    if video is not None:
        tracks.append(video)
    else:
        tracks.append(_default_video_track())
    if audio is not None:
        tracks.append(audio)
    else:
        tracks.append(_default_audio_track())
    return {"media": {"track": tracks}}


def _write_json(tmp_path, payload, name="mediainfo.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


# ---------------------------------------------------------------------------
# parse_mediainfo_json_file
# ---------------------------------------------------------------------------

def test_parse_mediainfo_json_file_happy_path(tmp_path):
    path = _write_json(tmp_path, _make_mediainfo_payload())
    data = mi.parse_mediainfo_json_file(path)
    assert data is not None
    assert data["General"]["Format"] == "Matroska"
    assert data["Video"]["Format"] == "FFV1"
    assert data["Audio"]["Format"] == "FLAC"


def test_parse_mediainfo_json_file_missing_file():
    assert mi.parse_mediainfo_json_file("/nonexistent.json") is None


def test_parse_mediainfo_json_file_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{definitely not json}")
    assert mi.parse_mediainfo_json_file(str(path)) is None


def test_parse_mediainfo_json_file_missing_media_structure(tmp_path):
    path = _write_json(tmp_path, {"no_media_key": True})
    assert mi.parse_mediainfo_json_file(str(path)) is None


def test_parse_mediainfo_json_file_only_general_track(tmp_path):
    payload = {"media": {"track": [_default_general_track()]}}
    path = _write_json(tmp_path, payload)
    data = mi.parse_mediainfo_json_file(path)
    assert data is not None
    assert data["General"]["Format"] == "Matroska"
    # Absent sections should be empty dicts.
    assert data["Video"] == {}
    assert data["Audio"] == {}


def test_parse_mediainfo_json_file_latin1_fallback(tmp_path):
    payload = _make_mediainfo_payload()
    raw = json.dumps(payload).encode("utf-8")
    raw = raw.replace(b'"Matroska"', b'"Matroska_\xe9"')
    path = tmp_path / "latin1.json"
    path.write_bytes(raw)
    data = mi.parse_mediainfo_json_file(str(path))
    assert data is not None
    assert "Matroska" in data["General"]["Format"]


# ---------------------------------------------------------------------------
# extract_* field extractors
# ---------------------------------------------------------------------------

def test_extract_general_profile_fields_picks_known_keys():
    general = _default_general_track()
    general["UnrelatedField"] = "ignored"
    extracted = mi.extract_general_profile_fields(general)
    assert extracted["Format"] == "Matroska"
    assert "UnrelatedField" not in extracted
    assert "@type" not in extracted


def test_extract_general_profile_fields_error_detection_type_from_extra():
    general = {
        "@type": "General",
        "FileExtension": "mkv",
        "Format": "Matroska",
        "OverallBitRate_Mode": "VBR",
        "extra": {"ErrorDetectionType": "Per slice"},
    }
    # ErrorDetectionType is not on MediainfoGeneralValues — should be ignored.
    extracted = mi.extract_general_profile_fields(general)
    assert "ErrorDetectionType" not in extracted


def test_extract_video_profile_fields_pulls_extra_subfields():
    video = _default_video_track()
    extracted = mi.extract_video_profile_fields(video)
    # Extra sub-dict fields must be hoisted to the top level.
    assert extracted["MaxSlicesCount"] == "24"
    assert extracted["ErrorDetectionType"] == "Per slice"
    # Plain top-level fields should also be present.
    assert extracted["Width"] == "720"


def test_extract_video_profile_fields_no_extra_dict():
    video = _default_video_track()
    video.pop("extra")
    extracted = mi.extract_video_profile_fields(video)
    # MaxSlicesCount / ErrorDetectionType should not be present.
    assert "MaxSlicesCount" not in extracted
    assert "ErrorDetectionType" not in extracted
    assert extracted["Width"] == "720"


def test_extract_audio_profile_fields_picks_known_keys():
    audio = _default_audio_track()
    audio["UnrelatedField"] = "ignored"
    extracted = mi.extract_audio_profile_fields(audio)
    assert extracted["Format"] == "FLAC"
    assert extracted["Channels"] == "2"
    assert "UnrelatedField" not in extracted


# ---------------------------------------------------------------------------
# _apply_defaults
# ---------------------------------------------------------------------------

def test_apply_defaults_fills_missing_str_with_empty_string():
    result = mi._apply_defaults({"Format": "Matroska"}, MediainfoGeneralValues)
    assert result["Format"] == "Matroska"
    assert result["FileExtension"] == ""


def test_apply_defaults_fills_missing_list_with_empty_list():
    # MediainfoAudioValues.Format is a List[str].
    result = mi._apply_defaults({}, MediainfoAudioValues)
    assert result["Format"] == []
    assert result["Channels"] == ""


# ---------------------------------------------------------------------------
# import_mediainfo_file_to_profile
# ---------------------------------------------------------------------------

def test_import_mediainfo_file_to_profile_happy_path(tmp_path):
    path = _write_json(tmp_path, _make_mediainfo_payload())
    profile = mi.import_mediainfo_file_to_profile(path)
    assert profile is not None
    assert isinstance(profile, MediainfoProfile)
    assert profile.general.Format == "Matroska"
    assert profile.video.Format == "FFV1"
    assert profile.video.MaxSlicesCount == "24"
    assert profile.audio.Format == "FLAC"


def test_import_mediainfo_file_to_profile_missing_file_returns_none():
    assert mi.import_mediainfo_file_to_profile("/nonexistent.json") is None


def test_import_mediainfo_file_to_profile_video_only(tmp_path):
    # Audio absent — audio fields should default to empty list/str.
    payload = {"media": {"track": [_default_general_track(), _default_video_track()]}}
    path = _write_json(tmp_path, payload)
    profile = mi.import_mediainfo_file_to_profile(path)
    assert profile is not None
    assert profile.audio.Format == []
    assert profile.audio.Channels == ""


# ---------------------------------------------------------------------------
# compare_with_expected
# ---------------------------------------------------------------------------

def _make_expected_profile(**overrides):
    general = MediainfoGeneralValues(
        FileExtension="mkv",
        Format="Matroska",
        OverallBitRate_Mode="VBR",
    )
    video = MediainfoVideoValues(
        Format="FFV1",
        Format_Settings_GOP="N=1",
        CodecID="V_MS/VFW/FOURCC / FFV1",
        Width="720",
        Height="486",
        PixelAspectRatio="0.900",
        DisplayAspectRatio="1.333",
        FrameRate_Mode_String="Constant",
        FrameRate="29.970",
        Standard="NTSC",
        ColorSpace="YUV",
        ChromaSubsampling="4:2:2",
        BitDepth="10",
        ScanType="Interlaced",
        ScanOrder="BFF",
        Compression_Mode="Lossless",
        colour_primaries="BT.601 NTSC",
        colour_primaries_Source="Container",
        transfer_characteristics="BT.709",
        transfer_characteristics_Source="Container",
        matrix_coefficients="BT.601",
        MaxSlicesCount="24",
        ErrorDetectionType="Per slice",
    )
    audio = MediainfoAudioValues(
        Format=["FLAC", "PCM"],
        Channels="2",
        SamplingRate="48000",
        BitDepth="24",
        Compression_Mode="Lossless",
    )
    return MediainfoProfile(general=general, video=video, audio=audio)


def test_compare_with_expected_all_match():
    expected = _make_expected_profile()
    imported = {
        "general": mi.extract_general_profile_fields(_default_general_track()),
        "video": mi.extract_video_profile_fields(_default_video_track()),
        "audio": mi.extract_audio_profile_fields(_default_audio_track()),
    }
    results = mi.compare_with_expected(imported, expected)
    # No video mismatches.
    assert results["video"]["mismatches"] == {}
    assert "Format" in results["general"]["matches"]


def test_compare_with_expected_list_subset_matches():
    """Expected audio Format list ['FLAC', 'PCM'] — if actual is a list
    containing both, it matches; string 'FLAC' should compare directly."""
    expected = _make_expected_profile()
    # Actual is a scalar string (as mediainfo returns); compare_with_expected
    # falls back to string-in-list check.
    imported = {
        "general": {},
        "video": {},
        "audio": {"Format": "FLAC"},
    }
    results = mi.compare_with_expected(imported, expected)
    assert "Format" in results["audio"]["matches"]


def test_compare_with_expected_list_mismatch():
    expected = _make_expected_profile()
    imported = {
        "general": {},
        "video": {},
        "audio": {"Format": "AAC"},
    }
    results = mi.compare_with_expected(imported, expected)
    assert "Format" in results["audio"]["mismatches"]


def test_compare_with_expected_missing_flagged_only_for_nonempty():
    expected = _make_expected_profile()
    imported = {"general": {}, "video": {}, "audio": {}}
    results = mi.compare_with_expected(imported, expected)
    assert "Format" in results["general"]["missing"]
    # Audio Format list is non-empty → should also be flagged missing.
    assert "Format" in results["audio"]["missing"]


def test_compare_with_expected_empty_expected_not_missing():
    expected = _make_expected_profile()
    expected.general.Format = ""
    expected.audio.Format = []
    imported = {"general": {}, "video": {}, "audio": {}}
    results = mi.compare_with_expected(imported, expected)
    assert "Format" not in results["general"]["missing"]
    assert "Format" not in results["audio"]["missing"]


def test_compare_with_expected_list_to_list_subset():
    """Actual has multiple codecs, expected asks for a subset — should match."""
    expected = _make_expected_profile()  # expects ["FLAC", "PCM"]
    imported = {
        "general": {},
        "video": {},
        "audio": {"Format": ["FLAC", "PCM", "AC3"]},
    }
    results = mi.compare_with_expected(imported, expected)
    assert "Format" in results["audio"]["matches"]


# ---------------------------------------------------------------------------
# validate_file_against_profile
# ---------------------------------------------------------------------------

def test_validate_file_against_profile_valid(tmp_path):
    path = _write_json(tmp_path, _make_mediainfo_payload())
    expected = _make_expected_profile()
    # Actual audio Format is scalar "FLAC"; expected asks for ["FLAC", "PCM"].
    # String "FLAC" is in the expected list → match.
    result = mi.validate_file_against_profile(path, expected)
    assert result["valid"] is True
    assert result["matching_fields"] == result["total_fields"]
    assert "sections" in result


def test_validate_file_against_profile_mismatch(tmp_path):
    video = _default_video_track()
    video["Format"] = "H264"
    payload = _make_mediainfo_payload(video=video)
    path = _write_json(tmp_path, payload)

    expected = _make_expected_profile()
    result = mi.validate_file_against_profile(path, expected)
    assert result["valid"] is False
    assert "Format" in result["sections"]["video"]["mismatches"]


def test_validate_file_against_profile_parse_failure_returns_invalid():
    result = mi.validate_file_against_profile(
        "/nonexistent.json", _make_expected_profile()
    )
    assert result["valid"] is False
    assert "error" in result
    assert result["sections"] == {}
