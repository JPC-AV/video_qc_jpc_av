"""Tests for checks.make_access.

Covers:
* get_duration / get_video_dimensions — ffprobe wrapper helpers
* make_access_file — the ffmpeg command construction logic, especially:
  - NTSC (486) crop_to_480 toggle (720x480 vs native 720x486)
  - PAL (576) leave-as-is
  - Optional crop_area for sophisticated border detection
  - Optional start_time to skip color bars detected by qct-parse
  - Even-dimension safety for yuv420p
* process_access_file — config gating, existing-file short-circuit, signal emission
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from AV_Spex.checks import make_access as ma


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_popen(stdout_lines=()):
    """Build a Popen-like object whose stdout.readline() returns the given lines
    then "" forever. stderr.read() returns ""."""
    proc = MagicMock()
    iterator = iter(list(stdout_lines) + [""])
    proc.stdout.readline.side_effect = lambda: next(iterator, "")
    proc.stderr.read.return_value = ""
    return proc


def _capture_ffmpeg_cmd(monkeypatch, dim=(720, 486), duration="60.0"):
    """Patch helpers + Popen so we can inspect the ffmpeg command list build."""
    monkeypatch.setattr(ma, "get_duration", lambda _p: duration)
    monkeypatch.setattr(ma, "get_video_dimensions", lambda _p: dim)
    fake_proc = _fake_popen()
    popen_mock = MagicMock(return_value=fake_proc)
    monkeypatch.setattr(ma.subprocess, "Popen", popen_mock)
    return popen_mock


def _vf_filters_from(cmd):
    """Pull the comma-separated -vf argument out of a captured ffmpeg command list."""
    vf_index = cmd.index("-vf")
    return cmd[vf_index + 1].split(",")


# ---------------------------------------------------------------------------
# get_duration / get_video_dimensions
# ---------------------------------------------------------------------------

def test_get_duration_strips_whitespace(monkeypatch):
    fake_result = MagicMock(stdout=b"  60.123\n")
    monkeypatch.setattr(ma.subprocess, "run", lambda *a, **kw: fake_result)
    assert ma.get_duration("/v.mkv") == "60.123"


def test_get_video_dimensions_parses_wxh(monkeypatch):
    fake_result = MagicMock(stdout=b"720x486\n")
    monkeypatch.setattr(ma.subprocess, "run", lambda *a, **kw: fake_result)
    assert ma.get_video_dimensions("/v.mkv") == (720, 486)


def test_get_video_dimensions_empty_returns_none_pair(monkeypatch):
    fake_result = MagicMock(stdout=b"\n")
    monkeypatch.setattr(ma.subprocess, "run", lambda *a, **kw: fake_result)
    assert ma.get_video_dimensions("/v.mkv") == (None, None)


def test_get_video_dimensions_garbage_returns_none_pair(monkeypatch):
    fake_result = MagicMock(stdout=b"not-a-resolution")
    monkeypatch.setattr(ma.subprocess, "run", lambda *a, **kw: fake_result)
    assert ma.get_video_dimensions("/v.mkv") == (None, None)


def test_get_video_dimensions_subprocess_error_returns_none_pair(monkeypatch):
    monkeypatch.setattr(ma.subprocess, "run", MagicMock(side_effect=ma.subprocess.SubprocessError("boom")))
    assert ma.get_video_dimensions("/v.mkv") == (None, None)


# ---------------------------------------------------------------------------
# make_access_file — crop_to_480 vs native
# ---------------------------------------------------------------------------

def test_make_access_file_ntsc_crops_to_480_by_default(monkeypatch):
    popen_mock = _capture_ffmpeg_cmd(monkeypatch, dim=(720, 486))
    ma.make_access_file("/v/in.mkv", "/v/out.mp4", check_cancelled=lambda: False)

    cmd = popen_mock.call_args[0][0]
    vf = _vf_filters_from(cmd)
    assert "crop=720:480:0:3" in vf
    assert "yadif=1" in vf
    assert "format=yuv420p" in vf


def test_make_access_file_ntsc_crop_to_480_false_keeps_486(monkeypatch):
    popen_mock = _capture_ffmpeg_cmd(monkeypatch, dim=(720, 486))
    ma.make_access_file("/v/in.mkv", "/v/out.mp4", check_cancelled=lambda: False, crop_to_480=False)

    cmd = popen_mock.call_args[0][0]
    vf = _vf_filters_from(cmd)
    # Should NOT add the default 720:480 crop when crop_to_480 is disabled
    assert not any(f.startswith("crop=720:480") for f in vf)
    # yadif + yuv420p still applied
    assert "yadif=1" in vf
    assert "format=yuv420p" in vf


def test_make_access_file_pal_keeps_native_576(monkeypatch):
    popen_mock = _capture_ffmpeg_cmd(monkeypatch, dim=(720, 576))
    ma.make_access_file("/v/in.mkv", "/v/out.mp4", check_cancelled=lambda: False)

    cmd = popen_mock.call_args[0][0]
    vf = _vf_filters_from(cmd)
    # PAL: no default crop applied
    assert not any(f.startswith("crop=") for f in vf)


def test_make_access_file_unknown_dim_falls_back_to_input(monkeypatch):
    popen_mock = _capture_ffmpeg_cmd(monkeypatch, dim=(640, 480))
    ma.make_access_file("/v/in.mkv", "/v/out.mp4", check_cancelled=lambda: False)

    cmd = popen_mock.call_args[0][0]
    vf = _vf_filters_from(cmd)
    # Unknown standard: no implicit crop
    assert not any(f.startswith("crop=") for f in vf)


# ---------------------------------------------------------------------------
# make_access_file — explicit crop_area
# ---------------------------------------------------------------------------

def test_make_access_file_with_crop_area_ntsc_scales_to_480(monkeypatch):
    popen_mock = _capture_ffmpeg_cmd(monkeypatch, dim=(720, 486))
    ma.make_access_file(
        "/v/in.mkv", "/v/out.mp4",
        check_cancelled=lambda: False,
        crop_area=(8, 6, 700, 472),
    )

    cmd = popen_mock.call_args[0][0]
    vf = _vf_filters_from(cmd)
    # Crop applied first
    assert vf[0] == "crop=700:472:8:6"
    # Then scaled to 720x480 because crop_to_480 defaults to True for NTSC
    assert "scale=720:480" in vf


def test_make_access_file_with_crop_area_ntsc_native_height_when_crop_to_480_false(monkeypatch):
    popen_mock = _capture_ffmpeg_cmd(monkeypatch, dim=(720, 486))
    ma.make_access_file(
        "/v/in.mkv", "/v/out.mp4",
        check_cancelled=lambda: False,
        crop_area=(0, 0, 720, 486),
        crop_to_480=False,
    )

    cmd = popen_mock.call_args[0][0]
    vf = _vf_filters_from(cmd)
    # Scaled to native 720x486 because crop_to_480 is disabled
    assert "scale=720:486" in vf


def test_make_access_file_crop_area_odd_dimensions_made_even(monkeypatch):
    """yuv420p requires even crop dimensions, so odd width/height get rounded down by 1."""
    popen_mock = _capture_ffmpeg_cmd(monkeypatch, dim=(720, 486))
    ma.make_access_file(
        "/v/in.mkv", "/v/out.mp4",
        check_cancelled=lambda: False,
        crop_area=(0, 0, 701, 473),
    )

    cmd = popen_mock.call_args[0][0]
    vf = _vf_filters_from(cmd)
    # 701 → 700, 473 → 472
    assert vf[0] == "crop=700:472:0:0"


# ---------------------------------------------------------------------------
# make_access_file — start_time (color-bars trim)
# ---------------------------------------------------------------------------

def test_make_access_file_with_start_time_adds_ss_flag(monkeypatch):
    popen_mock = _capture_ffmpeg_cmd(monkeypatch, dim=(720, 486))
    ma.make_access_file(
        "/v/in.mkv", "/v/out.mp4",
        check_cancelled=lambda: False,
        start_time=12.345,
    )

    cmd = popen_mock.call_args[0][0]
    # -ss appears before -i
    ss_idx = cmd.index("-ss")
    i_idx = cmd.index("-i")
    assert ss_idx < i_idx
    assert cmd[ss_idx + 1] == "12.345"


def test_make_access_file_zero_start_time_no_ss(monkeypatch):
    popen_mock = _capture_ffmpeg_cmd(monkeypatch, dim=(720, 486))
    ma.make_access_file(
        "/v/in.mkv", "/v/out.mp4",
        check_cancelled=lambda: False,
        start_time=0,
    )

    cmd = popen_mock.call_args[0][0]
    assert "-ss" not in cmd


# ---------------------------------------------------------------------------
# process_access_file — config / existing-file guards
# ---------------------------------------------------------------------------

def _patch_process_config(monkeypatch, *, access_file_enabled):
    fake_cfg = MagicMock()
    fake_cfg.outputs.access_file = access_file_enabled
    fake_mgr = MagicMock()
    fake_mgr.get_config.return_value = fake_cfg
    monkeypatch.setattr(ma, "ConfigManager", lambda: fake_mgr)


def test_process_access_file_returns_none_when_disabled(tmp_path, monkeypatch):
    _patch_process_config(monkeypatch, access_file_enabled=False)
    out = ma.process_access_file("/v/in.mkv", str(tmp_path), "video_x")
    assert out is None


def test_process_access_file_short_circuits_on_existing_mp4(tmp_path, monkeypatch):
    _patch_process_config(monkeypatch, access_file_enabled=True)
    # Pre-existing .mp4 in source dir
    (tmp_path / "video_x_existing.mp4").write_text("")

    signals = MagicMock()
    with patch.object(ma, "make_access_file") as make_mock:
        out = ma.process_access_file("/v/in.mkv", str(tmp_path), "video_x", signals=signals)

    assert out is None
    make_mock.assert_not_called()
    signals.step_completed.emit.assert_called_with("Generate Access File")


def test_process_access_file_calls_make_access_file_when_enabled(tmp_path, monkeypatch):
    _patch_process_config(monkeypatch, access_file_enabled=True)

    signals = MagicMock()
    with patch.object(ma, "make_access_file") as make_mock:
        out = ma.process_access_file(
            "/v/in.mkv", str(tmp_path), "video_x",
            signals=signals,
            color_bars_end_time=4.5,
            crop_area=(0, 0, 720, 480),
            crop_to_480=True,
        )

    expected_out = os.path.join(str(tmp_path), "video_x_access.mp4")
    assert out == expected_out

    # make_access_file called with the right kwargs propagated through
    _, call_kwargs = make_mock.call_args
    assert call_kwargs["start_time"] == 4.5
    assert call_kwargs["crop_area"] == (0, 0, 720, 480)
    assert call_kwargs["crop_to_480"] is True
    assert call_kwargs["signals"] is signals

    signals.step_completed.emit.assert_called_with("Generate Access File")
