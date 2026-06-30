"""Tests for processing.run_tools.

Covers:
* run_command  — shell command construction + PATH handling
* run_tool_command — per-tool command selection and conditional execution
* _get_file_extension — extension lookup
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from AV_Spex.processing import run_tools as rt


# ---------------------------------------------------------------------------
# _get_file_extension
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool,expected", [
    ("exiftool", "json"),
    ("mediainfo", "json"),
    ("mediatrace", "xml"),
    ("ffprobe", "txt"),
    ("mkvalidator", "txt"),
])
def test_get_file_extension_known_tools(tool, expected):
    assert rt._get_file_extension(tool) == expected


def test_get_file_extension_unknown_tool_defaults_to_txt():
    assert rt._get_file_extension("not-a-real-tool") == "txt"


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------

def test_run_command_builds_quoted_shell_string_and_prepends_path():
    with patch.object(rt.subprocess, "run") as run_mock:
        rt.run_command("exiftool -j", "/v/in.mkv", ">", "/v/out.json")

    assert run_mock.call_count == 1
    cmd = run_mock.call_args[0][0]
    # Paths must be double-quoted to handle spaces
    assert cmd == 'exiftool -j "/v/in.mkv" > "/v/out.json"'

    # shell=True is required for the redirect to work
    assert run_mock.call_args.kwargs["shell"] is True

    # PATH must be prepended with the Homebrew bin dirs, /opt/homebrew/bin
    # first so Apple Silicon native binaries win over x86_64 ones under Rosetta
    env = run_mock.call_args.kwargs["env"]
    assert env["PATH"].startswith("/opt/homebrew/bin:/usr/local/bin:")


def test_run_command_handles_paths_with_spaces():
    with patch.object(rt.subprocess, "run") as run_mock:
        rt.run_command("ffprobe ...", "/path with spaces/in.mkv", ">", "/out dir/out.txt")

    cmd = run_mock.call_args[0][0]
    # The space-bearing paths must remain quoted as a single argument each
    assert '"/path with spaces/in.mkv"' in cmd
    assert '"/out dir/out.txt"' in cmd


# ---------------------------------------------------------------------------
# run_tool_command
# ---------------------------------------------------------------------------

def _make_fake_checks_config(tool_run_flags, video_file_extension="mkv"):
    """Build a stand-in ChecksConfig where each named tool has run_tool=<bool>.

    Defaults video_file_extension to 'mkv' so the mediatrace non-MKV guardrail
    is a no-op; tests exercising that guardrail pass a non-MKV extension.
    """
    checks = MagicMock()
    checks.video_file_extension = video_file_extension
    for name, run in tool_run_flags.items():
        getattr(checks.tools, name).run_tool = run
    return checks


@pytest.mark.parametrize("tool,extension", [
    ("exiftool", "json"),
    ("mediainfo", "json"),
    ("mediatrace", "xml"),
    ("ffprobe", "txt"),
])
def test_run_tool_command_runs_when_enabled(tool, extension, tmp_path, monkeypatch):
    fake_cfg = _make_fake_checks_config({tool: True})
    monkeypatch.setattr(rt.config_mgr, "get_config", lambda *a, **kw: fake_cfg)

    with patch.object(rt, "run_command") as run_cmd_mock:
        out = rt.run_tool_command(tool, "/v/in.mkv", str(tmp_path), "JPC_AV_05000")

    expected = os.path.join(str(tmp_path), f"JPC_AV_05000_{tool}_output.{extension}")
    assert out == expected

    # run_command should have been called once with the right command + paths
    assert run_cmd_mock.call_count == 1
    args, _ = run_cmd_mock.call_args
    cmd_template = args[0]
    assert tool in cmd_template or cmd_template.startswith(("exiftool", "mediainfo", "ffprobe"))
    assert args[1] == "/v/in.mkv"
    assert args[2] == ">"
    assert args[3] == expected


def test_run_tool_command_skips_when_run_tool_false(tmp_path, monkeypatch):
    fake_cfg = _make_fake_checks_config({"exiftool": False})
    monkeypatch.setattr(rt.config_mgr, "get_config", lambda *a, **kw: fake_cfg)

    with patch.object(rt, "run_command") as run_cmd_mock:
        out = rt.run_tool_command("exiftool", "/v/in.mkv", str(tmp_path), "JPC_AV_05000")

    # Output path is still returned (callers compute parser expectations from it),
    # but no shell command should have been invoked.
    assert out.endswith("JPC_AV_05000_exiftool_output.json")
    run_cmd_mock.assert_not_called()


def test_run_tool_command_mediatrace_skips_on_non_mkv(tmp_path, monkeypatch):
    """mediatrace only applies to Matroska; on non-MKV input it should be
    skipped (no command run) and return None so downstream parsing is skipped."""
    fake_cfg = _make_fake_checks_config({"mediatrace": True}, video_file_extension="mxf")
    monkeypatch.setattr(rt.config_mgr, "get_config", lambda *a, **kw: fake_cfg)

    with patch.object(rt, "run_command") as run_cmd_mock:
        out = rt.run_tool_command("mediatrace", "/v/in.mxf", str(tmp_path), "JPC_AV_05000")

    assert out is None
    run_cmd_mock.assert_not_called()


def test_run_tool_command_unknown_tool_returns_none(monkeypatch):
    monkeypatch.setattr(rt.config_mgr, "get_config", lambda *a, **kw: MagicMock())

    with patch.object(rt, "run_command") as run_cmd_mock:
        out = rt.run_tool_command("totally-fake-tool", "/v/in.mkv", "/dest", "video_id")

    assert out is None
    run_cmd_mock.assert_not_called()


# ---------------------------------------------------------------------------
# mkvalidator branch + progress estimate
# ---------------------------------------------------------------------------

def test_run_tool_command_mkvalidator_runs_and_forwards_signals(tmp_path, monkeypatch):
    fake_cfg = _make_fake_checks_config({"mkvalidator": True})
    monkeypatch.setattr(rt.config_mgr, "get_config", lambda *a, **kw: fake_cfg)
    signals = MagicMock()

    # mkvalidator does not go through run_command (it needs stderr capture); it is
    # dispatched to run_mkvalidator_command, which must receive the signals object.
    with patch.object(rt, "run_mkvalidator_command") as mk_mock:
        out = rt.run_tool_command(
            "mkvalidator", "/v/in.mkv", str(tmp_path), "JPC_AV_05000", signals=signals
        )

    expected = os.path.join(str(tmp_path), "JPC_AV_05000_mkvalidator_output.txt")
    assert out == expected
    mk_mock.assert_called_once_with("/v/in.mkv", expected, signals=signals)


def test_run_tool_command_mkvalidator_skips_on_non_mkv(tmp_path, monkeypatch):
    fake_cfg = _make_fake_checks_config({"mkvalidator": True}, video_file_extension="mxf")
    monkeypatch.setattr(rt.config_mgr, "get_config", lambda *a, **kw: fake_cfg)

    with patch.object(rt, "run_mkvalidator_command") as mk_mock:
        out = rt.run_tool_command("mkvalidator", "/v/in.mxf", str(tmp_path), "JPC_AV_05000")

    assert out is None
    mk_mock.assert_not_called()


@pytest.mark.parametrize("size_gib", [2, 5, 10, 37])
def test_estimate_mkvalidator_seconds_uses_power_law(size_gib, monkeypatch):
    monkeypatch.setattr(rt.os.path, "getsize", lambda p: int(size_gib * rt._GIB))
    expected = rt.MKVALIDATOR_EST_COEFF * size_gib ** rt.MKVALIDATOR_EST_EXP
    assert rt._estimate_mkvalidator_seconds("/v/in.mkv") == pytest.approx(expected, rel=1e-3)


def test_estimate_mkvalidator_seconds_grows_superlinearly(monkeypatch):
    """Per-GiB cost must increase with size (exponent > 1) so large files aren't
    underestimated — that was the cause of the bar pinning at 99%."""
    def est_for(gib):
        monkeypatch.setattr(rt.os.path, "getsize", lambda p: int(gib * rt._GIB))
        return rt._estimate_mkvalidator_seconds("/v/in.mkv")
    assert est_for(40) / 40 > est_for(4) / 4


def test_estimate_mkvalidator_seconds_floor_on_oserror(monkeypatch):
    def boom(_p):
        raise OSError("cannot stat")
    monkeypatch.setattr(rt.os.path, "getsize", boom)
    assert rt._estimate_mkvalidator_seconds("/v/in.mkv") == 1.0


def test_estimate_mkvalidator_seconds_floor_on_tiny_file(monkeypatch):
    monkeypatch.setattr(rt.os.path, "getsize", lambda p: int(0.05 * rt._GIB))
    assert rt._estimate_mkvalidator_seconds("/v/in.mkv") == 1.0


def test_run_mkvalidator_emits_progress_and_snaps_to_100(tmp_path, monkeypatch):
    signals = MagicMock()
    out_path = str(tmp_path / "out.txt")

    fake_proc = MagicMock()
    # Running for two poll cycles, then exited.
    fake_proc.poll.side_effect = [None, None, 0]
    monkeypatch.setattr(rt.subprocess, "Popen", lambda *a, **kw: fake_proc)
    monkeypatch.setattr(rt.time, "sleep", lambda *a, **kw: None)
    # Fix the estimate so the progress math is independent of the size model.
    monkeypatch.setattr(rt, "_estimate_mkvalidator_seconds", lambda p: 30.0)

    # Deterministic clock: each call advances 3 s.
    clock = {"t": 0.0}
    def fake_monotonic():
        now = clock["t"]
        clock["t"] += 3.0
        return now
    monkeypatch.setattr(rt.time, "monotonic", fake_monotonic)

    rt.run_mkvalidator_command("/v/in.mkv", out_path, signals=signals)

    emitted = [c.args[0] for c in signals.mkvalidator_progress.emit.call_args_list]
    assert emitted[0] == 0            # reset the bar at start
    assert emitted[-1] == 100         # snap to 100 on exit
    assert all(0 <= p <= 100 for p in emitted)
    # The mid-run ticks are the time-based estimate (3s/30s=10%, 6s/30s=20%).
    assert 10 in emitted and 20 in emitted


def test_run_mkvalidator_no_signals_still_runs(tmp_path, monkeypatch):
    out_path = str(tmp_path / "out.txt")
    fake_proc = MagicMock()
    fake_proc.poll.side_effect = [None, 0]
    popen_mock = MagicMock(return_value=fake_proc)
    monkeypatch.setattr(rt.subprocess, "Popen", popen_mock)
    monkeypatch.setattr(rt.time, "sleep", lambda *a, **kw: None)
    monkeypatch.setattr(rt.os.path, "getsize", lambda p: rt._GIB)

    # Should not raise without a signals object.
    rt.run_mkvalidator_command("/v/in.mkv", out_path, signals=None)
    popen_mock.assert_called_once()
