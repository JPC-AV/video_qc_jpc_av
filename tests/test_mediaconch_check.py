"""Tests for checks.mediaconch_check.

Covers parse_mediaconch_output (CSV → dict), run_mediaconch_command (subprocess
construction + return-value handling), and find_mediaconch_policy (config + path
lookup).
"""

import csv
from unittest.mock import MagicMock, patch

import pytest

from AV_Spex.checks import mediaconch_check as mc


# ---------------------------------------------------------------------------
# parse_mediaconch_output
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def test_parse_mediaconch_output_all_pass(tmp_path):
    csv_path = tmp_path / "mc_output.csv"
    _write_csv(csv_path, [
        ["filename", "policy", "Format", "Width"],
        ["test.mkv", "pass", "pass", "pass"],
    ])

    results = mc.parse_mediaconch_output(str(csv_path))
    assert results == {
        "filename": "test.mkv",
        "policy": "pass",
        "Format": "pass",
        "Width": "pass",
    }


def test_parse_mediaconch_output_with_failures(tmp_path):
    csv_path = tmp_path / "mc_output.csv"
    _write_csv(csv_path, [
        ["filename", "Format", "Width", "Height"],
        ["test.mkv", "pass", "fail", "fail"],
    ])

    results = mc.parse_mediaconch_output(str(csv_path))
    assert results["Format"] == "pass"
    assert results["Width"] == "fail"
    assert results["Height"] == "fail"


def test_parse_mediaconch_output_missing_file_returns_empty():
    results = mc.parse_mediaconch_output("/no/such/path.csv")
    assert results == {}


def test_parse_mediaconch_output_empty_csv_returns_empty(tmp_path):
    """csv.reader.__next__ raises StopIteration on empty file → caught and {} returned."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")

    results = mc.parse_mediaconch_output(str(csv_path))
    assert results == {}


def test_parse_mediaconch_output_only_header_returns_empty(tmp_path):
    """Header but no value row also raises StopIteration internally."""
    csv_path = tmp_path / "header_only.csv"
    _write_csv(csv_path, [["filename", "Format"]])

    results = mc.parse_mediaconch_output(str(csv_path))
    assert results == {}


def test_parse_mediaconch_output_zip_truncates_to_shorter_row(tmp_path):
    """If the value row is shorter than the header, zip() naturally truncates."""
    csv_path = tmp_path / "mismatched.csv"
    _write_csv(csv_path, [
        ["filename", "Format", "Width"],
        ["test.mkv", "pass"],  # only two values
    ])

    results = mc.parse_mediaconch_output(str(csv_path))
    assert results == {"filename": "test.mkv", "Format": "pass"}
    assert "Width" not in results


# ---------------------------------------------------------------------------
# run_mediaconch_command
# ---------------------------------------------------------------------------

def test_run_mediaconch_command_success_returns_true():
    fake_result = MagicMock(returncode=0, stderr="")
    with patch.object(mc.subprocess, "run", return_value=fake_result) as run_mock:
        ok = mc.run_mediaconch_command(
            command="mediaconch",
            input_path="/v/in.mkv",
            output_type="-oc",
            output_path="/v/out.csv",
            policy_path="/p/policy.xml",
        )
    assert ok is True

    called_cmd = run_mock.call_args[0][0]
    assert called_cmd == [
        "mediaconch",
        "-p", "/p/policy.xml",
        "/v/in.mkv",
        "-oc",
        "/v/out.csv",
    ]
    # capture_output=True, text=True so we can read .stderr/.stdout downstream
    assert run_mock.call_args.kwargs.get("capture_output") is True
    assert run_mock.call_args.kwargs.get("text") is True


def test_run_mediaconch_command_nonzero_returncode_returns_false():
    fake_result = MagicMock(returncode=1, stderr="boom")
    with patch.object(mc.subprocess, "run", return_value=fake_result):
        ok = mc.run_mediaconch_command("mediaconch", "/v/in.mkv", "-oc", "/v/out.csv", "/p/policy.xml")
    assert ok is False


def test_run_mediaconch_command_subprocess_raises_returns_false():
    with patch.object(mc.subprocess, "run", side_effect=OSError("ENOENT")):
        ok = mc.run_mediaconch_command("mediaconch", "/v/in.mkv", "-oc", "/v/out.csv", "/p/policy.xml")
    assert ok is False


# ---------------------------------------------------------------------------
# find_mediaconch_policy
# ---------------------------------------------------------------------------

class _FakePolicyConfigMgr:
    """Stand-in for ConfigManager that returns a fixed path from find_file()."""

    def __init__(self, found_path):
        self._found_path = found_path

    def find_file(self, filename, subdir):
        # Mimic real signature: returns a path or None
        return self._found_path


def test_find_mediaconch_policy_returns_path(monkeypatch):
    fake_mgr = _FakePolicyConfigMgr("/abs/path/policy.xml")
    monkeypatch.setattr(mc, "config_mgr", fake_mgr)

    # The module-level checks_config still has whatever policy filename — patch it too
    fake_checks_config = MagicMock()
    fake_checks_config.tools.mediaconch.mediaconch_policy = "policy.xml"
    monkeypatch.setattr(mc, "checks_config", fake_checks_config)

    assert mc.find_mediaconch_policy() == "/abs/path/policy.xml"


def test_find_mediaconch_policy_missing_file_returns_none(monkeypatch):
    fake_mgr = _FakePolicyConfigMgr(None)
    monkeypatch.setattr(mc, "config_mgr", fake_mgr)

    fake_checks_config = MagicMock()
    fake_checks_config.tools.mediaconch.mediaconch_policy = "missing.xml"
    monkeypatch.setattr(mc, "checks_config", fake_checks_config)

    assert mc.find_mediaconch_policy() is None


def test_find_mediaconch_policy_handles_unexpected_exception(monkeypatch):
    """An exception while looking up the config (e.g. attribute error) → None."""
    bad_config = MagicMock()
    # Accessing mediaconch_policy raises
    type(bad_config.tools.mediaconch).mediaconch_policy = property(
        fget=lambda self: (_ for _ in ()).throw(RuntimeError("config blew up"))
    )
    monkeypatch.setattr(mc, "checks_config", bad_config)

    assert mc.find_mediaconch_policy() is None
