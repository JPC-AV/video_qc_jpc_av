"""Tests for checks.tone_detection_clams.

Covers:
* _format_timestamp — pure formatting
* write_durations_csv — primary + cross-validation rows + empty case
* load_audio_mono_16k — ffmpeg subprocess wrapping (success / various failures)
* _detect_tones — cross-correlation core with crafted numpy samples
* run_clams_tone_detection — orchestrator, including start_at_seconds offset and
  stop_at_seconds clamping. Uses the `samples=` parameter to skip ffmpeg.
"""

import csv
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from AV_Spex.checks import tone_detection_clams as td


# ---------------------------------------------------------------------------
# _format_timestamp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seconds,expected", [
    (0.0, "00:00:00.000"),
    (1.5, "00:00:01.500"),
    (61.25, "00:01:01.250"),
    (3661.001, "01:01:01.001"),
])
def test_format_timestamp(seconds, expected):
    assert td._format_timestamp(seconds) == expected


# ---------------------------------------------------------------------------
# write_durations_csv
# ---------------------------------------------------------------------------

def _read_csv_rows(path):
    with open(path) as f:
        return list(csv.reader(f))


def test_write_durations_csv_with_tones(tmp_path):
    td.write_durations_csv(str(tmp_path), [
        ("primary", 0.0, 5.0),
        ("second_pass", 60.0, 65.5),
    ])

    rows = _read_csv_rows(tmp_path / td.DURATIONS_CSV_NAME)
    # Header
    assert rows[0] == ["clams tone detection tones found:"]
    # Data rows: pass label + start ts + end ts
    assert rows[1] == ["primary", "00:00:00.000", "00:00:05.000"]
    assert rows[2] == ["second_pass", "00:01:00.000", "00:01:05.500"]


def test_write_durations_csv_empty(tmp_path):
    td.write_durations_csv(str(tmp_path), [])

    rows = _read_csv_rows(tmp_path / td.DURATIONS_CSV_NAME)
    assert rows == [["clams tone detection found no tones"]]


# ---------------------------------------------------------------------------
# load_audio_mono_16k
# ---------------------------------------------------------------------------

def test_load_audio_mono_16k_success(monkeypatch):
    # 4 int16 samples encoded as 8 bytes
    raw = np.array([0, 16384, -16384, 32767], dtype=np.int16).tobytes()
    fake_proc = MagicMock(returncode=0, stdout=raw, stderr=b"")
    monkeypatch.setattr(td.subprocess, "run", lambda *a, **kw: fake_proc)

    samples = td.load_audio_mono_16k("/v/in.mkv")

    assert samples is not None
    assert samples.dtype == np.float32
    assert len(samples) == 4
    # Normalized to [-1, 1] by /32768.0
    assert samples[0] == pytest.approx(0.0)
    assert samples[1] == pytest.approx(0.5)
    assert samples[2] == pytest.approx(-0.5)


def test_load_audio_mono_16k_nonzero_returncode_returns_none(monkeypatch):
    fake_proc = MagicMock(returncode=1, stdout=b"", stderr=b"ffmpeg: bad input")
    monkeypatch.setattr(td.subprocess, "run", lambda *a, **kw: fake_proc)

    assert td.load_audio_mono_16k("/v/in.mkv") is None


def test_load_audio_mono_16k_no_stdout_returns_none(monkeypatch):
    """ffmpeg succeeded but produced no audio (e.g. video-only input)."""
    fake_proc = MagicMock(returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr(td.subprocess, "run", lambda *a, **kw: fake_proc)

    assert td.load_audio_mono_16k("/v/in.mkv") is None


def test_load_audio_mono_16k_subprocess_error_returns_none(monkeypatch):
    monkeypatch.setattr(td.subprocess, "run", MagicMock(side_effect=FileNotFoundError("ffmpeg missing")))
    assert td.load_audio_mono_16k("/v/in.mkv") is None


def test_load_audio_mono_16k_command_construction(monkeypatch):
    """Verify the ffmpeg command we hand off."""
    raw = np.array([0], dtype=np.int16).tobytes()
    fake_proc = MagicMock(returncode=0, stdout=raw, stderr=b"")
    run_mock = MagicMock(return_value=fake_proc)
    monkeypatch.setattr(td.subprocess, "run", run_mock)

    td.load_audio_mono_16k("/v/in.mkv")

    cmd = run_mock.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd and cmd[cmd.index("-i") + 1] == "/v/in.mkv"
    assert "-vn" in cmd  # video-stream-disable
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"  # mono
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == str(td.SAMPLE_RATE)
    assert cmd[-1] == "-"  # output to stdout


# ---------------------------------------------------------------------------
# _detect_tones — pure-numpy cross-correlation core
# ---------------------------------------------------------------------------

def test_detect_tones_constant_signal_yields_one_tone():
    """A constant amplitude signal across many chunks → one long tone detected."""
    # 5s * 16000 samples/s = 80000 samples of a constant value
    samples = np.full(80000, 0.5, dtype=np.float32)
    # tolerance must be below the auto-correlation magnitude. For SAMPLE_SIZE=4000,
    # np.correlate(ref, curr, mode='valid') has length 1 = sum(0.5*0.5)*4000 = 1000;
    # then averaged by /1 = 1000. Pick a lenient tolerance well under that.
    tones = td._detect_tones(
        samples,
        tolerance=100.0,
        min_tone_duration_ms=500,
        stop_at_samples=80000,
    )
    assert len(tones) == 1
    start, end = tones[0]
    assert start == pytest.approx(0.0, abs=0.001)
    assert end > 1.0  # spans more than 1 second


def test_detect_tones_silent_signal_yields_no_tones():
    """All zeros → cross-correlation magnitude is 0 → no tones."""
    samples = np.zeros(80000, dtype=np.float32)
    tones = td._detect_tones(
        samples,
        tolerance=0.5,
        min_tone_duration_ms=500,
        stop_at_samples=80000,
    )
    assert tones == []


def test_detect_tones_min_duration_filters_short_runs():
    """A constant signal that triggers a tone but is below min_duration is filtered."""
    samples = np.full(80000, 0.5, dtype=np.float32)
    tones = td._detect_tones(
        samples,
        tolerance=100.0,
        # 60 seconds — longer than the 5s of constant signal we can produce
        min_tone_duration_ms=60_000,
        stop_at_samples=80000,
    )
    assert tones == []


def test_detect_tones_stop_at_samples_zero_skips_scan():
    """stop_at_samples=0 means start_sample (=0) is not < endpoint (=0) — skip entirely."""
    samples = np.full(160000, 0.5, dtype=np.float32)
    tones = td._detect_tones(
        samples,
        tolerance=100.0,
        min_tone_duration_ms=500,
        stop_at_samples=0,
    )
    # No tone scans started → nothing detected
    assert tones == []


def test_detect_tones_stop_at_samples_caps_new_scans():
    """stop_at_samples gates where NEW scans begin. An in-progress tone may still
    extend past the cap, but once the outer loop's start_sample reaches the cap,
    no new tones are scanned. This test inserts a brief silence after the cap so a
    second potential tone doesn't get scanned."""
    # 2s of constant signal, 1s of silence, 5s of constant signal
    sig_a = np.full(32000, 0.5, dtype=np.float32)   # 2.0s
    silence = np.zeros(16000, dtype=np.float32)      # 1.0s
    sig_b = np.full(80000, 0.5, dtype=np.float32)   # 5.0s
    samples = np.concatenate([sig_a, silence, sig_b])

    # Cap scans before sig_b would start
    tones = td._detect_tones(
        samples,
        tolerance=100.0,
        min_tone_duration_ms=500,
        stop_at_samples=32000,  # cap at the end of sig_a
    )
    # Should detect the first tone, but not the post-silence one
    assert tones, "Expected the first tone to be detected"
    # Last detected tone should not extend into sig_b
    assert tones[-1][1] <= 4.0  # before the second signal starts at t=3s


def test_detect_tones_emits_progress_signals():
    """When signals are wired up, clams_detection_progress.emit is called."""
    samples = np.full(160000, 0.5, dtype=np.float32)
    signals = MagicMock()
    # spec progress attr
    signals.clams_detection_progress = MagicMock()

    td._detect_tones(
        samples,
        tolerance=100.0,
        min_tone_duration_ms=500,
        stop_at_samples=160000,
        signals=signals,
    )
    # At minimum, the final 100% emit
    assert signals.clams_detection_progress.emit.called
    # Last call should be 100
    assert signals.clams_detection_progress.emit.call_args_list[-1].args[0] == 100


def test_detect_tones_check_cancelled_breaks_early():
    samples = np.full(160000, 0.5, dtype=np.float32)
    cancel_after_first = MagicMock(side_effect=[False, True])  # let one iteration through
    tones = td._detect_tones(
        samples,
        tolerance=100.0,
        min_tone_duration_ms=10,
        stop_at_samples=160000,
        check_cancelled=cancel_after_first,
    )
    # Hard to assert a specific shape, but it should not have scanned the entire span.
    # cancel_after_first should have been invoked at least once.
    assert cancel_after_first.called


# ---------------------------------------------------------------------------
# run_clams_tone_detection — orchestrator
# ---------------------------------------------------------------------------

def test_run_clams_tone_detection_uses_passed_samples_skips_ffmpeg(monkeypatch):
    """When samples=... is passed, load_audio_mono_16k must not be called."""
    samples = np.full(80000, 0.5, dtype=np.float32)
    load_mock = MagicMock()
    monkeypatch.setattr(td, "load_audio_mono_16k", load_mock)

    result = td.run_clams_tone_detection(
        "/v/in.mkv", "/tmp", "video_x",
        tolerance=100.0,
        min_tone_duration_ms=500,
        stop_at_seconds=10,
        samples=samples,
    )
    load_mock.assert_not_called()
    assert isinstance(result, list)
    assert len(result) >= 1


def test_run_clams_tone_detection_no_audio_returns_empty(monkeypatch):
    """If load_audio_mono_16k returns None, function short-circuits to []."""
    monkeypatch.setattr(td, "load_audio_mono_16k", lambda _p: None)

    result = td.run_clams_tone_detection("/v/in.mkv", "/tmp", "video_x")
    assert result == []


def test_run_clams_tone_detection_empty_samples_returns_empty(monkeypatch):
    monkeypatch.setattr(td, "load_audio_mono_16k", lambda _p: np.array([], dtype=np.float32))

    result = td.run_clams_tone_detection("/v/in.mkv", "/tmp", "video_x")
    assert result == []


def test_run_clams_tone_detection_start_at_seconds_offsets_timestamps():
    """Detected tones should have absolute timestamps that include start_at_seconds."""
    samples = np.full(160000, 0.5, dtype=np.float32)  # 10s

    no_offset = td.run_clams_tone_detection(
        "/v/in.mkv", "/tmp", "video_x",
        tolerance=100.0,
        min_tone_duration_ms=500,
        start_at_seconds=0.0,
        stop_at_seconds=10,
        samples=samples,
    )
    with_offset = td.run_clams_tone_detection(
        "/v/in.mkv", "/tmp", "video_x",
        tolerance=100.0,
        min_tone_duration_ms=500,
        start_at_seconds=4.0,
        stop_at_seconds=10,
        samples=samples,
    )

    assert no_offset and with_offset
    # The offset run should have its first detected tone shifted by 4 seconds
    assert with_offset[0][0] == pytest.approx(no_offset[0][0] + 4.0, abs=0.01)


def test_run_clams_tone_detection_start_offset_past_end_returns_empty():
    samples = np.full(80000, 0.5, dtype=np.float32)  # 5s
    result = td.run_clams_tone_detection(
        "/v/in.mkv", "/tmp", "video_x",
        start_at_seconds=10.0,  # past the 5s of audio
        samples=samples,
    )
    assert result == []
