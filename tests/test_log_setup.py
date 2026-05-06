"""Tests for utils.log_setup per-file logging.

Covers start_file_log / stop_file_log lifecycle:
* attaches a FileHandler with `_is_per_file_handler` marker
* writes log records to the per-file log
* removes the handler on stop, preventing further writes
* idempotent stop is safe
* orphaned per-file handlers are also cleaned up
"""

import logging
import os

import pytest

from AV_Spex.utils import log_setup


@pytest.fixture(autouse=True)
def _ensure_clean_state():
    """Make sure no per-file handler is lingering between tests."""
    log_setup.stop_file_log()
    yield
    log_setup.stop_file_log()


def _per_file_handlers():
    return [h for h in log_setup.logger.handlers if getattr(h, "_is_per_file_handler", False)]


# ---------------------------------------------------------------------------
# start_file_log
# ---------------------------------------------------------------------------

def test_start_file_log_creates_file_and_attaches_marked_handler(tmp_path):
    log_path = log_setup.start_file_log(str(tmp_path), "JPC_AV_05000")

    # File created at the expected location
    expected = os.path.join(str(tmp_path), "JPC_AV_05000_avspex_processing.log")
    assert log_path == expected
    assert os.path.exists(log_path)

    # Marker present on exactly one handler
    handlers = _per_file_handlers()
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.FileHandler)
    assert log_setup._current_file_handler is handlers[0]


def test_start_file_log_writes_records_to_file(tmp_path):
    log_path = log_setup.start_file_log(str(tmp_path), "video_123")
    log_setup.logger.info("hello from per-file log")
    # Force flush before reading
    log_setup._current_file_handler.flush()

    contents = open(log_path).read()
    assert "hello from per-file log" in contents
    # The start_file_log header lines should also be present
    assert "Processing log for: video_123" in contents


# ---------------------------------------------------------------------------
# stop_file_log
# ---------------------------------------------------------------------------

def test_stop_file_log_removes_handler_and_stops_writes(tmp_path):
    log_path = log_setup.start_file_log(str(tmp_path), "video_stop")
    log_setup.logger.info("before stop")
    log_setup._current_file_handler.flush()
    log_setup.stop_file_log()

    # Marker handlers gone, _current_file_handler reset to None
    assert _per_file_handlers() == []
    assert log_setup._current_file_handler is None

    # New writes after stop don't land in the per-file log
    log_setup.logger.info("after stop")
    contents = open(log_path).read()
    assert "before stop" in contents
    assert "after stop" not in contents


def test_stop_file_log_is_idempotent_when_no_handler(tmp_path):
    """Calling stop without an active handler should be a no-op (no exceptions)."""
    assert log_setup._current_file_handler is None
    # Should not raise
    log_setup.stop_file_log()
    log_setup.stop_file_log()
    assert _per_file_handlers() == []


def test_start_file_log_replaces_previous_handler(tmp_path):
    """Calling start a second time should detach the first handler and attach a new one."""
    log_path_a = log_setup.start_file_log(str(tmp_path), "first")
    handler_a = log_setup._current_file_handler

    log_path_b = log_setup.start_file_log(str(tmp_path), "second")
    handler_b = log_setup._current_file_handler

    # Different handler instances + paths
    assert handler_a is not handler_b
    assert log_path_a != log_path_b
    # Only one per-file handler attached at any time
    assert _per_file_handlers() == [handler_b]

    # Writes after the swap should land in the new file, not the old one
    log_setup.logger.info("only in second log")
    handler_b.flush()
    assert "only in second log" not in open(log_path_a).read()
    assert "only in second log" in open(log_path_b).read()


def test_stop_file_log_cleans_up_orphaned_per_file_handlers(tmp_path):
    """If something attaches a per-file handler without the module's tracking
    (e.g. a stale session), stop_file_log() should still detach it via the
    `_is_per_file_handler` marker safety sweep."""

    # Manually create + attach a marked handler that's NOT recorded in
    # _current_file_handler — simulating a leak.
    log_path = tmp_path / "orphan.log"
    orphan = logging.FileHandler(str(log_path))
    orphan._is_per_file_handler = True
    log_setup.logger.addHandler(orphan)

    assert orphan in log_setup.logger.handlers

    log_setup.stop_file_log()

    assert orphan not in log_setup.logger.handlers
    assert _per_file_handlers() == []
