"""Tests for utils.dir_setup.

Covers the small, mostly-pure helpers:
* validate_input_paths — file vs directory mode, sys.exit on invalid
* find_mkv — single match, multiple match, qctools-extension exclusion, no match
* check_directory — name-matches-video-id check
* make_qc_output_dir — creates dir only when something is enabled
* make_report_dir — wipes-and-recreates behavior
* move_vrec_files — moves matching extensions, leaves others alone

initialize_directory is exercised indirectly through find_mkv/check_directory;
its full orchestration depends on module-level config + filename validation
which is covered elsewhere.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from AV_Spex.utils import dir_setup as ds


# ---------------------------------------------------------------------------
# validate_input_paths
# ---------------------------------------------------------------------------

def test_validate_input_paths_directory_mode_returns_dir(tmp_path):
    result = ds.validate_input_paths([str(tmp_path)], is_file_mode=False)
    assert result == [str(tmp_path)]


def test_validate_input_paths_file_mode_returns_parent_dir(tmp_path):
    f = tmp_path / "video.mkv"
    f.write_text("")
    result = ds.validate_input_paths([str(f)], is_file_mode=True)
    assert result == [str(tmp_path)]


def test_validate_input_paths_invalid_dir_exits():
    with pytest.raises(SystemExit) as exc:
        ds.validate_input_paths(["/no/such/dir"], is_file_mode=False)
    assert exc.value.code == 1


def test_validate_input_paths_invalid_file_exits():
    with pytest.raises(SystemExit) as exc:
        ds.validate_input_paths(["/no/such/file.mkv"], is_file_mode=True)
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# find_mkv
# ---------------------------------------------------------------------------

def test_find_mkv_returns_single_match(tmp_path):
    (tmp_path / "JPC_AV_05000.mkv").write_text("")
    assert ds.find_mkv(str(tmp_path)) == str(tmp_path / "JPC_AV_05000.mkv")


def test_find_mkv_ignores_qctools_mkv(tmp_path):
    """*.qctools.mkv files should not count as the source video."""
    (tmp_path / "JPC_AV_05000.mkv").write_text("")
    (tmp_path / "JPC_AV_05000.qctools.mkv").write_text("")
    assert ds.find_mkv(str(tmp_path)) == str(tmp_path / "JPC_AV_05000.mkv")


def test_find_mkv_returns_none_when_no_mkv(tmp_path):
    (tmp_path / "notes.txt").write_text("")
    assert ds.find_mkv(str(tmp_path)) is None


def test_find_mkv_returns_none_when_multiple_mkvs(tmp_path):
    (tmp_path / "a.mkv").write_text("")
    (tmp_path / "b.mkv").write_text("")
    assert ds.find_mkv(str(tmp_path)) is None


def test_find_mkv_case_insensitive(tmp_path):
    (tmp_path / "VIDEO.MKV").write_text("")
    assert ds.find_mkv(str(tmp_path)) == str(tmp_path / "VIDEO.MKV")


# ---------------------------------------------------------------------------
# check_directory
# ---------------------------------------------------------------------------

def test_check_directory_match(tmp_path):
    d = tmp_path / "JPC_AV_05000"
    d.mkdir()
    assert ds.check_directory(str(d), "JPC_AV_05000") is True


def test_check_directory_startswith_match(tmp_path):
    """check_directory accepts startswith() matches, not just exact equality."""
    d = tmp_path / "JPC_AV_05000_extra"
    d.mkdir()
    assert ds.check_directory(str(d), "JPC_AV_05000") is True


def test_check_directory_mismatch(tmp_path):
    d = tmp_path / "different_name"
    d.mkdir()
    assert ds.check_directory(str(d), "JPC_AV_05000") is False


# ---------------------------------------------------------------------------
# make_report_dir
# ---------------------------------------------------------------------------

def test_make_report_dir_creates_new(tmp_path):
    out = ds.make_report_dir(str(tmp_path), "video_42")
    assert out == os.path.join(str(tmp_path), "video_42_report_csvs")
    assert os.path.isdir(out)


def test_make_report_dir_wipes_existing(tmp_path):
    """An existing report dir is removed and recreated."""
    existing = tmp_path / "video_42_report_csvs"
    existing.mkdir()
    (existing / "old_artifact.csv").write_text("stale data")

    out = ds.make_report_dir(str(tmp_path), "video_42")

    assert os.path.isdir(out)
    assert not os.path.exists(os.path.join(out, "old_artifact.csv"))


# ---------------------------------------------------------------------------
# make_qc_output_dir
# ---------------------------------------------------------------------------

def _checks_with(*, mediainfo=False, mediatrace=False, exiftool=False, ffprobe=False,
                 qctools=False, qct_parse=False, check_fixity=False):
    """Build a fake ChecksConfig where individual sub-fields can be toggled on."""
    cfg = MagicMock()
    cfg.tools.mediainfo.run_tool = mediainfo
    cfg.tools.mediainfo.check_tool = mediainfo
    cfg.tools.mediatrace.run_tool = mediatrace
    cfg.tools.mediatrace.check_tool = mediatrace
    cfg.tools.exiftool.run_tool = exiftool
    cfg.tools.exiftool.check_tool = exiftool
    cfg.tools.ffprobe.run_tool = ffprobe
    cfg.tools.ffprobe.check_tool = ffprobe
    cfg.tools.qctools.run_tool = qctools
    cfg.tools.qct_parse.run_tool = qct_parse
    cfg.fixity.check_fixity = check_fixity
    return cfg


def test_make_qc_output_dir_creates_when_metadata_enabled(tmp_path, monkeypatch):
    fake_cfg = _checks_with(mediainfo=True)
    fake_mgr = MagicMock()
    fake_mgr.get_config.return_value = fake_cfg
    monkeypatch.setattr(ds, "ConfigManager", lambda: fake_mgr)

    out = ds.make_qc_output_dir(str(tmp_path), "video_X")

    assert out == os.path.join(str(tmp_path), "video_X_qc_metadata")
    assert os.path.isdir(out)


def test_make_qc_output_dir_creates_when_fixity_enabled(tmp_path, monkeypatch):
    """check_fixity alone should be enough to create the dir."""
    fake_cfg = _checks_with(check_fixity=True)
    fake_mgr = MagicMock()
    fake_mgr.get_config.return_value = fake_cfg
    monkeypatch.setattr(ds, "ConfigManager", lambda: fake_mgr)

    out = ds.make_qc_output_dir(str(tmp_path), "video_X")
    assert os.path.isdir(out)


def test_make_qc_output_dir_skips_creation_when_nothing_enabled(tmp_path, monkeypatch):
    fake_cfg = _checks_with()  # all defaults false
    fake_mgr = MagicMock()
    fake_mgr.get_config.return_value = fake_cfg
    monkeypatch.setattr(ds, "ConfigManager", lambda: fake_mgr)

    out = ds.make_qc_output_dir(str(tmp_path), "video_X")

    # Path is still returned for downstream use, but no directory is created.
    assert out == os.path.join(str(tmp_path), "video_X_qc_metadata")
    assert not os.path.exists(out)


# ---------------------------------------------------------------------------
# move_vrec_files
# ---------------------------------------------------------------------------

def test_move_vrec_files_moves_matching_extensions(tmp_path):
    # Files that should be moved
    moved_names = [
        "x_QC_output_graphs.jpeg",
        "x_vrecord_input.log",
        "x_capture_options.log",
        "x.framemd5",
    ]
    for name in moved_names:
        (tmp_path / name).write_text("")
    # Files that should be left alone
    (tmp_path / "x.mkv").write_text("")
    (tmp_path / "notes.txt").write_text("")

    ds.move_vrec_files(str(tmp_path), "x")

    vrec_dir = tmp_path / "x_vrecord_metadata"
    assert vrec_dir.is_dir()
    for name in moved_names:
        assert (vrec_dir / name).exists(), f"{name} should have been moved"
        assert not (tmp_path / name).exists(), f"{name} should have been removed from source"

    # Untouched files
    assert (tmp_path / "x.mkv").exists()
    assert (tmp_path / "notes.txt").exists()


def test_move_vrec_files_no_matching_files_does_not_create_dir(tmp_path):
    (tmp_path / "x.mkv").write_text("")
    ds.move_vrec_files(str(tmp_path), "x")
    assert not (tmp_path / "x_vrecord_metadata").exists()


def test_move_vrec_files_already_in_subdir_no_op(tmp_path):
    """If the vrecord subdir already holds the matching files, function is a no-op."""
    vrec_dir = tmp_path / "x_vrecord_metadata"
    vrec_dir.mkdir()
    (vrec_dir / "x.framemd5").write_text("already moved")

    ds.move_vrec_files(str(tmp_path), "x")
    # Pre-existing file untouched
    assert (vrec_dir / "x.framemd5").read_text() == "already moved"
