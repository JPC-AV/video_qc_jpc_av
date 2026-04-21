"""
Tests for embed_fixity. Focused on the pure-Python helpers:
detect_hash_algorithm, extract_hashes, remove_existing_stream_hashes,
add_stream_hash_tag, compare_hashes.

The subprocess-heavy functions (make_stream_hash, extract_tags,
write_tags_to_mkv) are exercised indirectly via embed_fixity /
validate_embedded_md5 with mocked subprocess calls.
"""

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

from AV_Spex.checks import embed_fixity as ef


# ---------------------------------------------------------------------------
# detect_hash_algorithm
# ---------------------------------------------------------------------------

def test_detect_hash_algorithm_md5():
    assert ef.detect_hash_algorithm("d41d8cd98f00b204e9800998ecf8427e") == "md5"


def test_detect_hash_algorithm_sha256():
    assert ef.detect_hash_algorithm("a" * 64) == "sha256"


def test_detect_hash_algorithm_unknown_length():
    assert ef.detect_hash_algorithm("short") is None


def test_detect_hash_algorithm_none():
    assert ef.detect_hash_algorithm(None) is None


# ---------------------------------------------------------------------------
# extract_hashes
# ---------------------------------------------------------------------------

def _mkv_tags_xml(video_hash=None, audio_hash=None, extra_simple=False):
    """Build a minimal MKV-tags XML string."""
    xml = '<Tags><Tag><Targets></Targets>'
    if extra_simple:
        xml += '<Simple><Name>TITLE</Name><String>Test</String></Simple>'
    if video_hash is not None:
        xml += (
            f'<Simple><Name>VIDEO_STREAM_HASH</Name>'
            f'<String>{video_hash}</String></Simple>'
        )
    if audio_hash is not None:
        xml += (
            f'<Simple><Name>AUDIO_STREAM_HASH</Name>'
            f'<String>{audio_hash}</String></Simple>'
        )
    xml += '</Tag></Tags>'
    return xml


def test_extract_hashes_both_present():
    xml = _mkv_tags_xml(video_hash="v" * 32, audio_hash="a" * 32)
    video_hash, audio_hash = ef.extract_hashes(xml)
    assert video_hash == "v" * 32
    assert audio_hash == "a" * 32


def test_extract_hashes_none_present():
    xml = _mkv_tags_xml(extra_simple=True)
    video_hash, audio_hash = ef.extract_hashes(xml)
    assert video_hash is None
    assert audio_hash is None


def test_extract_hashes_video_only():
    xml = _mkv_tags_xml(video_hash="v" * 32)
    video_hash, audio_hash = ef.extract_hashes(xml)
    assert video_hash == "v" * 32
    assert audio_hash is None


# ---------------------------------------------------------------------------
# remove_existing_stream_hashes
# ---------------------------------------------------------------------------

def test_remove_existing_stream_hashes_clears_both():
    xml = _mkv_tags_xml(
        video_hash="v" * 32, audio_hash="a" * 32, extra_simple=True
    )
    cleaned = ef.remove_existing_stream_hashes(xml)
    # After cleanup, extracting should return (None, None).
    video_hash, audio_hash = ef.extract_hashes(cleaned)
    assert video_hash is None
    assert audio_hash is None
    # But non-hash Simple tags (like TITLE) should be preserved.
    assert "TITLE" in cleaned


def test_remove_existing_stream_hashes_is_idempotent_when_absent():
    xml = _mkv_tags_xml(extra_simple=True)
    cleaned = ef.remove_existing_stream_hashes(xml)
    assert "TITLE" in cleaned
    video_hash, audio_hash = ef.extract_hashes(cleaned)
    assert video_hash is None
    assert audio_hash is None


# ---------------------------------------------------------------------------
# add_stream_hash_tag
# ---------------------------------------------------------------------------

def test_add_stream_hash_tag_roundtrip():
    """Adding stream hashes and then extracting them should return the same values."""
    # add_stream_hash_tag expects at least one Tag without a <Targets> element
    # (for whole-file tags). Build a minimal structure.
    xml = '<Tags><Tag><Simple><Name>TITLE</Name><String>x</String></Simple></Tag></Tags>'
    result = ef.add_stream_hash_tag(xml, "v" * 32, "a" * 32)
    video_hash, audio_hash = ef.extract_hashes(result)
    assert video_hash == "v" * 32
    assert audio_hash == "a" * 32


# ---------------------------------------------------------------------------
# get_stream_hash_algorithm
# ---------------------------------------------------------------------------

def test_get_stream_hash_algorithm(monkeypatch):
    class _FakeFixity:
        stream_hash_algorithm = "SHA256"

    fake_checks = MagicMock()
    fake_checks.fixity = _FakeFixity()
    mock_mgr = MagicMock()
    mock_mgr.get_config.return_value = fake_checks
    monkeypatch.setattr(
        "AV_Spex.checks.embed_fixity.ConfigManager", lambda: mock_mgr
    )
    assert ef.get_stream_hash_algorithm() == "sha256"


# ---------------------------------------------------------------------------
# compare_hashes (logging only, test that it runs without error)
# ---------------------------------------------------------------------------

def test_compare_hashes_match():
    # Should not raise and should not log critical for matches.
    ef.compare_hashes("a" * 32, "b" * 32, "a" * 32, "b" * 32)


def test_compare_hashes_video_mismatch():
    ef.compare_hashes("a" * 32, "b" * 32, "c" * 32, "b" * 32)


def test_compare_hashes_audio_mismatch():
    ef.compare_hashes("a" * 32, "b" * 32, "a" * 32, "d" * 32)


# ---------------------------------------------------------------------------
# embed_fixity / validate_embedded_md5 integration (subprocess mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_config_for_embed(monkeypatch):
    """Configure the checks config with a controllable stream_hash_algorithm
    and overwrite_stream_fixity flag."""
    state = {"algorithm": "md5", "overwrite": True}

    def _configure(algorithm="md5", overwrite=True):
        state["algorithm"] = algorithm
        state["overwrite"] = overwrite

    class _FakeFixity:
        @property
        def stream_hash_algorithm(self):
            return state["algorithm"]

        @property
        def overwrite_stream_fixity(self):
            return state["overwrite"]

    fake_checks = MagicMock()
    fake_checks.fixity = _FakeFixity()
    mock_mgr = MagicMock()
    mock_mgr.get_config.return_value = fake_checks
    monkeypatch.setattr(
        "AV_Spex.checks.embed_fixity.ConfigManager", lambda: mock_mgr
    )
    return _configure


def test_embed_fixity_aborts_when_hash_cancelled(patch_config_for_embed, monkeypatch):
    patch_config_for_embed()
    monkeypatch.setattr(ef, "make_stream_hash", lambda *a, **kw: None)
    result = ef.embed_fixity("/fake/path.mkv", check_cancelled=lambda: False)
    assert result is None


def test_embed_fixity_no_existing_tags_aborts(patch_config_for_embed, monkeypatch):
    patch_config_for_embed()
    monkeypatch.setattr(
        ef, "make_stream_hash",
        lambda *a, **kw: ("v" * 32, "a" * 32),
    )
    monkeypatch.setattr(ef, "extract_tags", lambda path: "")
    # write_tags_to_mkv should not be called when there are no extractable tags.
    called = []
    monkeypatch.setattr(
        ef, "write_tags_to_mkv",
        lambda *a, **kw: called.append(a),
    )
    ef.embed_fixity("/fake/path.mkv", check_cancelled=lambda: False)
    assert called == []


def test_embed_fixity_respects_overwrite_disabled(patch_config_for_embed, monkeypatch):
    patch_config_for_embed(overwrite=False)
    monkeypatch.setattr(
        ef, "make_stream_hash",
        lambda *a, **kw: ("v" * 32, "a" * 32),
    )
    existing = _mkv_tags_xml(video_hash="x" * 32, audio_hash="y" * 32)
    monkeypatch.setattr(ef, "extract_tags", lambda path: existing)
    called = []
    monkeypatch.setattr(
        ef, "write_tags_to_mkv",
        lambda *a, **kw: called.append(a),
    )
    ef.embed_fixity("/fake/path.mkv", check_cancelled=lambda: False)
    # Overwrite disabled + existing hashes → write_tags_to_mkv should not be called.
    assert called == []


def test_embed_fixity_overwrites_when_enabled(
    patch_config_for_embed, monkeypatch, tmp_path
):
    patch_config_for_embed(overwrite=True)
    new_video = "n" * 32
    new_audio = "m" * 32
    monkeypatch.setattr(
        ef, "make_stream_hash",
        lambda *a, **kw: (new_video, new_audio),
    )
    existing = _mkv_tags_xml(video_hash="x" * 32, audio_hash="y" * 32)
    monkeypatch.setattr(ef, "extract_tags", lambda path: existing)

    captured = {}

    def fake_write(mkv_file, temp_xml_file):
        captured["mkv"] = mkv_file
        with open(temp_xml_file) as fh:
            captured["xml"] = fh.read()

    monkeypatch.setattr(ef, "write_tags_to_mkv", fake_write)
    ef.embed_fixity("/fake/path.mkv", check_cancelled=lambda: False)

    # The XML handed to mkvpropedit should contain the new hashes, not the old.
    assert new_video in captured["xml"]
    assert new_audio in captured["xml"]
    assert "x" * 32 not in captured["xml"]
    assert "y" * 32 not in captured["xml"]


def test_validate_embedded_md5_algorithm_mismatch(patch_config_for_embed, monkeypatch):
    """If stored hashes are SHA256 but config wants MD5, validation aborts."""
    patch_config_for_embed(algorithm="md5")
    existing = _mkv_tags_xml(video_hash="v" * 64, audio_hash="a" * 64)
    monkeypatch.setattr(ef, "extract_tags", lambda path: existing)
    called = []
    monkeypatch.setattr(
        ef, "make_stream_hash",
        lambda *a, **kw: called.append(a) or ("v", "a"),
    )
    result = ef.validate_embedded_md5("/fake/path.mkv", check_cancelled=lambda: False)
    assert result is False
    # make_stream_hash must NOT be called when algorithms mismatch.
    assert called == []


def test_validate_embedded_md5_matches(patch_config_for_embed, monkeypatch):
    """Happy path: stored MD5 hashes match computed hashes → returns True."""
    patch_config_for_embed(algorithm="md5")
    stored_v = "v" * 32
    stored_a = "a" * 32
    existing = _mkv_tags_xml(video_hash=stored_v, audio_hash=stored_a)
    monkeypatch.setattr(ef, "extract_tags", lambda path: existing)
    monkeypatch.setattr(
        ef, "make_stream_hash",
        lambda *a, **kw: (stored_v, stored_a),
    )
    result = ef.validate_embedded_md5("/fake/path.mkv", check_cancelled=lambda: False)
    assert result is True


def test_validate_embedded_md5_no_existing_tags(patch_config_for_embed, monkeypatch):
    patch_config_for_embed()
    monkeypatch.setattr(ef, "extract_tags", lambda path: "")
    result = ef.validate_embedded_md5("/fake/path.mkv", check_cancelled=lambda: False)
    assert result is False
