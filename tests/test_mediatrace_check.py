"""Tests for mediatrace_check.parse_mediatrace and create_metadata_difference_report."""

import csv
import os
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

import pytest

from AV_Spex.checks.mediatrace_check import (
    parse_mediatrace,
    create_metadata_difference_report,
    write_to_csv,
)


# ---------------------------------------------------------------------------
# Test-only config dataclasses (mirror structure expected by parse_mediatrace
# without forcing us to track drift in the real ones).
# ---------------------------------------------------------------------------

def make_encoder_settings(**fields):
    """Dynamically build an EncoderSettings-like dataclass with only the
    requested fields. parse_mediatrace flags any expected encoder-settings key
    whose value the actual XML doesn't provide (even if the expected value is
    an empty list), so keeping the field set minimal avoids spurious diffs."""

    @dataclass
    class _Settings:
        pass

    cls_fields = [
        (name, list, field(default_factory=lambda v=val: list(v)))
        for name, val in fields.items()
    ]
    from dataclasses import make_dataclass
    return make_dataclass("EncoderSettings", cls_fields)()


def FakeEncoderSettings(**fields):
    """Factory that defaults to a single Source_VTR field if none supplied."""
    if not fields:
        fields = {"Source_VTR": ["SVO-5800", "SN 11111"]}
    return make_encoder_settings(**fields)


@dataclass
class FakeMediatraceValues:
    TITLE: Optional[str] = "JPC_AV_01581"
    CATALOG_NUMBER: Optional[str] = "01581"
    ENCODER_SETTINGS: object = field(default_factory=FakeEncoderSettings)


@dataclass
class FakeSpex:
    mediatrace_values: FakeMediatraceValues = field(
        default_factory=FakeMediatraceValues
    )


@pytest.fixture
def patch_config_mgr(monkeypatch):
    """Patch ConfigManager to return a controllable fake spex config."""
    state = {"spex": FakeSpex()}

    def _configure(spex=None):
        if spex is not None:
            state["spex"] = spex

    mock_mgr = MagicMock()

    def fake_get_config(name, _cls):
        if name == "spex":
            return state["spex"]
        return MagicMock()

    mock_mgr.get_config.side_effect = fake_get_config
    monkeypatch.setattr(
        "AV_Spex.checks.mediatrace_check.ConfigManager", lambda: mock_mgr
    )
    return _configure


# ---------------------------------------------------------------------------
# XML builders
# ---------------------------------------------------------------------------

MT_NS = "https://mediaarea.net/mediatrace"


def _simple_tag(tag_name, tag_string):
    """Build a <block name='SimpleTag'> element matching the shape parse_mediatrace looks for."""
    return f"""
    <block name="SimpleTag">
        <block name="TagName">
            <data name="Data">{tag_name}</data>
        </block>
        <block name="TagString">
            <data name="Data">{tag_string}</data>
        </block>
    </block>
    """


def make_mediatrace_xml(tags):
    """Build a mediatrace XML document from a list of (name, value) tuples."""
    inner = "\n".join(_simple_tag(name, value) for name, value in tags)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<MediaTrace xmlns="{MT_NS}">
    <media>
        {inner}
    </media>
</MediaTrace>
"""


def write_xml(tmp_path, xml_str, name="mediatrace.xml"):
    path = tmp_path / name
    path.write_text(xml_str)
    return str(path)


# ---------------------------------------------------------------------------
# Happy path + mismatches
# ---------------------------------------------------------------------------

def test_parse_mediatrace_matches_expected(tmp_path, patch_config_mgr, setup_logging):
    patch_config_mgr()
    xml = make_mediatrace_xml([
        ("TITLE", "JPC_AV_01581"),
        ("CATALOG_NUMBER", "01581"),
        ("ENCODER_SETTINGS", "Source_VTR: SVO-5800, SN 11111"),
    ])
    diffs = parse_mediatrace(write_xml(tmp_path, xml))
    assert diffs == {}


def test_parse_mediatrace_missing_field(tmp_path, patch_config_mgr, setup_logging):
    patch_config_mgr()
    # Omit CATALOG_NUMBER.
    xml = make_mediatrace_xml([
        ("TITLE", "JPC_AV_01581"),
        ("ENCODER_SETTINGS", "Source_VTR: SVO-5800, SN 11111"),
    ])
    diffs = parse_mediatrace(write_xml(tmp_path, xml))
    assert "CATALOG_NUMBER" in diffs
    assert diffs["CATALOG_NUMBER"] == ["metadata field not found", ""]


def test_parse_mediatrace_encoder_settings_mismatch(tmp_path, patch_config_mgr, setup_logging):
    patch_config_mgr()
    xml = make_mediatrace_xml([
        ("TITLE", "JPC_AV_01581"),
        ("CATALOG_NUMBER", "01581"),
        ("ENCODER_SETTINGS", "Source_VTR: WrongModel, SN 00000"),
    ])
    diffs = parse_mediatrace(write_xml(tmp_path, xml))
    assert "Source_VTR" in diffs
    actual, expected = diffs["Source_VTR"]
    assert "WrongModel" in actual
    assert expected == ["SVO-5800", "SN 11111"]


def test_parse_mediatrace_encoder_settings_order_insensitive(tmp_path, patch_config_mgr, setup_logging):
    """The encoder settings comparison uses set(), so field order shouldn't matter."""
    patch_config_mgr()
    xml = make_mediatrace_xml([
        ("TITLE", "JPC_AV_01581"),
        ("CATALOG_NUMBER", "01581"),
        # Swap order.
        ("ENCODER_SETTINGS", "Source_VTR: SN 11111, SVO-5800"),
    ])
    diffs = parse_mediatrace(write_xml(tmp_path, xml))
    assert "Source_VTR" not in diffs


def test_parse_mediatrace_encoder_settings_missing_device(tmp_path, patch_config_mgr, setup_logging):
    """Expected encoder device not present in actual settings."""
    # Config expects Source_VTR AND TBC_Framesync.
    spex = FakeSpex()
    spex.mediatrace_values.ENCODER_SETTINGS = FakeEncoderSettings(
        Source_VTR=["SVO-5800"],
        TBC_Framesync=["DPS-475"],
    )
    patch_config_mgr(spex)
    xml = make_mediatrace_xml([
        ("TITLE", "JPC_AV_01581"),
        ("CATALOG_NUMBER", "01581"),
        ("ENCODER_SETTINGS", "Source_VTR: SVO-5800"),
    ])
    diffs = parse_mediatrace(write_xml(tmp_path, xml))
    assert "Encoder setting field TBC_Framesync" in diffs


def test_parse_mediatrace_multiple_semicolon_devices(tmp_path, patch_config_mgr, setup_logging):
    """ENCODER_SETTINGS with multiple devices separated by ';' are all parsed."""
    spex = FakeSpex()
    spex.mediatrace_values.ENCODER_SETTINGS = FakeEncoderSettings(
        Source_VTR=["SVO-5800"],
        TBC_Framesync=["DPS-475"],
        ADC=["Leitch"],
    )
    patch_config_mgr(spex)
    xml = make_mediatrace_xml([
        ("TITLE", "JPC_AV_01581"),
        ("CATALOG_NUMBER", "01581"),
        (
            "ENCODER_SETTINGS",
            "Source_VTR: SVO-5800; TBC_Framesync: DPS-475; ADC: Leitch",
        ),
    ])
    diffs = parse_mediatrace(write_xml(tmp_path, xml))
    # None of the three configured encoder fields should appear in diffs.
    assert "Source_VTR" not in diffs
    assert "TBC_Framesync" not in diffs
    assert "ADC" not in diffs


# ---------------------------------------------------------------------------
# Parse errors
# ---------------------------------------------------------------------------

def test_parse_mediatrace_malformed_xml_returns_none(tmp_path, patch_config_mgr, setup_logging):
    patch_config_mgr()
    path = tmp_path / "broken.xml"
    path.write_text("<not really xml>")
    result = parse_mediatrace(str(path))
    assert result is None


def test_parse_mediatrace_latin1_encoded_file(tmp_path, patch_config_mgr, setup_logging):
    """Fall back to latin-1 when UTF-8 declaration is wrong."""
    patch_config_mgr()
    # Declare utf-8 but embed a latin-1 byte (0xe9 is é in latin-1) inside the title.
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<MediaTrace xmlns="{MT_NS}"><media>'
        + _simple_tag("TITLE", "JPC_AV_01581")
        + _simple_tag("CATALOG_NUMBER", "01581")
        + _simple_tag("ENCODER_SETTINGS", "Source_VTR: SVO-5800, SN 11111")
        + "</media></MediaTrace>"
    )
    # Inject a non-UTF-8 byte into a comment so UTF-8 parse fails but latin-1 succeeds.
    raw = body.replace(
        "<media>",
        "<!-- comment with latin-1 byte: \xe9 --><media>",
    ).encode("latin-1")
    path = tmp_path / "latin1.xml"
    path.write_bytes(raw)
    diffs = parse_mediatrace(str(path))
    # If parsing succeeds via latin-1 fallback, we get a dict (possibly empty).
    assert isinstance(diffs, dict)


# ---------------------------------------------------------------------------
# write_to_csv / create_metadata_difference_report
# ---------------------------------------------------------------------------

def test_create_metadata_difference_report_empty_returns_none(tmp_path):
    result = create_metadata_difference_report({}, str(tmp_path), "JPC_AV_00001")
    assert result is None
    # No file should have been created.
    assert os.listdir(str(tmp_path)) == []


def test_create_metadata_difference_report_writes_csv(tmp_path):
    metadata_differences = {
        "exiftool": {"FileType": ["MP4", "MKV"]},
        "mediainfo": {"Format": ["AVI", "Matroska"]},
        "mediatrace": {},  # Empty — shouldn't produce rows.
        "ffprobe": {"codec_name": ["h264", "ffv1"]},
    }

    csv_path = create_metadata_difference_report(
        metadata_differences, str(tmp_path), "JPC_AV_00001"
    )

    assert csv_path is not None
    assert os.path.exists(csv_path)

    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 3  # exiftool + mediainfo + ffprobe
    tools = [r["Metadata Tool"] for r in rows]
    assert "exiftool" in tools
    assert "mediainfo" in tools
    assert "ffprobe" in tools
    assert "mediatrace" not in tools


def test_write_to_csv_row_format(tmp_path):
    """Sanity check on the row shape produced by write_to_csv."""
    csv_path = tmp_path / "out.csv"
    fieldnames = ["Metadata Tool", "Metadata Field", "Expected Value", "Actual Value"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        write_to_csv({"width": ["1920", "720"]}, "exiftool", writer)

    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    assert rows[0] == {
        "Metadata Tool": "exiftool",
        "Metadata Field": "width",
        "Expected Value": "720",
        "Actual Value": "1920",
    }
