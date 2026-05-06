# tests/test_profile_piecharts.py
import pytest
import pandas as pd
import os
from base64 import b64encode

from AV_Spex.utils.generate_report import make_profile_piecharts

@pytest.fixture
def sample_colorbars_eval_csv(tmp_path):
    """Create a sample colorbars evaluation CSV file"""
    csv_content = """Metadata line 1
Metadata line 2
TotalFrames,15000
Tag,Number of failed frames,Percentage of failed frames
YMAX,0,0.00
YMIN,0,0.00
UMIN,0,0.00
UMAX,0,0.00
VMIN,3,0.02
VMAX,8,0.05
SATMAX,7,0.04
SATMIN,0,0.00
Total,14,0.09"""
    
    csv_path = tmp_path / "qct-parse_colorbars_values.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)

@pytest.fixture
def sample_colorbars_eval_csv_no_failures(tmp_path):
    """Create a sample colorbars evaluation CSV file with no failures"""
    csv_content = """Metadata line 1
Metadata line 2
TotalFrames,15000
Tag,Number of failed frames,Percentage of failed frames
YMAX,0,0.00
YMIN,0,0.00
UMIN,0,0.00
UMAX,0,0.00
VMIN,0,0.00
VMAX,0,0.00
SATMAX,0,0.00
SATMIN,0,0.00
Total,0,0.00"""
    
    csv_path = tmp_path / "qct-parse_colorbars_values_no_failures.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)

@pytest.fixture
def sample_colorbars_thumbs_dict(tmp_path):
    """Create a sample thumbs dictionary with test images for colorbars evaluation"""
    thumb_dir = tmp_path / "ThumbExports"
    thumb_dir.mkdir()
    
    # Create dummy image files for colorbars failures
    thumbs = {}
    for field in ['SATMAX', 'VMAX']:  # Example failing fields
        test_image = thumb_dir / f"JPC_AV_05000.color_bars_evaluation.{field}.value.timestamp.png"
        # Create a minimal valid PNG file
        with open(test_image, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x00\x00\x02\x00\x01\xe5\x27\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82')
        
        thumbs[f'Failed frame \n\n{field}:704\n\n0:00:00:11:8120'] = (
            str(test_image),
            field,
            '0:00:00:11:8120'
        )
    
    return thumbs

@pytest.fixture
def sample_colorbars_failure_summary():
    """Create a sample failure info summary for colorbars"""
    return {
        "0:00:00:11:8120": [
            {"tag": "SATMAX", "tagValue": "704", "over": "405"},
        ],
        "0:00:07:12:3320": [
            {"tag": "VMAX", "tagValue": "1016", "over": "867"},
        ]
    }

@pytest.fixture
def check_cancelled():
    """Mock check_cancelled function"""
    return lambda: False

VIDEO_ID = "JPC_AV_05000"


def test_make_profile_piecharts_colorbars_success(sample_colorbars_eval_csv,
                                                 sample_colorbars_thumbs_dict,
                                                 sample_colorbars_failure_summary,
                                                 check_cancelled,
                                                 setup_logging):
    """Test successful generation of colorbars evaluation pie charts"""
    html_output = make_profile_piecharts(
        sample_colorbars_eval_csv,
        sample_colorbars_thumbs_dict,
        sample_colorbars_failure_summary,
        VIDEO_ID,
        check_cancelled=check_cancelled,
    )

    assert html_output is not None
    # Check for expected content
    assert 'SATMAX' in html_output
    assert 'VMAX' in html_output
    assert 'Failed Frames' in html_output
    assert 'Other Frames' in html_output
    assert 'data:image/jpeg;base64,' in html_output
    assert '704' in html_output  # Check for specific failure value
    assert '405' in html_output  # Check for threshold value

def test_make_profile_piecharts_colorbars_no_failures(sample_colorbars_eval_csv_no_failures,
                                                     sample_colorbars_thumbs_dict,
                                                     check_cancelled,
                                                     setup_logging):
    """Test with no colorbars failures"""
    empty_failure_summary = {}

    html_output = make_profile_piecharts(
        sample_colorbars_eval_csv_no_failures,
        sample_colorbars_thumbs_dict,
        empty_failure_summary,
        VIDEO_ID,
        check_cancelled=check_cancelled,
    )

    assert html_output is not None
    assert 'Peak Values outside of Threshold' not in html_output

def test_make_profile_piecharts_colorbars_cancelled(sample_colorbars_eval_csv,
                                                   sample_colorbars_thumbs_dict,
                                                   sample_colorbars_failure_summary,
                                                   setup_logging):
    """Test when check_cancelled returns True"""
    def cancelled_check():
        return True

    html_output = make_profile_piecharts(
        sample_colorbars_eval_csv,
        sample_colorbars_thumbs_dict,
        sample_colorbars_failure_summary,
        VIDEO_ID,
        check_cancelled=cancelled_check,
    )

    assert html_output is None

def test_make_profile_piecharts_colorbars_missing_file(sample_colorbars_thumbs_dict,
                                                      sample_colorbars_failure_summary,
                                                      check_cancelled,
                                                      setup_logging):
    """Test behavior when colorbars CSV file is missing"""
    html_output = make_profile_piecharts(
        "nonexistent_file.csv",
        sample_colorbars_thumbs_dict,
        sample_colorbars_failure_summary,
        VIDEO_ID,
        check_cancelled=check_cancelled,
    )

    assert html_output is None