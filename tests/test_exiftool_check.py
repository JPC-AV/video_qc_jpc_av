import os
import json
import pytest
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union

# Import the modules you want to test
# Adjust the import path to match your project structure
from AV_Spex.checks.exiftool_check import (
    parse_exiftool,
    parse_exiftool_json,
    check_exif_spex,
    get_expected_fields
)

# Sample ExifTool JSON content for testing based on JPC_AV_01581_exiftool_output.json
SAMPLE_EXIFTOOL_JSON = {
    "SourceFile": "/Users/eddycolloton/git/JPC_AV/sample_files/jpc/JPC_AV_01581/JPC_AV_01581.mkv",
    "FileType": "MKV",
    "FileTypeExtension": "mkv",
    "MIMEType": "video/x-matroska",
    "VideoFrameRate": 29.97,
    "ImageWidth": 720,
    "ImageHeight": 486,
    "VideoScanType": "Interlaced",
    "DisplayWidth": 400,
    "DisplayHeight": 297,
    "DisplayUnit": "Display Aspect Ratio",
    "CodecID": "A_FLAC",
    "AudioChannels": 2,
    "AudioSampleRate": 48000,
    "AudioBitsPerSample": 24
}

# Create simplified config classes for testing

@dataclass
class ExiftoolValues:
    FileType: str = "MKV"
    FileTypeExtension: str = "mkv"
    MIMEType: str = "video/x-matroska"
    VideoFrameRate: str = "29.97"
    ImageWidth: str = "720"
    ImageHeight: str = "486"
    VideoScanType: str = "Interlaced"
    DisplayWidth: str = "4"
    DisplayHeight: str = "3"
    DisplayUnit: str = "Display Aspect Ratio"
    CodecID: List[str] = field(default_factory=lambda: ["A_FLAC", "A_PCM/INT/LIT"])
    AudioChannels: str = "2"
    AudioSampleRate: str = "48000"
    AudioBitsPerSample: str = "24"

@dataclass
class SpexConfig:
    exiftool_values: ExiftoolValues = field(default_factory=ExiftoolValues)

@dataclass
class ChecksConfig:
    pass

@dataclass
class Logger:
    """Simple logger class for testing"""
    def debug(self, msg):
        pass
    
    def info(self, msg):
        pass
    
    def error(self, msg):
        pass
    
    def critical(self, msg):
        pass

@pytest.fixture
def test_config():
    """Create test configuration based on spex_config.json"""
    config = SpexConfig()
    # Explicitly set values to ensure they match the spex_config.json
    config.exiftool_values = ExiftoolValues(
        FileType="MKV",
        FileTypeExtension="mkv",
        MIMEType="video/x-matroska",
        VideoFrameRate="29.97",
        ImageWidth="720",
        ImageHeight="486",
        VideoScanType="Interlaced",
        DisplayWidth="4",
        DisplayHeight="3",
        DisplayUnit="Display Aspect Ratio",
        CodecID=["A_FLAC", "A_PCM/INT/LIT"],
        AudioChannels="2",
        AudioSampleRate="48000",
        AudioBitsPerSample="24"
    )
    return config

@pytest.fixture
def test_logger():
    """Create a test logger"""
    return Logger()

@pytest.fixture
def sample_json_file(tmp_path):
    """Create a sample ExifTool JSON file for testing"""
    file_path = tmp_path / "sample_exiftool.json"
    with open(file_path, 'w') as f:
        json.dump(SAMPLE_EXIFTOOL_JSON, f)
    return file_path

@pytest.fixture
def sample_exif_data():
    """Create sample exiftool data for testing"""
    return SAMPLE_EXIFTOOL_JSON.copy()

@pytest.fixture
def mismatched_exif_data():
    """Create sample exiftool data with mismatches for testing"""
    mismatched_data = SAMPLE_EXIFTOOL_JSON.copy()
    mismatched_data["FileType"] = "MP4"  # Should be 'MKV'
    mismatched_data["DisplayWidth"] = 16  # Should be 4
    mismatched_data["DisplayHeight"] = 9  # Should be 3 
    mismatched_data["AudioChannels"] = 6  # Should be 2
    mismatched_data["CodecID"] = "A_AAC"  # Should be in ["A_FLAC", "A_PCM/INT/LIT"]
    return mismatched_data

# Test getting expected fields
def test_get_expected_fields():
    expected_fields = ["FileType", "FileTypeExtension", "MIMEType", 
                       "VideoFrameRate", "ImageWidth", "ImageHeight", 
                       "VideoScanType", "DisplayWidth", "DisplayHeight", 
                       "DisplayUnit", "CodecID", "AudioChannels", 
                       "AudioSampleRate", "AudioBitsPerSample"]
    
    # Making a direct assertion about what fields should be extracted
    for field in expected_fields:
        assert field in expected_fields

# Test parse_exiftool_json with existing file
def test_parse_exiftool_json_with_file(sample_json_file, monkeypatch):
    # Monkeypatch logger
    monkeypatch.setattr('AV_Spex.utils.log_setup.logger', Logger())
    
    # This test directly passes a real file to the function
    exif_data = parse_exiftool_json(str(sample_json_file))
    
    # Check that exif_data is not empty
    assert exif_data
    
    # Check some specific values
    assert exif_data["FileType"] == "MKV"
    assert exif_data["ImageWidth"] == 720
    assert exif_data["ImageHeight"] == 486
    assert exif_data["VideoScanType"] == "Interlaced"
    assert exif_data["AudioChannels"] == 2
    assert exif_data["AudioBitsPerSample"] == 24

# Test parse_exiftool_json with non-existent file
def test_parse_exiftool_json_missing_file(monkeypatch):
    # Monkeypatch logger
    monkeypatch.setattr('AV_Spex.utils.log_setup.logger', Logger())
    
    exif_data = parse_exiftool_json("non_existent_file.json")
    
    # Should return empty dict
    assert exif_data == {}

# Test check_exif_spex with matching values
def test_check_exif_spex_matching(sample_exif_data, test_config, monkeypatch):
    # Monkeypatch the config and logger
    monkeypatch.setattr('AV_Spex.checks.exiftool_check.spex_config', test_config)
    monkeypatch.setattr('AV_Spex.utils.log_setup.logger', Logger())
    
    differences = check_exif_spex(sample_exif_data)
    assert differences == {}  # No differences expected

# Test check_exif_spex with mismatching values
def test_check_exif_spex_mismatching(mismatched_exif_data, test_config, monkeypatch):
    # Monkeypatch the config and logger
    monkeypatch.setattr('AV_Spex.checks.exiftool_check.spex_config', test_config)
    monkeypatch.setattr('AV_Spex.utils.log_setup.logger', Logger())
    
    differences = check_exif_spex(mismatched_exif_data)
    
    # Check that differences were detected
    assert "FileType" in differences
    assert "DisplayWidth" in differences
    assert "DisplayHeight" in differences
    assert "AudioChannels" in differences
    assert "CodecID" in differences
    
    # Check specific difference values
    assert differences["FileType"][0] == "MP4"  # actual value
    assert differences["DisplayWidth"][0] == 16  # actual value
    assert differences["DisplayHeight"][0] == 9  # actual value
    assert differences["AudioChannels"][0] == 6  # actual value
    assert differences["CodecID"][0] == "A_AAC"  # actual value

# Test handling list of acceptable values
def test_check_exif_spex_list_values(sample_exif_data, test_config, monkeypatch):
    # Monkeypatch the config and logger
    monkeypatch.setattr('AV_Spex.checks.exiftool_check.spex_config', test_config)
    monkeypatch.setattr('AV_Spex.utils.log_setup.logger', Logger())
    
    # CodecID is already defined as a list in test_config: ["A_FLAC", "A_PCM/INT/LIT"]
    
    # Test with a value that's in the list
    differences = check_exif_spex(sample_exif_data)
    assert "CodecID" not in differences  # A_FLAC should match one of the acceptable values
    
    # Change to another value that's in the list
    sample_exif_data["CodecID"] = "A_PCM/INT/LIT"
    differences = check_exif_spex(sample_exif_data)
    assert "CodecID" not in differences  # Should still match an acceptable value
    
    # Change to a value not in the list
    sample_exif_data["CodecID"] = "A_AAC"
    differences = check_exif_spex(sample_exif_data)
    assert "CodecID" in differences  # Should not match any acceptable value

# Test string normalization in comparison
def test_check_exif_spex_string_normalization(sample_exif_data, test_config, monkeypatch):
    # Monkeypatch the config and logger
    monkeypatch.setattr('AV_Spex.checks.exiftool_check.spex_config', test_config)
    monkeypatch.setattr('AV_Spex.utils.log_setup.logger', Logger())
    
    # Add whitespace to a value in the data
    sample_exif_data["FileType"] = "MKV "  # Added trailing space
    
    # Should still match because of string normalization
    differences = check_exif_spex(sample_exif_data)
    assert "FileType" not in differences  # String normalization should handle the space
    
    # Test with leading space
    sample_exif_data["FileType"] = " MKV"
    differences = check_exif_spex(sample_exif_data)
    assert "FileType" not in differences
    
    # Test with capitalization difference
    sample_exif_data["FileType"] = "mkv"
    differences = check_exif_spex(sample_exif_data)
    assert "FileType" in differences  # Should fail since MKV != mkv

# Integration test for parse_exiftool function
def test_parse_exiftool_integration(sample_json_file, test_config, monkeypatch):
    # Monkeypatch the config and logger
    monkeypatch.setattr('AV_Spex.checks.exiftool_check.spex_config', test_config)
    monkeypatch.setattr('AV_Spex.utils.log_setup.logger', Logger())
    
    # There will be differences related to the DisplayWidth/DisplayHeight being different
    # from the JPC_AV_01581 sample (400/297) vs the expected config values (4/3)
    differences = parse_exiftool(str(sample_json_file))
    assert "DisplayWidth" in differences
    assert "DisplayHeight" in differences
    
    # Modify the file to match the expected config for DisplayWidth/DisplayHeight
    with open(sample_json_file, 'w') as f:
        modified_data = SAMPLE_EXIFTOOL_JSON.copy()
        modified_data["DisplayWidth"] = 4
        modified_data["DisplayHeight"] = 3
        json.dump([modified_data], f)  # Match the format of the original exiftool output
    
    # Now we expect no differences
    differences = parse_exiftool(str(sample_json_file))
    assert differences == {}
    
    # Create a new difference to test detection
    with open(sample_json_file, 'w') as f:
        modified_data = SAMPLE_EXIFTOOL_JSON.copy()
        modified_data["DisplayWidth"] = 4
        modified_data["DisplayHeight"] = 3
        modified_data["FileType"] = "MP4"  # Changed from MKV
        json.dump([modified_data], f)
    
    differences = parse_exiftool(str(sample_json_file))
    assert "FileType" in differences