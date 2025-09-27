"""
Utility functions for importing exiftool data from various file formats.
Supports JSON and text (colon-separated) formats.
"""

import json
import os
from typing import Dict, Optional, List, Union
from pathlib import Path

from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ExiftoolProfile


def parse_exiftool_json(file_path: str) -> Optional[Dict[str, any]]:
    """
    Parse exiftool JSON output file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict with exiftool data or None if parsing fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle both single object and array formats
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # Take first item if it's an array
        elif isinstance(data, dict):
            return data
        else:
            logger.error(f"Unexpected JSON structure in {file_path}")
            return None
            
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


def parse_exiftool_text(file_path: str) -> Optional[Dict[str, any]]:
    """
    Parse exiftool text output (colon-separated format).
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Dict with exiftool data or None if parsing fails
    """
    try:
        data = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not ':' in line:
                    continue
                    
                # Split on first colon and clean up
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Convert key to match JSON format (remove spaces)
                    key = key.replace(' ', '')
                    
                    # Try to parse numeric values
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        try:
                            value = float(value)
                        except:
                            pass  # Keep as string
                    
                    data[key] = value
                    
        return data if data else None
        
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


def parse_exiftool_file(file_path: str) -> Optional[Dict[str, any]]:
    """
    Parse exiftool output file, automatically detecting format.
    
    Args:
        file_path: Path to the exiftool output file
        
    Returns:
        Dict with exiftool data or None if parsing fails
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
        
    # Try to determine format by extension first
    ext = Path(file_path).suffix.lower()
    
    if ext == '.json':
        return parse_exiftool_json(file_path)
    elif ext in ['.txt', '.log']:
        return parse_exiftool_text(file_path)
    else:
        # Try both formats
        logger.debug(f"Unknown extension {ext}, trying JSON first")
        data = parse_exiftool_json(file_path)
        if data is None:
            logger.debug("JSON parsing failed, trying text format")
            data = parse_exiftool_text(file_path)
        return data


def extract_profile_fields(exiftool_data: Dict[str, any]) -> Dict[str, any]:
    """
    Extract fields relevant to ExiftoolProfile from raw exiftool data.
    
    Maps various possible field names to the expected profile fields.
    
    Args:
        exiftool_data: Raw exiftool data dictionary
        
    Returns:
        Dict with fields matching ExiftoolProfile structure
    """
    # Define field mappings (exiftool output name -> profile field name)
    field_mappings = {
        'FileType': ['FileType', 'File Type'],
        'FileTypeExtension': ['FileTypeExtension', 'FileTypeExt', 'File Type Extension'],
        'MIMEType': ['MIMEType', 'MIME Type'],
        'VideoFrameRate': ['VideoFrameRate', 'Video Frame Rate', 'FrameRate'],
        'ImageWidth': ['ImageWidth', 'Image Width', 'Width'],
        'ImageHeight': ['ImageHeight', 'Image Height', 'Height'],
        'VideoScanType': ['VideoScanType', 'Video Scan Type', 'ScanType'],
        'DisplayWidth': ['DisplayWidth', 'Display Width'],
        'DisplayHeight': ['DisplayHeight', 'Display Height'],
        'DisplayUnit': ['DisplayUnit', 'Display Unit'],
        'AudioChannels': ['AudioChannels', 'Audio Channels', 'Channels'],
        'AudioSampleRate': ['AudioSampleRate', 'Audio Sample Rate', 'SampleRate'],
        'AudioBitsPerSample': ['AudioBitsPerSample', 'Audio Bits Per Sample', 'BitsPerSample'],
    }
    
    profile_fields = {}
    
    # Extract mapped fields
    for profile_field, possible_names in field_mappings.items():
        for name in possible_names:
            if name in exiftool_data:
                value = exiftool_data[name]
                # Convert numeric strings if needed
                if profile_field in ['ImageWidth', 'ImageHeight', 'DisplayWidth', 'DisplayHeight']:
                    value = str(value)
                elif profile_field in ['AudioSampleRate']:
                    value = str(int(value)) if isinstance(value, (int, float)) else str(value)
                elif profile_field in ['VideoFrameRate'] and isinstance(value, (int, float)):
                    value = str(value)
                    
                profile_fields[profile_field] = value
                break
    
    # Handle CodecID specially (it can be a single value or list)
    codec_id_names = ['CodecID', 'Codec ID', 'AudioCodecID', 'Audio Codec ID']
    codec_ids = []
    
    for name in codec_id_names:
        if name in exiftool_data:
            value = exiftool_data[name]
            if isinstance(value, list):
                codec_ids.extend(value)
            else:
                codec_ids.append(str(value))
    
    if codec_ids:
        # Remove duplicates while preserving order
        seen = set()
        unique_codec_ids = []
        for codec in codec_ids:
            if codec not in seen:
                seen.add(codec)
                unique_codec_ids.append(codec)
        profile_fields['CodecID'] = unique_codec_ids
    
    return profile_fields


def import_exiftool_file_to_profile(file_path: str, profile_name: Optional[str] = None) -> Optional[ExiftoolProfile]:
    """
    Import an exiftool output file and create an ExiftoolProfile from it.
    
    Args:
        file_path: Path to the exiftool output file
        profile_name: Optional name for the profile (defaults to filename)
        
    Returns:
        ExiftoolProfile instance or None if import fails
    """
    # Parse the file
    exiftool_data = parse_exiftool_file(file_path)
    if not exiftool_data:
        return None
    
    # Extract relevant fields
    profile_fields = extract_profile_fields(exiftool_data)
    
    if not profile_fields:
        logger.error("No relevant fields found in exiftool data")
        return None
    
    # Set default values for missing required fields
    defaults = {
        'FileType': 'Unknown',
        'FileTypeExtension': 'unknown',
        'MIMEType': 'application/octet-stream',
        'VideoFrameRate': '',
        'ImageWidth': '',
        'ImageHeight': '',
        'VideoScanType': '',
        'DisplayWidth': '',
        'DisplayHeight': '',
        'DisplayUnit': '',
        'CodecID': [],
        'AudioChannels': '',
        'AudioSampleRate': '',
        'AudioBitsPerSample': ''
    }
    
    # Merge with defaults
    for key, default_value in defaults.items():
        if key not in profile_fields:
            profile_fields[key] = default_value
    
    # Create the profile
    try:
        profile = ExiftoolProfile(**profile_fields)
        
        # Log what was imported
        logger.info(f"Successfully imported exiftool data from {file_path}")
        logger.debug(f"Imported fields: {list(profile_fields.keys())}")
        
        return profile
        
    except Exception as e:
        logger.error(f"Failed to create ExiftoolProfile: {e}")
        return None


def compare_with_expected(imported_data: Dict[str, any], expected_profile: ExiftoolProfile) -> Dict[str, Dict]:
    """
    Compare imported exiftool data with expected profile values.
    
    Args:
        imported_data: Dictionary of imported exiftool data
        expected_profile: The expected ExiftoolProfile to compare against
        
    Returns:
        Dict with 'matches', 'mismatches', and 'missing' keys
    """
    from dataclasses import asdict
    
    expected_dict = asdict(expected_profile)
    profile_fields = extract_profile_fields(imported_data)
    
    matches = {}
    mismatches = {}
    missing = {}
    
    for field, expected_value in expected_dict.items():
        if field in profile_fields:
            actual_value = profile_fields[field]
            
            # Special handling for lists (like CodecID)
            if isinstance(expected_value, list) and isinstance(actual_value, list):
                # Check if all expected codecs are present
                if set(expected_value).issubset(set(actual_value)):
                    matches[field] = {'expected': expected_value, 'actual': actual_value}
                else:
                    mismatches[field] = {'expected': expected_value, 'actual': actual_value}
            elif str(actual_value) == str(expected_value):
                matches[field] = {'expected': expected_value, 'actual': actual_value}
            else:
                mismatches[field] = {'expected': expected_value, 'actual': actual_value}
        elif expected_value:  # Only flag as missing if expected value is not empty
            missing[field] = {'expected': expected_value, 'actual': None}
    
    return {
        'matches': matches,
        'mismatches': mismatches,
        'missing': missing
    }


def validate_file_against_profile(file_path: str, profile: ExiftoolProfile) -> Dict[str, any]:
    """
    Validate an exiftool output file against an expected profile.
    
    Args:
        file_path: Path to the exiftool output file
        profile: The expected ExiftoolProfile
        
    Returns:
        Validation results dictionary
    """
    # Parse the file
    exiftool_data = parse_exiftool_file(file_path)
    if not exiftool_data:
        return {
            'valid': False,
            'error': f"Failed to parse {file_path}",
            'matches': {},
            'mismatches': {},
            'missing': {}
        }
    
    # Compare with expected
    comparison = compare_with_expected(exiftool_data, profile)
    
    # Determine if valid (no mismatches or missing required fields)
    is_valid = len(comparison['mismatches']) == 0 and len(comparison['missing']) == 0
    
    return {
        'valid': is_valid,
        'file': file_path,
        'total_fields': len(comparison['matches']) + len(comparison['mismatches']) + len(comparison['missing']),
        'matching_fields': len(comparison['matches']),
        **comparison
    }