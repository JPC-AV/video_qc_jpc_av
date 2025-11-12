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
    Any field can have multiple values (returned as a list).
    
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
        'CodecID': ['CodecID', 'Codec ID', 'AudioCodecID', 'Audio Codec ID'],
    }
    
    profile_fields = {}
    
    # Extract mapped fields - handle both single values and lists
    for profile_field, possible_names in field_mappings.items():
        values = []
        
        for name in possible_names:
            if name in exiftool_data:
                value = exiftool_data[name]
                
                # Handle list values
                if isinstance(value, list):
                    values.extend(value)
                else:
                    values.append(value)
                break  # Found the field, move to next profile_field
        
        if values:
            # Normalize and convert values
            normalized_values = []
            for val in values:
                # Convert numeric strings if needed
                if profile_field in ['ImageWidth', 'ImageHeight', 'DisplayWidth', 'DisplayHeight']:
                    normalized_values.append(str(val))
                elif profile_field in ['AudioSampleRate']:
                    normalized_values.append(str(int(val)) if isinstance(val, (int, float)) else str(val))
                elif profile_field in ['VideoFrameRate'] and isinstance(val, (int, float)):
                    normalized_values.append(str(val))
                else:
                    normalized_values.append(str(val))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_values = []
            for val in normalized_values:
                if val not in seen:
                    seen.add(val)
                    unique_values.append(val)
            
            # Store as list if multiple values, single string if one value
            if len(unique_values) > 1:
                profile_fields[profile_field] = unique_values
            elif len(unique_values) == 1:
                profile_fields[profile_field] = unique_values[0]
    
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
            
            # Normalize expected_value to always be a list for comparison
            expected_list = expected_value if isinstance(expected_value, list) else [expected_value]
            
            # Special handling for when both are lists (like CodecID)
            if isinstance(actual_value, list) and isinstance(expected_value, list):
                # Check if all expected values are present in actual values
                if set(expected_value).issubset(set(actual_value)):
                    matches[field] = {'expected': expected_value, 'actual': actual_value}
                else:
                    mismatches[field] = {'expected': expected_value, 'actual': actual_value}
            else:
                # Compare as strings, allowing for actual_value to match any item in expected_list
                actual_str = str(actual_value).strip()
                expected_str_list = [str(e).strip() for e in expected_list]
                
                if actual_str in expected_str_list:
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