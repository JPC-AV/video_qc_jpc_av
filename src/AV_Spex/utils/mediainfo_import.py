"""
Utility functions for importing MediaInfo data from JSON files.
Supports MediaInfo JSON output format (--Output=JSON).

Mirrors exiftool_import.py but adapted for MediaInfo's three-section
(General/Video/Audio) structure.
"""

import json
import os
import dataclasses
from typing import Dict, Optional, List, Any
from pathlib import Path
from dataclasses import asdict

from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import (
    MediainfoProfile, MediainfoGeneralValues,
    MediainfoVideoValues, MediainfoAudioValues
)


def parse_mediainfo_json_file(file_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Parse a MediaInfo JSON output file into section dictionaries.
    
    Handles the standard MediaInfo JSON structure:
        {"media": {"track": [{"@type": "General", ...}, {"@type": "Video", ...}, ...]}}
    
    Args:
        file_path: Path to the MediaInfo JSON file
        
    Returns:
        Dict with 'General', 'Video', 'Audio' keys containing track data,
        or None on failure
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        # Read in binary to handle encoding issues (same approach as mediainfo_check)
        with open(file_path, 'rb') as f:
            content = f.read()
        
        try:
            decoded_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                decoded_content = content.decode('latin-1')
                logger.warning(f"Used latin-1 encoding as fallback for {file_path}")
            except Exception as e:
                logger.error(f"Failed to decode {file_path}: {e}")
                return None
        
        mediainfo = json.loads(decoded_content)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None
    
    section_data = {"General": {}, "Video": {}, "Audio": {}}
    
    # Extract track information from the JSON structure
    if 'media' in mediainfo and 'track' in mediainfo['media']:
        tracks = mediainfo['media']['track']
        
        for track in tracks:
            track_type = track.get('@type')
            
            if track_type == 'General':
                section_data["General"] = track
            elif track_type == 'Video':
                section_data["Video"] = track
            elif track_type == 'Audio':
                section_data["Audio"] = track
    else:
        logger.error(f"Expected MediaInfo JSON structure not found in {file_path}")
        return None
    
    # Validate we got at least some data
    if not any(section_data.values()):
        logger.error(f"No valid MediaInfo data found in {file_path}")
        return None
    
    return section_data


def _get_fields_for_dataclass(dataclass_type) -> List[str]:
    """Get field names from a dataclass type."""
    return [f.name for f in dataclasses.fields(dataclass_type)]


def extract_general_profile_fields(track_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract General section fields matching MediainfoGeneralValues.
    
    Args:
        track_data: Raw General track data from MediaInfo JSON
        
    Returns:
        Dict with fields matching MediainfoGeneralValues
    """
    fields_to_extract = _get_fields_for_dataclass(MediainfoGeneralValues)
    profile_fields = {}
    
    for field_name in fields_to_extract:
        if field_name in track_data:
            profile_fields[field_name] = track_data[field_name]
    
    # Handle extra fields
    if "extra" in track_data:
        extra = track_data["extra"]
        if "ErrorDetectionType" in extra and "ErrorDetectionType" in fields_to_extract:
            profile_fields["ErrorDetectionType"] = extra["ErrorDetectionType"]
    
    return profile_fields


def extract_video_profile_fields(track_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Video section fields matching MediainfoVideoValues.
    
    Handles special cases from the 'extra' sub-dict (MaxSlicesCount,
    ErrorDetectionType) matching mediainfo_check.extract_video_data().
    
    Args:
        track_data: Raw Video track data from MediaInfo JSON
        
    Returns:
        Dict with fields matching MediainfoVideoValues
    """
    fields_to_extract = _get_fields_for_dataclass(MediainfoVideoValues)
    profile_fields = {}
    
    for field_name in fields_to_extract:
        if field_name in track_data:
            profile_fields[field_name] = track_data[field_name]
    
    # Handle special cases from extra field
    if "extra" in track_data:
        extra = track_data["extra"]
        if "MaxSlicesCount" in extra and "MaxSlicesCount" in fields_to_extract:
            profile_fields["MaxSlicesCount"] = extra["MaxSlicesCount"]
        if "ErrorDetectionType" in extra and "ErrorDetectionType" in fields_to_extract:
            profile_fields["ErrorDetectionType"] = extra["ErrorDetectionType"]
    
    return profile_fields


def extract_audio_profile_fields(track_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Audio section fields matching MediainfoAudioValues.
    
    Args:
        track_data: Raw Audio track data from MediaInfo JSON
        
    Returns:
        Dict with fields matching MediainfoAudioValues
    """
    fields_to_extract = _get_fields_for_dataclass(MediainfoAudioValues)
    profile_fields = {}
    
    for field_name in fields_to_extract:
        if field_name in track_data:
            value = track_data[field_name]
            profile_fields[field_name] = value
    
    return profile_fields


def _apply_defaults(fields: Dict[str, Any], dataclass_type) -> Dict[str, Any]:
    """
    Fill in default values for any fields missing from extracted data.
    
    Uses empty string for str fields and empty list for List fields.
    
    Args:
        fields: Extracted field values
        dataclass_type: The target dataclass type for introspection
        
    Returns:
        Fields dict with defaults applied
    """
    import typing
    type_hints = typing.get_type_hints(dataclass_type)
    
    for field_info in dataclasses.fields(dataclass_type):
        field_name = field_info.name
        if field_name not in fields:
            field_type = type_hints.get(field_name)
            origin = typing.get_origin(field_type)
            
            if origin is list or origin is typing.List:
                fields[field_name] = []
            else:
                fields[field_name] = ""
    
    return fields


def import_mediainfo_file_to_profile(file_path: str) -> Optional[MediainfoProfile]:
    """
    Import a MediaInfo JSON file and create a MediainfoProfile from it.
    
    This is the main entry point for file import, paralleling
    exiftool_import.import_exiftool_file_to_profile().
    
    Args:
        file_path: Path to the MediaInfo JSON output file
        
    Returns:
        MediainfoProfile instance or None if import fails
    """
    section_data = parse_mediainfo_json_file(file_path)
    if not section_data:
        return None
    
    # Extract fields for each section
    general_fields = extract_general_profile_fields(section_data.get('General', {}))
    video_fields = extract_video_profile_fields(section_data.get('Video', {}))
    audio_fields = extract_audio_profile_fields(section_data.get('Audio', {}))
    
    if not general_fields and not video_fields and not audio_fields:
        logger.error("No relevant fields found in MediaInfo data")
        return None
    
    # Apply defaults for missing required fields
    general_fields = _apply_defaults(general_fields, MediainfoGeneralValues)
    video_fields = _apply_defaults(video_fields, MediainfoVideoValues)
    audio_fields = _apply_defaults(audio_fields, MediainfoAudioValues)
    
    try:
        general = MediainfoGeneralValues(**general_fields)
        video = MediainfoVideoValues(**video_fields)
        audio = MediainfoAudioValues(**audio_fields)
        
        profile = MediainfoProfile(general=general, video=video, audio=audio)
        
        logger.info(f"Successfully imported MediaInfo data from {file_path}")
        logger.debug(f"General fields: {list(general_fields.keys())}")
        logger.debug(f"Video fields: {list(video_fields.keys())}")
        logger.debug(f"Audio fields: {list(audio_fields.keys())}")
        
        return profile
        
    except Exception as e:
        logger.error(f"Failed to create MediainfoProfile: {e}")
        return None


def compare_with_expected(imported_data: Dict[str, Dict], 
                          expected_profile: MediainfoProfile) -> Dict[str, Dict]:
    """
    Compare imported MediaInfo data with expected profile values.
    
    Returns per-section comparison results.
    
    Args:
        imported_data: Dict with 'general', 'video', 'audio' keys
                      containing extracted field dicts
        expected_profile: The expected MediainfoProfile to compare against
        
    Returns:
        Dict keyed by section name, each containing 'matches',
        'mismatches', and 'missing' sub-dicts
    """
    expected_dict = asdict(expected_profile)
    results = {}
    
    for section_name in ['general', 'video', 'audio']:
        section_expected = expected_dict.get(section_name, {})
        section_actual = imported_data.get(section_name, {})
        
        matches = {}
        mismatches = {}
        missing = {}
        
        for field_name, expected_value in section_expected.items():
            if field_name in section_actual:
                actual_value = section_actual[field_name]
                
                # Normalize to list for comparison
                expected_list = expected_value if isinstance(expected_value, list) else [expected_value]
                
                # Handle list-to-list comparison (e.g., Audio Format)
                if isinstance(actual_value, list) and isinstance(expected_value, list):
                    if set(expected_value).issubset(set(actual_value)):
                        matches[field_name] = {'expected': expected_value, 'actual': actual_value}
                    else:
                        mismatches[field_name] = {'expected': expected_value, 'actual': actual_value}
                else:
                    actual_str = str(actual_value).strip()
                    expected_str_list = [str(e).strip() for e in expected_list]
                    
                    if actual_str in expected_str_list:
                        matches[field_name] = {'expected': expected_value, 'actual': actual_value}
                    else:
                        mismatches[field_name] = {'expected': expected_value, 'actual': actual_value}
            elif expected_value:
                # Only flag as missing if the expected value is non-empty
                missing[field_name] = {'expected': expected_value, 'actual': None}
        
        results[section_name] = {
            'matches': matches,
            'mismatches': mismatches,
            'missing': missing
        }
    
    return results


def validate_file_against_profile(file_path: str, 
                                   profile: MediainfoProfile) -> Dict[str, Any]:
    """
    Validate a MediaInfo JSON file against an expected profile.
    
    Args:
        file_path: Path to the MediaInfo JSON output file
        profile: The expected MediainfoProfile
        
    Returns:
        Validation results dictionary with 'valid', 'sections', and
        aggregate counts
    """
    section_data = parse_mediainfo_json_file(file_path)
    if not section_data:
        return {
            'valid': False,
            'error': f"Failed to parse {file_path}",
            'sections': {}
        }
    
    # Extract profile-relevant fields from raw section data
    imported = {
        'general': extract_general_profile_fields(section_data.get('General', {})),
        'video': extract_video_profile_fields(section_data.get('Video', {})),
        'audio': extract_audio_profile_fields(section_data.get('Audio', {}))
    }
    
    comparison = compare_with_expected(imported, profile)
    
    # Aggregate across all sections
    total_matches = sum(len(s['matches']) for s in comparison.values())
    total_mismatches = sum(len(s['mismatches']) for s in comparison.values())
    total_missing = sum(len(s['missing']) for s in comparison.values())
    total_fields = total_matches + total_mismatches + total_missing
    
    return {
        'valid': total_mismatches == 0 and total_missing == 0,
        'file': file_path,
        'total_fields': total_fields,
        'matching_fields': total_matches,
        'sections': comparison
    }