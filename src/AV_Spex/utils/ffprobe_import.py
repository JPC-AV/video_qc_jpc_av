"""
Utility functions for importing FFprobe data from JSON files.
Supports FFprobe JSON output format (ffprobe -print_format json).

Mirrors mediainfo_import.py but adapted for FFprobe's three-section
(video_stream/audio_stream/format) structure.
"""

import json
import os
import dataclasses
from typing import Dict, Optional, List, Any
from pathlib import Path
from dataclasses import asdict

from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import (
    FfprobeProfile, FFmpegVideoStream,
    FFmpegAudioStream, FFmpegFormat, EncoderSettings
)


def parse_ffprobe_json_file(file_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Parse an FFprobe JSON output file into section dictionaries.
    
    Handles the standard FFprobe JSON structure:
        {"streams": [{...}, {...}], "format": {...}}
    
    Args:
        file_path: Path to the FFprobe JSON file
        
    Returns:
        Dict with 'video_stream', 'audio_stream', 'format' keys containing
        track data, or None on failure
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
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
        
        ffprobe_data = json.loads(decoded_content)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None
    
    section_data = {"video_stream": {}, "audio_stream": {}, "format": {}}
    
    # Extract stream information
    if 'streams' in ffprobe_data:
        streams = ffprobe_data['streams']
        
        for stream in streams:
            codec_type = stream.get('codec_type')
            
            if codec_type == 'video' and not section_data["video_stream"]:
                section_data["video_stream"] = stream
            elif codec_type == 'audio':
                # For audio, handle the case of multiple audio streams
                if not section_data["audio_stream"]:
                    section_data["audio_stream"] = stream
                else:
                    # Merge multi-stream audio list fields (codec_name, codec_long_name)
                    for list_field in ['codec_name', 'codec_long_name']:
                        if list_field in stream:
                            existing = section_data["audio_stream"].get(list_field, "")
                            if isinstance(existing, list):
                                existing.append(stream[list_field])
                            else:
                                section_data["audio_stream"][list_field] = [existing, stream[list_field]]
    else:
        logger.error(f"No 'streams' key found in {file_path}")
        return None
    
    # Extract format information
    if 'format' in ffprobe_data:
        section_data["format"] = ffprobe_data['format']
    
    # Validate we got at least some data
    if not any(section_data.values()):
        logger.error(f"No valid FFprobe data found in {file_path}")
        return None
    
    return section_data


def _get_fields_for_dataclass(dataclass_type) -> List[str]:
    """Get field names from a dataclass type."""
    return [f.name for f in dataclasses.fields(dataclass_type)]


def extract_video_stream_fields(stream_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract video stream fields matching FFmpegVideoStream.
    
    Args:
        stream_data: Raw video stream data from FFprobe JSON
        
    Returns:
        Dict with fields matching FFmpegVideoStream
    """
    fields_to_extract = _get_fields_for_dataclass(FFmpegVideoStream)
    profile_fields = {}
    
    for field_name in fields_to_extract:
        if field_name in stream_data:
            profile_fields[field_name] = stream_data[field_name]
    
    return profile_fields


def extract_audio_stream_fields(stream_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract audio stream fields matching FFmpegAudioStream.
    
    Handles list fields (codec_name, codec_long_name) which may represent
    multiple audio streams.
    
    Args:
        stream_data: Raw audio stream data from FFprobe JSON
        
    Returns:
        Dict with fields matching FFmpegAudioStream
    """
    fields_to_extract = _get_fields_for_dataclass(FFmpegAudioStream)
    profile_fields = {}
    
    for field_name in fields_to_extract:
        if field_name in stream_data:
            value = stream_data[field_name]
            # Ensure list fields remain lists
            if field_name in ('codec_name', 'codec_long_name'):
                if not isinstance(value, list):
                    value = [value]
            profile_fields[field_name] = value
    
    return profile_fields


def extract_format_fields(format_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract format fields matching FFmpegFormat.
    
    Handles the tags sub-dictionary and ENCODER_SETTINGS.
    
    Args:
        format_data: Raw format data from FFprobe JSON
        
    Returns:
        Dict with fields matching FFmpegFormat
    """
    fields_to_extract = _get_fields_for_dataclass(FFmpegFormat)
    profile_fields = {}
    
    for field_name in fields_to_extract:
        if field_name == 'tags':
            # Handle tags specially — extract known tag fields
            if 'tags' in format_data:
                tags = format_data['tags']
                profile_tags = {}
                
                # Known tag keys from FFmpegFormat.tags default
                known_tag_keys = [
                    'creation_time', 'ENCODER', 'TITLE',
                    'ENCODER_SETTINGS', 'DESCRIPTION',
                    'ORIGINAL MEDIA TYPE', 'ENCODED_BY'
                ]
                
                for tag_key in known_tag_keys:
                    if tag_key in tags:
                        profile_tags[tag_key] = tags[tag_key]
                    else:
                        profile_tags[tag_key] = None
                
                profile_fields['tags'] = profile_tags
        elif field_name in format_data:
            profile_fields[field_name] = format_data[field_name]
    
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
            elif origin is dict or origin is typing.Dict:
                fields[field_name] = {}
            else:
                fields[field_name] = ""
    
    return fields


def import_ffprobe_file_to_profile(file_path: str) -> Optional[FfprobeProfile]:
    """
    Import an FFprobe JSON file and create a FfprobeProfile from it.
    
    This is the main entry point for file import, paralleling
    mediainfo_import.import_mediainfo_file_to_profile().
    
    Args:
        file_path: Path to the FFprobe JSON output file
        
    Returns:
        FfprobeProfile instance or None if import fails
    """
    section_data = parse_ffprobe_json_file(file_path)
    if not section_data:
        return None
    
    # Extract fields for each section
    video_fields = extract_video_stream_fields(section_data.get('video_stream', {}))
    audio_fields = extract_audio_stream_fields(section_data.get('audio_stream', {}))
    format_fields = extract_format_fields(section_data.get('format', {}))
    
    if not video_fields and not audio_fields and not format_fields:
        logger.error("No relevant fields found in FFprobe data")
        return None
    
    # Apply defaults for missing required fields
    video_fields = _apply_defaults(video_fields, FFmpegVideoStream)
    audio_fields = _apply_defaults(audio_fields, FFmpegAudioStream)
    format_fields = _apply_defaults(format_fields, FFmpegFormat)
    
    # Handle tags default for FFmpegFormat
    if 'tags' not in format_fields or not format_fields['tags']:
        format_fields['tags'] = {
            'creation_time': None,
            'ENCODER': None,
            'TITLE': None,
            'ENCODER_SETTINGS': None,
            'DESCRIPTION': None,
            'ORIGINAL MEDIA TYPE': None,
            'ENCODED_BY': None
        }
    
    try:
        video_stream = FFmpegVideoStream(**video_fields)
        audio_stream = FFmpegAudioStream(**audio_fields)
        ffmpeg_format = FFmpegFormat(**format_fields)
        
        profile = FfprobeProfile(
            video_stream=video_stream,
            audio_stream=audio_stream,
            format=ffmpeg_format
        )
        
        logger.info(f"Successfully imported FFprobe data from {file_path}")
        logger.debug(f"Video stream fields: {list(video_fields.keys())}")
        logger.debug(f"Audio stream fields: {list(audio_fields.keys())}")
        logger.debug(f"Format fields: {list(format_fields.keys())}")
        
        return profile
        
    except Exception as e:
        logger.error(f"Failed to create FfprobeProfile: {e}")
        return None


def compare_with_expected(imported_data: Dict[str, Dict],
                          expected_profile: FfprobeProfile) -> Dict[str, Dict]:
    """
    Compare imported FFprobe data with expected profile values.
    
    Returns per-section comparison results.
    
    Args:
        imported_data: Dict with 'video_stream', 'audio_stream', 'format'
                      keys containing extracted field dicts
        expected_profile: The expected FfprobeProfile to compare against
        
    Returns:
        Dict keyed by section name, each containing 'matches',
        'mismatches', and 'missing' sub-dicts
    """
    expected_dict = asdict(expected_profile)
    results = {}
    
    for section_name in ['video_stream', 'audio_stream', 'format']:
        section_expected = expected_dict.get(section_name, {})
        section_actual = imported_data.get(section_name, {})
        
        matches = {}
        mismatches = {}
        missing = {}
        
        for field_name, expected_value in section_expected.items():
            # Skip tags comparison (handled by signal flow system)
            if field_name == 'tags':
                continue
            
            if field_name in section_actual:
                actual_value = section_actual[field_name]
                
                # Normalize to list for comparison
                expected_list = expected_value if isinstance(expected_value, list) else [expected_value]
                
                # Handle list-to-list comparison
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
            elif expected_value and expected_value != "" and expected_value != []:
                # Only flag as missing if the expected value is non-empty
                missing[field_name] = {'expected': expected_value, 'actual': None}
        
        results[section_name] = {
            'matches': matches,
            'mismatches': mismatches,
            'missing': missing
        }
    
    return results


def validate_file_against_profile(file_path: str,
                                   profile: FfprobeProfile) -> Dict[str, Any]:
    """
    Validate an FFprobe JSON file against an expected profile.
    
    Args:
        file_path: Path to the FFprobe JSON output file
        profile: The expected FfprobeProfile
        
    Returns:
        Validation results dictionary with 'valid', 'sections', and
        aggregate counts
    """
    section_data = parse_ffprobe_json_file(file_path)
    if not section_data:
        return {
            'valid': False,
            'error': f"Failed to parse {file_path}",
            'sections': {}
        }
    
    # Extract profile-relevant fields from raw section data
    imported = {
        'video_stream': extract_video_stream_fields(section_data.get('video_stream', {})),
        'audio_stream': extract_audio_stream_fields(section_data.get('audio_stream', {})),
        'format': extract_format_fields(section_data.get('format', {}))
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
