#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from typing import Dict, Any, Optional, List

from ..utils.log_setup import logger
from ..utils.config_setup import ChecksConfig, SpexConfig
from ..utils.config_manager import ConfigManager

config_mgr = ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)
spex_config = config_mgr.get_config('spex', SpexConfig)


def parse_mediainfo(file_path):
    section_data = parse_mediainfo_json(file_path)
    mediainfo_differences = check_mediainfo_spex(section_data)

    return mediainfo_differences


def parse_mediainfo_json(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse a MediaInfo JSON file and extract relevant information organized by section.
    
    Args:
        file_path: Path to the MediaInfo JSON file
        
    Returns:
        Dictionary with General, Video, and Audio sections containing key-value pairs
    """
    section_data = {"General": {}, "Video": {}, "Audio": {}}
    
    if not os.path.exists(file_path):
        logger.critical(f"Cannot perform MediaInfo check! No such file: {file_path}")
        return section_data
    
    try:
        # Read the file in binary mode first to handle encoding issues
        with open(file_path, 'rb') as file:
            content = file.read()
        
        # Try to decode with utf-8 first, with error reporting
        try:
            decoded_content = content.decode('utf-8')
        except UnicodeDecodeError as e:
            logger.error(f"UTF-8 decoding error in {file_path}: {e}")
            # Try with latin-1 as a fallback, which can handle any byte
            try:
                decoded_content = content.decode('latin-1')
                logger.warning(f"Used latin-1 encoding as fallback for {file_path}\n")
            except Exception as e2:
                logger.error(f"Failed to decode {file_path} with fallback encoding: {e2}\n")
                return section_data
        
        # Parse the JSON
        try:
            mediainfo = json.loads(decoded_content)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            return section_data
                
        # Extract track information from the JSON structure
        if 'media' in mediainfo and 'track' in mediainfo['media']:
            tracks = mediainfo['media']['track']
            
            for track in tracks:
                track_type = track.get('@type')
                
                if track_type == 'General':
                    section_data["General"] = extract_general_data(track)
                elif track_type == 'Video':
                    section_data["Video"] = extract_video_data(track)
                elif track_type == 'Audio':
                    section_data["Audio"] = extract_audio_data(track)
        else:
            logger.error(f"Expected JSON structure not found in {file_path}")
            
    except Exception as e:
        logger.error(f"Unexpected error processing {file_path}: {e}")
        
    # Validate that we got meaningful data
    if not section_data["Video"] and not section_data["Audio"] and not section_data["General"]:
        logger.error(f"No valid MediaInfo data extracted from {file_path}")
        
    return section_data


def get_expected_fields(section_type: str) -> List[str]:
    """
    Get the list of expected fields from config_setup dataclasses based on the section type.
    
    Args:
        section_type: Type of configuration ('General', 'Video', or 'Audio')
        
    Returns:
        List of field names to extract
    """
    # Import required for introspection
    import dataclasses
    from ..utils.config_setup import MediainfoGeneralValues, MediainfoVideoValues, MediainfoAudioValues
    
    # Map section types to their corresponding dataclass types
    dataclass_mapping = {
        "General": MediainfoGeneralValues,
        "Video": MediainfoVideoValues,
        "Audio": MediainfoAudioValues
    }
    
    if section_type in dataclass_mapping:
        # Get the dataclass corresponding to this section
        dataclass_type = dataclass_mapping[section_type]
        
        # Get field names from the dataclass
        return [field.name for field in dataclasses.fields(dataclass_type)]
    
    return []


def extract_general_data(track: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant General track data based on fields in the config"""
    general_data = {}
    
    # Get expected fields from config instead of hardcoding
    fields_to_extract = get_expected_fields('General')
    
    for field in fields_to_extract:
        if field in track:
            general_data[field] = track[field]
    
    # Handle extra fields in General track
    if "extra" in track:
        extra = track["extra"]
        if "ErrorDetectionType" in extra and "ErrorDetectionType" in fields_to_extract:
            general_data["ErrorDetectionType"] = extra["ErrorDetectionType"]
    
    return general_data


def extract_video_data(track: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant Video track data based on fields in the config"""
    video_data = {}
    
    # Get expected fields from config instead of hardcoding
    fields_to_extract = get_expected_fields('Video')
    
    for field in fields_to_extract:
        if field in track:
            video_data[field] = track[field]
    
    # Handle special cases from extra field
    if "extra" in track:
        extra = track["extra"]
        # Check if these fields are expected based on the config
        if "MaxSlicesCount" in extra and "MaxSlicesCount" in fields_to_extract:
            video_data["MaxSlicesCount"] = extra["MaxSlicesCount"]
        if "ErrorDetectionType" in extra and "ErrorDetectionType" in fields_to_extract:
            video_data["ErrorDetectionType"] = extra["ErrorDetectionType"]
    
    return video_data


def extract_audio_data(track: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant Audio track data based on fields in the config"""
    audio_data = {}
    
    # Get expected fields from config instead of hardcoding
    fields_to_extract = get_expected_fields('Audio')
    
    for field in fields_to_extract:
        if field in track:
            audio_data[field] = track[field]
    
    return audio_data


def check_mediainfo_spex(section_data):
    """
    Compare MediaInfo values with expected specifications.
    
    Args:
        section_data: Dictionary with General, Video, and Audio sections
        
    Returns:
        Dictionary of differences between actual and expected values
    """
    mediainfo_differences = {}
    
    

    expected_general = spex_config.mediainfo_values.get("expected_general", {})
    expected_video = spex_config.mediainfo_values.get("expected_video", {})
    expected_audio = spex_config.mediainfo_values.get("expected_audio", {})
        
    #logger.debug(f"Expected general type: {type(expected_general)}")
    #logger.debug(f"Expected video type: {type(expected_video)}")
    #logger.debug(f"Expected audio type: {type(expected_audio)}")
    
    # Check General section
    if "General" in section_data and hasattr(expected_general, "items"):
        for expected_key, expected_value in expected_general.items():
            if expected_key in section_data["General"]:
                actual_value = section_data["General"][expected_key]
                # Ensure expected_value is always a list for comparison
                expected_list = expected_value if isinstance(expected_value, list) else [expected_value]
                if actual_value not in expected_list:
                    mediainfo_differences[expected_key] = [actual_value, expected_value]
    
    # Check Video section
    if "Video" in section_data:
        for expected_key, expected_value in expected_video.items():
            if expected_key in section_data["Video"]:
                actual_value = section_data["Video"][expected_key]
                # Ensure expected_value is always a list for comparison
                expected_list = expected_value if isinstance(expected_value, list) else [expected_value]
                if actual_value not in expected_list:
                    mediainfo_differences[expected_key] = [actual_value, expected_value]
    
    # Check Audio section
    if "Audio" in section_data:
        for expected_key, expected_value in expected_audio.items():
            if expected_key in section_data["Audio"]:
                actual_value = section_data["Audio"][expected_key]
                # Ensure expected_value is always a list for comparison
                expected_list = expected_value if isinstance(expected_value, list) else [expected_value]
                if actual_value not in expected_list:
                    mediainfo_differences[expected_key] = [actual_value, expected_value]

    # Log results
    if not mediainfo_differences:
        logger.info("All specified fields and values found in the MediaInfo output.\n")
    else:
        logger.critical("Some specified MediaInfo fields or values are missing or don't match:")
        for mi_key, values in mediainfo_differences.items():
            actual_value, expected_value = values
            logger.critical(f"Metadata field {mi_key} has a value of: {actual_value}\nThe expected value is: {expected_value}")
        logger.debug("")  # adding a space after results if there are failures

    return mediainfo_differences


# Only execute if this file is run directly, not imported
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <mediainfo_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"Error: {file_path} is not a valid file.")
        sys.exit(1)
    mediainfo_differences = parse_mediainfo(file_path)
    if mediainfo_differences:
        for diff in mediainfo_differences:
            logger.critical(f"\t{diff}")