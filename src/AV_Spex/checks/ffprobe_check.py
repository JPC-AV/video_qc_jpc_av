#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json

from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ChecksConfig, SpexConfig
from AV_Spex.utils.config_manager import ConfigManager


def parse_ffprobe(file_path):
    """
    Parse an FFprobe JSON file and compare against expected specifications.
    
    Loads expected values from the current spex configuration, which may
    have been set by a custom FFprobe profile.
    """
    # Load config fresh each time to pick up any profile changes
    config_mgr = ConfigManager()
    checks_config = config_mgr.get_config('checks', ChecksConfig)
    spex_config = config_mgr.get_config('spex', SpexConfig)

    # creates a dictionary of expected keys and values
    expected_video_values = spex_config.ffmpeg_values['video_stream']
    expected_audio_values = spex_config.ffmpeg_values['audio_stream']
    expected_format_values = spex_config.ffmpeg_values['format']

    if not os.path.exists(file_path):
        logger.critical(f"Cannot perform ffprobe check! No such file: {file_path}")
        return

    with open(file_path, 'r') as file:
        ffmpeg_data = json.load(file)

    # Now you can proceed with the rest of your code
    ffmpeg_output = {}

    ffmpeg_output['ffmpeg_video'] = ffmpeg_data['streams'][0]
    ffmpeg_output['ffmpeg_audio'] = ffmpeg_data['streams'][1]
    ffmpeg_output['format'] = ffmpeg_data['format']

    ffprobe_differences = {}
    
    # Check video stream fields
    for expected_key, expected_value in expected_video_values.items():
        # Skip fields with no expected value (e.g., imported as "")
        if not expected_value or expected_value == "" or expected_value == []:
            continue
        if expected_key in ffmpeg_output['ffmpeg_video']:
            actual_value = str(ffmpeg_output['ffmpeg_video'][expected_key]).strip()
            # Ensure expected_value is always a list for comparison
            expected_list = expected_value if isinstance(expected_value, list) else [expected_value]
            if actual_value not in expected_list:
                ffprobe_differences[expected_key] = [actual_value, expected_value]

    # Check audio stream fields
    for expected_key, expected_value in expected_audio_values.items():
        # Skip fields with no expected value (e.g., imported as "")
        if not expected_value or expected_value == "" or expected_value == []:
            continue
        if expected_key in ffmpeg_output['ffmpeg_audio']:
            actual_value = str(ffmpeg_output['ffmpeg_audio'][expected_key]).strip()
            # Ensure expected_value is always a list for comparison
            expected_list = expected_value if isinstance(expected_value, list) else [expected_value]
            if actual_value not in expected_list:
                ffprobe_differences[expected_key] = [actual_value, expected_value]

    # Check format fields
    for expected_key, expected_value in expected_format_values.items():
        # Skip tags - handled separately for encoder settings
        if expected_key == 'tags':
            continue
        # Skip fields with no expected value
        if not expected_value or expected_value == "" or expected_value == []:
            continue
        if expected_key not in (ffmpeg_output['format']):
            ffprobe_differences[expected_key] = ['metadata field not found', '']
        elif len(ffmpeg_output['format'][expected_key]) == 0:
            ffprobe_differences[expected_key] = ['no metadata value found', '']

    # Check format_name and format_long_name specifically
    if expected_format_values.get('format_name') and expected_format_values['format_name'] != "":
        actual_fmt = str(ffmpeg_output['format']['format_name']).replace(',', ' ')
        expected_fmt = str(expected_format_values['format_name']).replace(',', ' ')
        if expected_fmt not in actual_fmt:
            ffprobe_differences["Encoder setting 'format_name'"] = [ffmpeg_output['format']['format_name'], expected_format_values['format_name']]
    
    if expected_format_values.get('format_long_name') and expected_format_values['format_long_name'] != "":
        if expected_format_values['format_long_name'] not in ffmpeg_output['format']['format_long_name']:
            ffprobe_differences["Encoder setting 'format_long_name'"] = [ffmpeg_output['format']['format_long_name'], expected_format_values['format_long_name']]

    # Check for ENCODER_SETTINGS in format tags
    # This is handled by the signal flow profile system, but we still
    # check for its presence as a basic validation
    if 'ENCODER_SETTINGS' not in ffmpeg_output['format'].get('tags', {}):
        ffprobe_differences["Encoder Settings"] = ['No Encoder Settings found, No Signal Flow data embedded', '']

    if not ffprobe_differences:
        logger.info("All specified fields and values found in the ffprobe output.\n")
    else:
        logger.critical(f"Some specified ffprobe fields or values are missing or don't match:")
        for ffprobe_key, values in ffprobe_differences.items():
            actual_value, expected_value = values
            if ffprobe_key == 'ENCODER_SETTINGS':
                logger.critical(f"{actual_value}")
            elif expected_value == "":
                logger.critical(f"{ffprobe_key} {actual_value}")
            else:
                if isinstance(expected_value, list):
                    expected_display = ", ".join(str(v) for v in expected_value)
                else:
                    expected_display = expected_value
                logger.critical(f"Metadata field {ffprobe_key} has a value of: {actual_value}\nThe expected value is: {expected_display}")
        logger.debug('')

    return ffprobe_differences


# Only execute if this file is run directly, not imported
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <ffprobe_json_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"Error: {file_path} is not a valid file.")
        sys.exit(1)
    ffprobe_differences = parse_ffprobe(file_path)
    if ffprobe_differences:
        for diff in ffprobe_differences:
            logger.critical(f"\t{diff}")