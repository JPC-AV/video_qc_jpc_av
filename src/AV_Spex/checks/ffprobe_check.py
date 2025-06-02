#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json

from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ChecksConfig, SpexConfig
from AV_Spex.utils.config_manager import ConfigManager

config_mgr = ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)
spex_config = config_mgr.get_config('spex', SpexConfig)


## creates the function 'parse_exiftool' which takes the argument 'file_path' 
# the majority of this script is defining this function. But the function is not run until the last line fo the script
def parse_ffprobe(file_path):
    # creates a dictionary of expected keys and values
    expected_video_values = spex_config.ffmpeg_values['video_stream']
    expected_audio_values = spex_config.ffmpeg_values['audio_stream']
    expected_format_values = spex_config.ffmpeg_values['format']
    expected_settings_values = spex_config.ffmpeg_values['format']['tags']['ENCODER_SETTINGS']

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
    # Create empty list, "ffprobe_differences"
    for expected_key, expected_value in expected_video_values.items():
        # defines variables "expected_key" and "expected_value" to the dictionary "expected_general"
        if expected_key in ffmpeg_output['ffmpeg_video']:
            # if the key in the dictionary "General"
            actual_value = str(ffmpeg_output['ffmpeg_video'][expected_key]).strip()
            # assigns the variable "actual_value" to the value that matches the key in the dictionary "General"
            # I'm not sure if this should be "key" or "expected_key" honestly. Perhaps there should be an additional line for if key = expected_key or something?
            if actual_value not in expected_value:
                # if variable "actual_value" does not match "expected value" defined in first line as the values from the dictionary expected_general, then
                ffprobe_differences[expected_key] = [actual_value, expected_value]

    for expected_key, expected_value in expected_audio_values.items():
        # defines variables "expected_key" and "expected_value" to the dictionary "expected_video"
        if expected_key in ffmpeg_output['ffmpeg_audio']:
            # if the key in the dictionary "Video"
            actual_value = str(ffmpeg_output['ffmpeg_audio'][expected_key]).strip()
            # assigns the variable "actual_value" to the value that matches the key in the dictionary "Video"
            # I'm not sure if this should be "key" or "expected_key" honestly. Perhaps there should be an additional line for if key = expected_key or something?
            if actual_value not in expected_value:
                # if variable "actual_value" does not match "expected value" defined in first line as the values from the dictionary expected_video, then
                ffprobe_differences[expected_key] = [actual_value, expected_value]

    for expected_key, expected_value in expected_format_values.items():
        # defines variables "expected_key" and "expected_value" to the dictionary "expected_audio"
        if expected_key not in (ffmpeg_output['format']):
            ffprobe_differences[expected_key] = ['metadata field not found', '']
        elif len(ffmpeg_output['format'][expected_key]) == 0:
            # count the values in the nested dictionary "format" with 'len', if the values are zero, then:
            ffprobe_differences[expected_key] = ['no metadata value found', '']

    if expected_format_values['format_name'] not in str(ffmpeg_output['format']['format_name']).replace(',', ' '):
        ffprobe_differences["Encoder setting 'format_name'"] = [ffmpeg_output['format']['format_name'], expected_format_values['format_name']]
    if expected_format_values['format_long_name'] not in ffmpeg_output['format']['format_long_name']:
        ffprobe_differences["Encoder setting 'format_long_name'"] = [ffmpeg_output['format']['format_long_name'], expected_format_values['format_long_name']]

    if 'ENCODER_SETTINGS' not in ffmpeg_output['format']['tags']:
        ffprobe_differences["Encoder Settings"] = ['No Encoder Settings found, No Signal Flow data embedded', '']

    if not ffprobe_differences:
        # if the list "ffprobe_differences" is empty, then
        logger.info("All specified fields and values found in the ffprobe output.\n")
    else:
        # if the list "ffprobe_differences" is not empty, then
        logger.critical(f"Some specified ffprobe fields or values are missing or don't match:")
        for ffprobe_key, values in ffprobe_differences.items():
            actual_value, expected_value = values
            if ffprobe_key == 'ENCODER_SETTINGS':
                # This exception is for if there are no encoder settings embedded (cleaner output)
                logger.critical(f"{actual_value}")
            elif expected_value == "":
                # This exception is for if there are missing subfields inside encoder settings (cleaner output)
                logger.critical(f"{ffprobe_key} {actual_value}")
            else:    
                logger.critical(f"Metadata field {ffprobe_key} has a value of: {actual_value}\nThe expected value is: {expected_value}")
        logger.debug('')


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