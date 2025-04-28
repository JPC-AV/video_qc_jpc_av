#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import csv
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict

from ..utils.log_setup import logger
from ..utils.config_setup import ChecksConfig, SpexConfig
from ..utils.config_manager import ConfigManager


def parse_mediatrace(xml_file):
    config_mgr = ConfigManager()
    checks_config = config_mgr.get_config('checks', ChecksConfig)
    spex_config = config_mgr.get_config('spex', SpexConfig)
    expected_mediatrace = asdict(spex_config.mediatrace_values)
    expected_mt_keys = expected_mediatrace.keys()
    expected_encoder_settings = expected_mediatrace['ENCODER_SETTINGS']

    # Default values
    root = None
    mediatrace_output = {}
    
    try:
        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        # Log the error
        print(f"Warning: Could not parse XML file {xml_file}: {e}")
        return None  # Return early only if parsing fails
    except Exception as e:
        # Handle other potential errors
        print(f"Unexpected error processing {xml_file}: {e}")
        return None  # Return early only if parsing fails

    # If we get here, XML parsing was successful
    # Define the namespace
    ns = {'mt': 'https://mediaarea.net/mediatrace'}

    mediatrace_output = {}
    for mt_key in expected_mt_keys:
        # Find all 'block' elements with the name attribute matching 'SimpleTag'
        simple_tags = root.findall(".//mt:block[@name='SimpleTag']", ns)
        # Iterate through all 'block' elements name matching 'SimpleTag'
        for simple_tag in simple_tags:
            # Find the 'TagName' block with the specific mediatrace field name (key) we are looking
            tag_name_block = simple_tag.find(f".//mt:block[@name='TagName']/mt:data[.='{mt_key}']", ns)
            if tag_name_block is not None:
                # Find the corresponding 'TagString' block
                tag_string_block = simple_tag.find(f".//mt:block[@name='TagString']/mt:data", ns)
                if tag_string_block is not None:
                    mediatrace_output[mt_key] = tag_string_block.text
                    break

    mediatrace_differences = {}
    for expected_key, expected_value in expected_mediatrace.items():
        # defines variables "expected_key" and "expected_value" to the dictionary "expected_mediatrace"
        if expected_key not in mediatrace_output: 
            mediatrace_differences[expected_key] = ['metadata field not found', '']
        elif len(mediatrace_output[expected_key]) == 0:
            # count the values in the dictionary "mediatrace_output" with 'len', if the values are zero, then:
             mediatrace_differences[expected_key] =  ['no metadata value found', '']

    if 'ENCODER_SETTINGS' in mediatrace_output:
        # initialize variables
        encoder_settings_string = mediatrace_output['ENCODER_SETTINGS']
        # split string into list using semicolons as separators. I find reg expression confusing. For reference:
            # r: Indicates a raw string, where backslashes are treated literally (important for regular expressions).
            # \s*: Matches zero or more whitespace characters (spaces, tabs, newlines, etc.).
            # ;: Matches a semicolon character.
            # \s*: Again, matches zero or more whitespace characters.
        encoder_settings_list = re.split(r'\s*;\s*', encoder_settings_string)
        encoder_settings_dict = {}
        for encoder_settings_device in encoder_settings_list:
            # splits the string into a list based on either colons or commas, ignoring any surrounding whitespace
                # r: Indicates a raw string literal, where backslashes are treated as literal characters.
                # \s*: Matches zero or more whitespace characters (space, tab, newline, etc.).
                # :: Matches a colon character.
                # |: Represents an "OR" condition, meaning either the pattern to the left or the pattern to the right can match.
                # ,: Matches a comma character.
            device_field_name, *device_subfields_w_values = re.split(r'\s*:\s*|\s*,\s*', encoder_settings_device)
            # The first element of the resulting list is assigned to device_field_name
            # The remaining elements of the list (if any) are packed into the list device_subfields_w_values
            encoder_settings_dict[device_field_name] = device_subfields_w_values
        for expected_es_key, expected_es_value in expected_encoder_settings.items():
            # defines variables "expected_es_key" and "expected_es_value" to the dictionary "expected_mediatrace"
            if expected_es_key not in encoder_settings_dict:
                # append this string to the list "mediatrace_differences"
                mediatrace_differences[f"Encoder setting field {expected_es_key}"] = ['metadata field not found', '']
            elif set(encoder_settings_dict[expected_es_key]) != set(expected_es_value):
                # Compare as sets for order insensitivity  
                mediatrace_differences[expected_es_key] = [encoder_settings_dict[expected_es_key], expected_es_value]

    if not mediatrace_differences:
        # if the list "mediatrace_differences" is empty, then
        logger.info("All specified mediatrace fields and values found in output.\n")

    if mediatrace_differences:
        logger.critical("Some specified MediaTrace fields or values are missing or don't match:")
        for mediatrace_key, values in mediatrace_differences.items():
            actual_value, expected_value = values
            if expected_value == '':
                logger.critical(f"{mediatrace_key} {actual_value}")
            else:
                logger.critical(f"Encoder settings field {mediatrace_key} has a value of: {actual_value}\nThe expected value is: {expected_value}")
        logger.debug("")  # adding a space after mediatrace results if there are failures

    return mediatrace_differences


def write_to_csv(diff_dict, tool_name, writer):
    for key, values in diff_dict.items():
        actual_value, expected_value = values
        writer.writerow({
            'Metadata Tool': tool_name,
            'Metadata Field': key,
            'Expected Value': expected_value,
            'Actual Value': actual_value
        })
        

def create_metadata_difference_report(metadata_differences, report_directory, video_id):
    """
    Create a CSV report of metadata differences.
    
    Args:
        metadata_differences (dict): Dictionary of metadata differences from various tools
        report_directory (str): Directory to save the report
        video_id (str): Unique identifier for the video
        
    Returns:
        str or None: Path to the created CSV report, or None if no differences
    """
    # If no metadata differences, log and return
    if not metadata_differences:
        logger.info("All specified metadata fields and values found, no CSV report written\n")
        return None

    # Prepare CSV file path
    csv_name = f'{video_id}_metadata_difference.csv'
    diff_csv_path = os.path.join(report_directory, csv_name)

    try:
        # Define CSV header and open file
        fieldnames = ['Metadata Tool', 'Metadata Field', 'Expected Value', 'Actual Value']
        
        with open(diff_csv_path, 'w', newline='') as diffs_csv:
            writer = csv.DictWriter(diffs_csv, fieldnames=fieldnames)
            writer.writeheader()

            # Write differences for each tool
            tools_to_check = ['exiftool', 'mediainfo', 'mediatrace', 'ffprobe']
            for tool in tools_to_check:
                if tool in metadata_differences and metadata_differences[tool]:
                    write_to_csv(metadata_differences[tool], tool, writer)

        logger.info(f"Metadata difference report created: {diff_csv_path}\n")
        return diff_csv_path

    except IOError as e:
        logger.critical(f"Error creating metadata difference CSV: {e}")
        return None
