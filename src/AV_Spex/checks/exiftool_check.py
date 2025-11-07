#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from dataclasses import asdict
from typing import List

from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ChecksConfig, SpexConfig, ExiftoolValues
from AV_Spex.utils.config_manager import ConfigManager


def parse_exiftool(file_path):
    exif_data_dict = parse_exiftool_json(file_path)
    exiftool_differences = check_exif_spex(exif_data_dict)

    return exiftool_differences


def get_expected_fields() -> List[str]:
    """
    Get the list of expected fields from ExiftoolValues dataclass.
    
    Returns:
        List of field names to extract
    """
    # Import required for introspection
    import dataclasses
    
    # Get field names from the ExiftoolValues dataclass
    return [field.name for field in dataclasses.fields(ExiftoolValues)]


def parse_exiftool_json(file_path):
    '''
    This function uses a dictionary (key:value pairs) defined in config/config.yaml 
    to check the values of an exiftool output against expected values.
    This function is called in the process_file.py script and is used to check exiftool output.
    '''

    if not os.path.exists(file_path):
        logger.critical(f"Cannot perform exiftool check! No such file: {file_path}")
        return {}

    try:
        with open(file_path, 'r') as file:
            # Parse the JSON data directly into a Python dictionary
            exif_data = json.load(file)
            
            # If exiftool outputs JSON as an array of objects, get the first object
            if isinstance(exif_data, list) and len(exif_data) > 0:
                exif_data = exif_data[0]
                
        return exif_data
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {file_path}: {e}")
        return {}


def check_exif_spex(exif_data):
    """
    Compare ExifTool values with expected specifications.
    
    Args:
        exif_data: Dictionary with ExifTool metadata
        
    Returns:
        Dictionary of differences between actual and expected values
    """
    config_mgr = ConfigManager()
    checks_config = config_mgr.get_config('checks', ChecksConfig)
    spex_config = config_mgr.get_config('spex', SpexConfig)
    
    # Safely get expected values, defaulting to empty dictionary if not present
    expected_exif = asdict(spex_config.exiftool_values)
    
    exiftool_differences = {}

    # Check all keys in expected_exif
    for expected_key, expected_value in expected_exif.items():
        if expected_key in exif_data:
            actual_value = exif_data[expected_key]
            
            # Ensure expected_value is always a list for comparison
            expected_list = expected_value if isinstance(expected_value, list) else [expected_value]
            
            # Try to normalize types for comparison
            # Convert both actual and expected values to strings for comparison
            actual_str = str(actual_value).strip()
            expected_str_list = [str(e).strip() for e in expected_list]
            
            if actual_str not in expected_str_list:
                exiftool_differences[expected_key] = [actual_value, expected_value]

    # Log results
    if not exiftool_differences:
        logger.info("All specified fields and values found in the ExifTool output.\n")
    else:
        logger.critical("Some specified ExifTool fields or values are missing or don't match:")
        for exif_key, values in exiftool_differences.items():
            actual_value, expected_value = values
            logger.critical(f"Metadata field {exif_key} has a value of: {actual_value}\nThe expected value is: {expected_value}")
        logger.debug("")  # adding a space after results if there are failures

    return exiftool_differences


# Only execute if this file is run directly, not imported)
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <exiftool_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"Error: {file_path} is not a valid file.")
        sys.exit(1)
        # assigns variable "file_path" to the text file "JPCspecs_ex.txt"
    exiftool_differences = parse_exiftool(file_path)
    #runs the function "parse_exiftool" on the file assigned to the variable "file_path"
    if exiftool_differences:
        for diff in exiftool_differences:
            logger.critical(f"\t{diff}")