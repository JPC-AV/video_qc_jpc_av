#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import subprocess
from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ChecksConfig
from AV_Spex.utils.config_manager import ConfigManager

config_mgr = ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)

def find_mediaconch_policy():
    try:
        policy_file = checks_config.tools.mediaconch.mediaconch_policy
        # Look in config/mediaconch_policies subdirectory
        policy_path = config_mgr.find_file(policy_file, os.path.join('config', 'mediaconch_policies'))
        
        if not policy_path:
            logger.critical(f'Policy file not found: {policy_file}')
            logger.critical('Make sure the file exists in the config/mediaconch_policies directory')
            return None
            
        return policy_path
    except Exception as e:
        logger.critical(f'Error finding MediaConch policy: {e}')
        return None


def run_mediaconch_command(command, input_path, output_type, output_path, policy_path):
    """
    Run MediaConch command with specified policy and input file.
    
    Args:
        command (str): Base MediaConch command
        input_path (str): Path to the input video file
        output_type (str): Output type flag (e.g., -oc for CSV)
        output_path (str): Path to save the output file
        policy_path (str): Path to the MediaConch policy file
        
    Returns:
        bool: True if command executed successfully, False otherwise
    """
    try:
        # Construct command as list (safer than shell=True)
        cmd_list = [
            command,
            '-p', policy_path,
            input_path,
            output_type,
            output_path
        ]
        
        logger.debug(f'Running command: {" ".join(cmd_list)}\n')
        
        # Run the command with shell=False (safer and handles spaces properly)
        result = subprocess.run(cmd_list, capture_output=True, text=True)
        
        # Check for command execution errors
        if result.returncode != 0:
            logger.error(f"MediaConch command failed: {result.stderr}")
            return False
        
        return True

    except Exception as e:
        logger.critical(f'Error running MediaConch command: {e}')
        return False

def parse_mediaconch_output(output_path):
    """
    Parse MediaConch CSV output and log policy validation results.
    
    Args:
        output_path (str): Path to the MediaConch CSV output file
        
    Returns:
        dict: Validation results with pass/fail status for each policy check
    """
    try:
        with open(output_path, 'r', newline='') as mc_file:
            reader = csv.reader(mc_file)
            mc_header = next(reader)  # Get the header row
            mc_values = next(reader)  # Get the values row

            # Create a dictionary to track validation results
            validation_results = {}
            found_failures = False

            # Zip headers and values to create key-value pairs
            for mc_field, mc_value in zip(mc_header, mc_values):
                validation_results[mc_field] = mc_value

                # Check for failures
                if mc_value == "fail":
                    if not found_failures:
                        logger.critical("MediaConch policy failed:")
                        found_failures = True
                    logger.critical(f"{mc_field}: {mc_value}")

            # Log overall validation status
            if not found_failures:
                logger.info("MediaConch policy passed\n")
            else:
                logger.debug("")  # Add empty line after mediaconch results

            return validation_results

    except FileNotFoundError:
        logger.critical(f"MediaConch output file not found: {output_path}\n")
        return {}
    except csv.Error as e:
        logger.critical(f"Error parsing MediaConch CSV: {e}")
        return {}
    except Exception as e:
        logger.critical(f"Unexpected error processing MediaConch output: {e}")
        return {}