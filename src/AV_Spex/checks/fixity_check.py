import os
import sys
import hashlib
import shutil
import re
from datetime import datetime
from AV_Spex.utils.log_setup import logger


def check_fixity(directory, video_id, actual_checksum=None, check_cancelled=None, signals=None):
    if check_cancelled():
        return None
    
    fixity_result_file = os.path.join(directory, f'{video_id}_qc_metadata', f'{video_id}_{datetime.now().strftime("%Y_%m_%d_%H_%M")}_fixity_check.txt')

    # Store paths to checksum files
    checksum_files = []  

    # Walk files of the source directory looking for file with '_checksums.md5' or '_fixity.txt' suffix
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Look for date pattern before "_fixity.txt" or "_checksums.md5"
            if file.endswith('_checksums.md5') or file.endswith('_fixity.txt'):
                checksum_file_path = os.path.join(root, file)
                try:
                    # Remove the suffix first
                    if file.endswith('_checksums.md5'):
                        base_name = file.replace('_checksums.md5', '')
                    else:  # file.endswith('_fixity.txt')
                        base_name = file.replace('_fixity.txt', '')
                    
                    # Use regex to find date patterns at the end of the base name
                    # Pattern 1: YYYY_MM_DD_HH_MM at the end of string ($ is a special character in regex that matches the end of the string)
                    date_match = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2})$', base_name)
                    if date_match:
                        date_str = date_match.group(1)
                        file_date = datetime.strptime(date_str, "%Y_%m_%d_%H_%M").date()
                    else:
                        # Pattern 2: YYYY_MM_DD at the end of string
                        date_match = re.search(r'(\d{4}_\d{2}_\d{2})$', base_name)
                        if date_match:
                            date_str = date_match.group(1)
                            file_date = datetime.strptime(date_str, "%Y_%m_%d").date()
                        else:
                            raise ValueError(f"No date pattern found in filename: {file}")
                    
                    checksum_files.append((checksum_file_path, file_date))
                
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping checksum file with invalid date format: {file}. Error: {str(e)}")

    # Sort checksum files by date (descending)
    checksum_files.sort(key=lambda x: x[1], reverse=True)

    if not checksum_files:
        logger.error("Unable to validate fixity against previous md5 checksum. No file ending in '_checksums.md5' or '_fixity.txt' found.\n")

    video_file_path = os.path.join(directory, f'{video_id}.mkv')

    if check_cancelled():
        return None
    
    # If video file exists, then:
    if os.path.exists(video_file_path):
        # If checksum has not yet been calculated, then:
        if not checksum_files and actual_checksum is None:
            output_fixity(directory, video_file_path, check_cancelled=check_cancelled, signals=signals)
            return
        elif checksum_files and actual_checksum is None:
            # Calculate the MD5 checksum of the video file
            actual_checksum = hashlib_md5(video_file_path, check_cancelled=check_cancelled, signals=signals)
    else:
        logger.critical(f'Video file not found: {video_file_path}')
        return

    # initialize variables
    checksums_match = True  
    most_recent_checksum = None
    most_recent_checksum_date = None
    # collision_found = False

    for checksum_file_path, file_date in checksum_files:
        # Read the MD5 checksum from the _checksums.md5 file
        expected_checksum = read_checksum_from_file(checksum_file_path)

        # Update most recent checksum if this one is newer
        if most_recent_checksum_date is None or file_date > most_recent_checksum_date:
            most_recent_checksum = expected_checksum
            most_recent_checksum_date = file_date

        if actual_checksum != expected_checksum:
            checksums_match = False
            # collision_found = True # not currently using this, but may want to acknowledge mismatch, even if most recent checksum matches

    if checksums_match:
        logger.info(f'Fixity check passed for {video_file_path}\n')
        result_file = open(fixity_result_file, 'w')
        print(f'Fixity check passed for {video_file_path}\n', file = result_file)
        result_file.close()
    else:
        logger.critical(f'Fixity check failed for {video_file_path}\n')
        logger.critical(f'Checksum read from {most_recent_checksum_date} .md5 file is: {expected_checksum}\nChecksum created now from MKV file = {actual_checksum}\n')
        result_file = open(fixity_result_file, 'w')
        print(f'Fixity check failed for {os.path.basename(video_file_path)} checksum read from .md5 file = {expected_checksum} checksum created from MKV file = {actual_checksum}\n', file = result_file)
        result_file.close()


def output_fixity(source_directory, video_path, check_cancelled=None, signals=None):
    # Parse video_id from video file path
    video_id = os.path.splitext(os.path.basename(os.path.basename(video_path)))[0]
    # Create fixity results files
    fixity_result_file = os.path.join(source_directory, f'{video_id}_{datetime.now().strftime("%Y_%m_%d_%H_%M")}_fixity.txt')
    fixity_md5_file = os.path.join(source_directory, f'{video_id}_{datetime.now().strftime("%Y_%m_%d_%H_%M")}_fixity.md5')

    if check_cancelled():
        return None
    
    # Calculate the MD5 checksum of the video file
    md5_checksum = hashlib_md5(video_path, check_cancelled=check_cancelled, signals=signals)
    if md5_checksum is None:  # Handle cancelled case
        return None
    
    if check_cancelled():
        return None
    
    # Open fixity_result_file
    result_file = open(fixity_result_file, 'w')
    # Print Md5 in 'filename[tab]Checksum' format
    print(f'{md5_checksum}  {os.path.basename(video_path)}', file = result_file)
    # Close fixity_result_file
    result_file.close()
    
    shutil.copy(fixity_result_file, fixity_md5_file)
    logger.debug(f'MD5 checksum written to {fixity_result_file}\n')    
    return md5_checksum


def read_checksum_from_file(file_path):
    # Read the file in binary mode first to handle encoding issues
    try:
        with open(file_path, 'rb') as file:
            content_bytes = file.read()
    except Exception as e:
        logger.critical(f'Error reading file {file_path}: {e}\n')
        return None
    
    # Try to decode with utf-8 first, with error reporting
    try:
        content = content_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        logger.warning(f'UTF-8 decoding error in {file_path}: {e}')
        # Try with latin-1 as a fallback, which can handle any byte
        try:
            content = content_bytes.decode('latin-1')
            logger.warning(f'Used latin-1 encoding as fallback for {file_path}\n')
        except Exception as e2:
            logger.error(f'Failed to decode {file_path} with fallback encoding: {e2}\n')
            return None

    # Try to find the MD5 checksum in the content
    checksum_parts = content.split()
    for part in checksum_parts:
        if len(part) == 32 and all(c in '0123456789abcdefABCDEF' for c in part):
            logger.info(f'MD5 checksum found in {os.path.basename(file_path)}: {part}\n')
            return part

    logger.critical(f'md5 checksum not found in {file_path}\n')
    return None


def hashlib_md5(filename, check_cancelled=None, signals=None):
    '''
    Create an md5 checksum.
    '''
    if check_cancelled():
        return None
    
    read_size = 0
    last_percent_done = 0
    md5_object = hashlib.md5()
    total_size = os.path.getsize(filename)
    logger.debug(f'Generating md5 checksum for {os.path.basename(filename)} via {os.path.basename(__file__)}:')
    
    with open(str(filename), 'rb') as file_object:
        while True:
            if check_cancelled():
                logger.warning("Checksum calculation cancelled.")
                return None
            buf = file_object.read(2**20)
            if not buf:
                break
            read_size += len(buf)
            md5_object.update(buf)
            
            # Calculate percentage (0-100)
            percent_done = int((read_size * 100) / total_size)
            
            # Ensure percent_done is capped at 100
            percent_done = min(100, percent_done)
            
            if percent_done > last_percent_done:
                # Add debug printing for both cases
                
                if signals:
                    # Make doubly sure we're emitting an integer percentage in range 0-100
                    safe_percent = min(100, max(0, int(percent_done)))
                    signals.md5_progress.emit(safe_percent)
                else:
                    sys.stdout.write('[%d%%]\r' % percent_done)
                    sys.stdout.flush()
                    
                last_percent_done = percent_done
                
    md5_output = md5_object.hexdigest()
    logger.info(f'Calculated md5 checksum is {md5_output}\n')
    return md5_output

## The function above, hashlib_md5 is a slightly modified version of the function from the open-source project IFIscripts
## More here: https://github.com/Irish-Film-Institute/IFIscripts/blob/master/scripts/copyit.py
## IFIscripts license information below:
# The MIT License (MIT)
# Copyright (c) 2015-2018 Kieran O'Leary for the Irish Film Institute.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    file_path = sys.argv[1]
    if not os.path.isdir(file_path):
        print(f"Error: {file_path} is not a directory.")
        sys.exit(1)
    check_fixity(file_path)