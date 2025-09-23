#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["NUMEXPR_MAX_THREADS"] = "11" # troubleshooting goofy numbpy related error "Note: NumExpr detected 11 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8. # NumExpr defaulting to 8 threads."

import csv
from base64 import b64encode
from AV_Spex.utils.config_setup import ChecksConfig
from AV_Spex.utils.config_manager import ConfigManager
from AV_Spex.utils.log_setup import logger

config_mgr = ConfigManager()

# Read CSV files and convert them to HTML tables
def csv_to_html_table(csv_file, style_mismatched=False, mismatch_color="#ff9999", match_color="#d2ffed", check_fail=False):
    try:
        # Try UTF-8 first
        try:
            with open(csv_file, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
        except UnicodeDecodeError:
            # If UTF-8 fails, try latin-1
            with open(csv_file, newline='', encoding='latin-1') as f:
                logger.warning(f"Used latin-1 encoding as fallback for CSV file {csv_file}")
                reader = csv.reader(f)
                rows = list(reader)

        # Rest of the function remains the same
        table_html = '<table>\n'
        header = rows[0]
        table_html += '  <tr>\n'
        for cell in header:
            table_html += f'    <th>{cell}</th>\n'
        table_html += '  </tr>\n'

        for row in rows[1:]:
            table_html += '  <tr>\n'
            for i, cell in enumerate(row):
                if check_fail and cell.lower() == "fail":
                    table_html += f'    <td style="background-color: {mismatch_color};">{cell}</td>\n'
                elif check_fail and cell.lower() == "pass":
                    table_html += f'    <td style="background-color: {match_color};">{cell}</td>\n'
                elif style_mismatched and i == 2 and row[2] != '' and row[1] != row[2]:
                    table_html += f'    <td style="background-color: {match_color};">{cell}</td>\n'
                elif style_mismatched and i == 3 and row[2] != '' and row[1] != row[2]:
                    table_html += f'    <td style="background-color: {mismatch_color};">{cell}</td>\n'
                else:
                    table_html += f'    <td>{cell}</td>\n'
            table_html += '  </tr>\n'
        table_html += '</table>\n'
        return table_html
    except Exception as e:
        logger.error(f"Error processing CSV file {csv_file}: {e}")
        return f"<p>Error processing CSV file: {e}</p>"


def read_text_file(text_file_path):
    try:
        # First try UTF-8
        with open(text_file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try latin-1 which can handle any byte
            with open(text_file_path, 'r', encoding='latin-1') as file:
                logger.warning(f"Used latin-1 encoding as fallback for {text_file_path}")
                return file.read()
        except Exception as e:
            logger.error(f"Failed to decode {text_file_path} with fallback encoding: {e}")
            return f"[Error reading file: {e}]"


def prepare_file_section(file_path, process_function=None):
    if file_path:
        if process_function:
            file_content = process_function(file_path)
        else:
            file_content = read_text_file(file_path)
        file_name = os.path.basename(file_path)
    else:
        file_content = ''
        file_name = ''
    return file_content, file_name


def parse_timestamp(timestamp_str):
    if not timestamp_str:
        return (9999, 99, 99, 99, 9999)  # Return a placeholder tuple for non-timestamp entries
    
    try:
        # Split timestamp into hours, minutes, and seconds
        parts = timestamp_str.split(':')
        if len(parts) != 3:
            return (9999, 99, 99, 99, 9999)  # Invalid format
        
        hours = int(parts[0])
        minutes = int(parts[1])
        
        # Handle seconds with decimals
        seconds_parts = parts[2].split('.')
        seconds = int(seconds_parts[0])
        
        # Handle milliseconds if present
        if len(seconds_parts) > 1:
            # Pad or truncate to 4 digits for consistency
            milliseconds = int(seconds_parts[1].ljust(4, '0')[:4])
        else:
            milliseconds = 0
        
        return (hours, minutes, seconds, milliseconds)
        
    except (ValueError, IndexError):
        # Return placeholder if parsing fails
        return (9999, 99, 99, 99, 9999)


def parse_profile(profile_name):
    # Define a custom order for the profile names
    profile_order = {
        "color_bars_detection": 0,
        "color_bars_evaluation": 1,
        "threshold_profile": 2, 
        "tag_check": 3
    }

    for key in profile_order:
        if profile_name.startswith(key):
            return profile_order[key]
    return 99  # Default order if profile_name does not match any known profiles


def find_qct_thumbs(report_directory):
    thumbs_dict = {}
    thumb_exports_dir = os.path.join(report_directory, 'ThumbExports')

    if os.path.isdir(thumb_exports_dir):
        for file in os.listdir(thumb_exports_dir):
            file_path = os.path.join(thumb_exports_dir, file)
            if os.path.isfile(file_path) and not file.startswith('.DS_Store'):
                qct_thumb_path = file_path
                filename_segments = file.split('.')
                if len(filename_segments) >= 5:
                    profile_name = filename_segments[1]
                    tag_name = filename_segments[2]
                    tag_value = filename_segments[3]
                    
                    # Find the timestamp pattern in the filename
                    # It should be HH.MM.SS.ssss before the .png extension
                    # We need to reconstruct it as HH:MM:SS.ssss
                    
                    # The timestamp starts after tag_value (index 4) and goes until .png
                    # For a file like: JPC_AV_01663.color_bars_evaluation.YMAX.940.0.00.00.53.7870.png
                    # segments[4:] would be ['0', '00', '00', '53', '7870', 'png']
                    # We want to reconstruct 00:00:53.7870
                    
                    timestamp_parts = filename_segments[4:-1]  # Exclude .png
                    
                    if len(timestamp_parts) >= 4:
                        # Assuming format is always HH.MM.SS.milliseconds
                        hours = timestamp_parts[1].zfill(2)    # Second element (skip the first '0')
                        minutes = timestamp_parts[2].zfill(2)
                        seconds = timestamp_parts[3].zfill(2)
                        milliseconds = timestamp_parts[4] if len(timestamp_parts) > 4 else '0000'
                        timestamp_as_string = f"{hours}:{minutes}:{seconds}.{milliseconds}"
                    else:
                        # Fallback
                        timestamp_as_string = ':'.join(timestamp_parts)
                    
                    if profile_name == 'color_bars_detection':
                        qct_thumb_name = f'First frame of color bars\n\nAt timecode: {timestamp_as_string}'
                    else:
                        qct_thumb_name = f'Failed frame \n\n{tag_name}:{tag_value}\n\n{timestamp_as_string}'
                    thumbs_dict[qct_thumb_name] = (qct_thumb_path, tag_name, timestamp_as_string)
                else:
                    qct_thumb_name = ':'.join(filename_segments)
                    thumbs_dict[qct_thumb_name] = (qct_thumb_path, "", "")

    # Sort thumbs_dict by timestamp if possible
    sorted_thumbs_dict = {}
    for key in sorted(thumbs_dict.keys(), key=lambda x: (parse_profile(thumbs_dict[x][1]), parse_timestamp(thumbs_dict[x][2]))):
        sorted_thumbs_dict[key] = thumbs_dict[key]

    return sorted_thumbs_dict

def find_report_csvs(report_directory):

    qctools_colorbars_duration_output = None
    qctools_bars_eval_check_output = None
    qctools_bars_eval_timestamps = None
    colorbars_values_output = None
    qctools_content_check_outputs = []
    qctools_profile_check_output = None
    qctools_profile_timestamps = None
    profile_fails_csv = None
    tags_check_output = None
    tag_fails_csv = None
    colorbars_eval_fails_csv = None
    difference_csv = None

    if os.path.isdir(report_directory):
        for file in os.listdir(report_directory):
            file_path = os.path.join(report_directory, file)
            if os.path.isfile(file_path) and not file.startswith('.DS_Store'):
                if file.startswith("qct-parse_"):
                    if "qct-parse_colorbars_durations" in file:
                        qctools_colorbars_duration_output = file_path
                    elif "qct-parse_colorbars_eval_summary" in file:
                        qctools_bars_eval_check_output = file_path
                    elif "qct-parse_colorbars_values" in file:
                        colorbars_values_output = file_path
                    elif "qct-parse_contentFilter" in file:
                        qctools_content_check_outputs.append(file_path)
                    elif "qct-parse_profile_summary" in file:
                        qctools_profile_check_output = file_path
                    elif "qct-parse_profile_failures" in file:
                        profile_fails_csv = file_path
                    elif "qct-parse_tags_summary.csv" in file:
                        tags_check_output = file_path
                    elif "qct-parse_tags_failures" in file:
                        tag_fails_csv = file_path
                    elif "qct-parse_colorbars_eval_failures" in file:
                        colorbars_eval_fails_csv = file_path
                elif "metadata_difference" in file:
                    difference_csv = file_path

    return qctools_colorbars_duration_output, qctools_bars_eval_check_output, colorbars_values_output, qctools_content_check_outputs, qctools_profile_check_output, profile_fails_csv, tags_check_output, tag_fails_csv, colorbars_eval_fails_csv, difference_csv


def read_xml_file(xml_file_path):
    """
    Read XML file content with proper encoding handling.
    
    Args:
        xml_file_path (str): Path to the XML file
        
    Returns:
        str: XML content or error message
    """
    try:
        # First try UTF-8
        with open(xml_file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try latin-1 which can handle any byte
            with open(xml_file_path, 'r', encoding='latin-1') as file:
                logger.warning(f"Used latin-1 encoding as fallback for {xml_file_path}")
                return file.read()
        except Exception as e:
            logger.error(f"Failed to decode {xml_file_path} with fallback encoding: {e}")
            return f"[Error reading XML file: {e}]"


def find_qc_metadata(destination_directory):
    """
    Find QC metadata files including MediaConch policy.
    
    Returns:
        tuple: Paths to various metadata files and policy content
    """
    exiftool_output_path = None 
    ffprobe_output_path = None
    mediainfo_output_path = None
    mediaconch_csv = None
    fixity_sidecar = None
    mediaconch_policy_content = None
    mediaconch_policy_name = None

    if os.path.isdir(destination_directory):
        for file in os.listdir(destination_directory):
            file_path = os.path.join(destination_directory, file)
            if os.path.isfile(file_path) and not file.startswith('.DS_Store'):
                if "_exiftool_output" in file:
                    exiftool_output_path = file_path
                if "_ffprobe_output" in file:
                    ffprobe_output_path = file_path
                if "_mediainfo_output" in file:
                    mediainfo_output_path = file_path
                if "_mediaconch_output" in file:
                    mediaconch_csv = file_path

    if os.path.isdir(os.path.dirname(destination_directory)):
        parent_dir = os.path.dirname(destination_directory) 
        for file in os.listdir(parent_dir):
            file_path = os.path.join(parent_dir, file)
            if file.endswith('fixity.txt'):
                fixity_sidecar = file_path

    # Get MediaConch policy file if MediaConch was run
    if mediaconch_csv:
        try:
            # Get policy file path from config manager
            policy_name = config_mgr.get_config('checks', ChecksConfig).tools.mediaconch.mediaconch_policy
            if policy_name:
                policy_path = config_mgr.get_policy_path(policy_name)
                if policy_path and os.path.isfile(policy_path):
                    mediaconch_policy_content = read_xml_file(policy_path)
                    mediaconch_policy_name = policy_name
                else:
                    logger.warning(f"MediaConch policy file not found: {policy_name}")
        except Exception as e:
            logger.error(f"Error retrieving MediaConch policy: {e}")

    return (exiftool_output_path, ffprobe_output_path, mediainfo_output_path, 
            mediaconch_csv, fixity_sidecar, mediaconch_policy_content, mediaconch_policy_name)


def generate_thumbnail_for_failure(video_path, tag, tagValue, timestamp, profile_name, thumbPath):
    """
    Generates a thumbnail for a specific failure based on the summarized failures.
    
    Parameters:
        video_path (str): Path to the video file.
        tag (str): The tag that failed.
        tagValue (float): The value of the tag at failure.
        timestamp (str): Timestamp in HH:MM:SS.ssss format.
        profile_name (str): Name of the profile (e.g., 'tag_check', 'color_bars_evaluation').
        thumbPath (str): Directory to save the thumbnail.
    
    Returns:
        str: Path to the generated thumbnail, or None if generation failed.
    """
    import subprocess
    import re
    
    if not os.path.isfile(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
    
    video_basename = os.path.basename(video_path)
    video_id = os.path.splitext(video_basename)[0]
    
    # Create filename matching existing convention
    outputFramePath = os.path.join(thumbPath, f"{video_id}.{profile_name}.{tag}.{tagValue}.{timestamp}.png")
    ffoutputFramePath = outputFramePath.replace(":", ".")
    
    # Windows drive letter fix
    match = re.search(r"[A-Z]\.\/", ffoutputFramePath)
    if match:
        ffoutputFramePath = ffoutputFramePath.replace(".", ":", 1)
    
    # Generate appropriate ffmpeg command based on tag
    if tag == "TOUT":
        ffmpegString = f'ffmpeg -ss {timestamp} -i "{video_path}" -vf signalstats=out=tout:color=yellow -vframes 1 -s 720x486 -y "{ffoutputFramePath}"'
    elif tag == "VREP":
        ffmpegString = f'ffmpeg -ss {timestamp} -i "{video_path}" -vf signalstats=out=vrep:color=pink -vframes 1 -s 720x486 -y "{ffoutputFramePath}"'
    else:
        ffmpegString = f'ffmpeg -ss {timestamp} -i "{video_path}" -vf signalstats=out=brng:color=cyan -vframes 1 -s 720x486 -y "{ffoutputFramePath}"'
    
    try:
        logger.debug(f"Generating thumbnail for {tag} failure at {timestamp}\n")
        result = subprocess.run(ffmpegString, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        if result.returncode == 0 and os.path.isfile(ffoutputFramePath):
            return ffoutputFramePath
        else:
            logger.error(f"Failed to generate thumbnail: {result.stderr.decode()}")
            return None
    except Exception as e:
        logger.error(f"Error generating thumbnail: {e}")
        return None


def summarize_failures(failure_csv_path):  # Change parameter to accept CSV file path
    """
    Summarizes the failure information from the CSV file, prioritizing tags 
    with the greatest difference between tag value and threshold, ensuring
    selected frames are at least 4 seconds apart.

    Args:
        failure_csv_path (str): The path to the CSV file containing failure details.

    Returns:
        dict: A dictionary with timestamps as keys and failure details as values.
    """
    
    def timestamp_to_seconds(timestamp_str):
        """Convert HH:MM:SS.ssss to total seconds"""
        parts = timestamp_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    def are_timestamps_far_enough(ts1, ts2, min_gap=4.0):
        """Check if two timestamps are at least min_gap seconds apart"""
        seconds1 = timestamp_to_seconds(ts1)
        seconds2 = timestamp_to_seconds(ts2)
        return abs(seconds1 - seconds2) >= min_gap
    
    failureInfo = {}
    # 0. Read the failure information from the CSV
    try:
        # Try UTF-8 first
        try:
            with open(failure_csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    timestamp = row['Timestamp']
                    if timestamp not in failureInfo:
                        failureInfo[timestamp] = []
                    failureInfo[timestamp].append({
                        'tag': row['Tag'],
                        'tagValue': float(row['Tag Value']),  # Convert to float
                        'over': float(row['Threshold'])     # Convert to float
                    })
        except UnicodeDecodeError:
            # If UTF-8 fails, try latin-1
            with open(failure_csv_path, 'r', encoding='latin-1') as csvfile:
                logger.warning(f"Used latin-1 encoding as fallback for CSV file {failure_csv_path}")
                reader = csv.DictReader(csvfile)
                for row in reader:
                    timestamp = row['Timestamp']
                    if timestamp not in failureInfo:
                        failureInfo[timestamp] = []
                    failureInfo[timestamp].append({
                        'tag': row['Tag'],
                        'tagValue': float(row['Tag Value']),  # Convert to float
                        'over': float(row['Threshold'])     # Convert to float
                    })
    except Exception as e:
        logger.error(f"Error reading failure CSV file {failure_csv_path}: {e}")
        return {}  # Return empty dictionary on error

    # 1. Collect all unique tags and count their occurrences
    tag_counts = {}
    for info_list in failureInfo.values():
        for info in info_list:
            tag = info['tag']
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # 2. Determine the maximum number of frames per tag
    num_tags = len(tag_counts)
    max_frames_per_tag = 5 if num_tags <= 2 else 3

    # 3. Flatten the failureInfo dictionary into a list of tuples
    all_failures = []
    for timestamp, info_list in failureInfo.items():
        for info in info_list:
            all_failures.append((timestamp, info))  # Store as (timestamp, info) tuples

    # 4. Sort the flattened list based on tag value difference
    all_failures.sort(key=lambda x: abs(x[1]['tagValue'] - x[1]['over']), reverse=True)

    # 5. Limit the number of frames per tag with 4-second spacing
    limited_failures = []
    selected_timestamps_per_tag = {}  # Track selected timestamps for each tag
    
    for timestamp, info in all_failures:
        tag = info['tag']
        
        # Initialize list for this tag if needed
        if tag not in selected_timestamps_per_tag:
            selected_timestamps_per_tag[tag] = []
        
        # Check if we've already selected enough frames for this tag
        if len(selected_timestamps_per_tag[tag]) >= max_frames_per_tag:
            continue
        
        # Check if this timestamp is far enough from all previously selected timestamps for this tag
        is_far_enough = True
        for selected_ts in selected_timestamps_per_tag[tag]:
            if not are_timestamps_far_enough(timestamp, selected_ts):
                is_far_enough = False
                break
        
        # If far enough from all selected timestamps, add it
        if is_far_enough:
            limited_failures.append((timestamp, info))
            selected_timestamps_per_tag[tag].append(timestamp)

    # 6. Group the limited failures back into a dictionary by timestamp
    summary_dict = {}
    for timestamp, info in limited_failures:
        if timestamp not in summary_dict:
            summary_dict[timestamp] = []
        summary_dict[timestamp].append(info)

    return summary_dict


def make_color_bars_graphs(video_id, qctools_colorbars_duration_output, colorbars_values_output, sorted_thumbs_dict):
    """
    Creates HTML visualizations for color bars analysis, including bar charts comparing 
    SMPTE color bars with the video's color bars values.

    Args:
        video_id (str): The identifier for the video being analyzed.
        qctools_colorbars_duration_output (str): Path to CSV file containing color bars duration data.
        colorbars_values_output (str): Path to CSV file containing color bars values data.
        sorted_thumbs_dict (dict): Dictionary containing thumbnail information with keys as descriptions
                                 and values as tuples of (path, profile_name, timestamp). 
                                 Output find_qct_thumbs().

    Returns:
        str or None: HTML string containing the visualization if successful, None if there are errors.
    """
    try:
        import plotly.graph_objs as go
    except ImportError as e:
        logger.critical(f"Error importing required libraries for graphs: {e}")
        return None
    
    try:
        with open(colorbars_values_output, 'r') as file:
            csv_reader = csv.DictReader(file)
            # Convert CSV data to lists for plotly
            colorbar_csv_data = {
                'QCTools Fields': [],
                'SMPTE Colorbars': [],
                f'{video_id} Colorbars': []
            }
            for row in csv_reader:
                colorbar_csv_data['QCTools Fields'].append(row['QCTools Fields'])
                # Convert string values to float
                colorbar_csv_data['SMPTE Colorbars'].append(float(row['SMPTE Colorbars']))
                colorbar_csv_data[f'{video_id} Colorbars'].append(float(row[f'{video_id} Colorbars']))
    except Exception as e:
        logger.critical(f"Error reading colorbars CSV file: {e}")
        return None

     # Initialize duration_text with default value
    duration_text = "Colorbars duration: Not available"

    if not os.path.isfile(qctools_colorbars_duration_output):
        logger.critical(f"Cannot open color bars csv file: {qctools_colorbars_duration_output}")
        return None  # Return None if duration file doesn't exist

    try:
        with open(qctools_colorbars_duration_output, 'r') as file:
            duration_lines = file.readlines()
            
            if len(duration_lines) <= 1:  # Check if file is empty or doesn't have enough lines
                logger.critical(f"The csv file {qctools_colorbars_duration_output} does not match the expected format")
                return None  # Return None if duration file is empty or malformed
            
            duration_text = duration_lines[1].strip()
            duration_text = duration_text.replace(',', ' - ')
            duration_text = "Colorbars duration: " + duration_text
            
            # Create the bar chart for the colorbars values
            colorbars_fig = go.Figure(data=[
                go.Bar(name='SMPTE Colorbars', 
                    x=colorbar_csv_data['QCTools Fields'], 
                    y=colorbar_csv_data['SMPTE Colorbars'], 
                    marker=dict(color='#378d6a')),
                go.Bar(name=f'{video_id} Colorbars', 
                    x=colorbar_csv_data['QCTools Fields'], 
                    y=colorbar_csv_data[f'{video_id} Colorbars'], 
                    marker=dict(color='#bf971b'))
            ])
            colorbars_fig.update_layout(barmode='group')
            
            # custom config values for png export
            config = {
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{video_id}_colorbars_comparison',
                    'height': 500,
                    'width': 800,
                    'scale': 1
                }
            }
            colorbars_barchart_html = colorbars_fig.to_html(full_html=False, include_plotlyjs='cdn', config=config)
    except Exception as e:
        logger.critical(f"Error processing duration file: {e}")
        return None 

    # Add annotations for the thumbnail
    thumbnail_html = ''
    for thumb_name, (thumb_path, profile_name, timestamp) in sorted_thumbs_dict.items():
        if "bars_found.first_frame" in thumb_path:
            thumb_name_with_breaks = thumb_name.replace("\n", "<br>")
            thumbnail_html = f'''
                <img src="{thumb_path}" alt="{thumb_name}" style="width:200px; height:auto;">
                <p>{thumb_name_with_breaks}</p>
            '''
            break

    # Create the complete HTML with the duration text and the thumbnail/barchart side-by-side
    colorbars_html = f'''
    <div style="display: flex; align-items: center; justify-content: center; background-color: #f5e9e3; padding: 10px;">
        <div>
            {thumbnail_html}
            <p>{duration_text}</p>
        </div>
        <div style="margin-left: 20px;">  
            {colorbars_barchart_html}
        </div>
    </div>
    '''

    return colorbars_html


def make_profile_piecharts(qctools_profile_check_output, sorted_thumbs_dict, failureInfoSummary, video_id, failure_csv_path=None, check_cancelled=None):
    """
    Creates HTML visualizations showing pie charts of profile check results with thumbnails 
    and detailed failure information for each failed profile check.

    Args:
        qctools_profile_check_output (str): Path to CSV file containing profile check results.
        sorted_thumbs_dict (dict): Dictionary containing thumbnail information with keys as descriptions
                                 and values as tuples of (path, profile_name, timestamp). Output of find_qct_thumbs().
        failureInfoSummary (dict): Dictionary containing detailed failure information, with timestamps
                                 as keys and lists of failure details as values. Output of summarize_failures().
        video_id (str): The identifier for the video being analyzed.
        failure_csv_path (str, optional): Path to the full failures CSV file.
        check_cancelled (callable, optional): Function to check if processing should be cancelled.
                                           Defaults to None.

    Returns:
        str or None: HTML string containing the visualizations if successful, None if there are errors.
    """
    try:
        import plotly.graph_objs as go
    except ImportError as e:
        logger.critical(f"Error importing required libraries for graphs: {e}")
        return None
    
    # Read the full failure data if CSV path is provided
    all_failures = {}
    if failure_csv_path and os.path.isfile(failure_csv_path):
        try:
            with open(failure_csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    tag = row['Tag']
                    if tag not in all_failures:
                        all_failures[tag] = []
                    all_failures[tag].append({
                        'timestamp': row['Timestamp'],
                        'tagValue': row['Tag Value'],
                        'threshold': row['Threshold']
                    })
        except Exception as e:
            logger.error(f"Error reading full failures CSV: {e}")
    
    # Read and validate CSV file
    try:
        # Check if file exists
        if not os.path.isfile(qctools_profile_check_output):
            logger.critical(f"Profile check CSV file not found: {qctools_profile_check_output}")
            return None
    
        # First, get the total frames from the third line
        with open(qctools_profile_check_output, 'r') as file:
            lines = file.readlines()
            total_frames_line = lines[2].strip()
            _, total_frames = total_frames_line.split(',')
            total_frames = int(total_frames)
            
            # Read the profile summary data, skipping the first three lines
            profile_data = []
            csv_reader = csv.DictReader(lines[3:])
            for row in csv_reader:
                profile_data.append(row)

    except Exception as e:
        logger.critical(f"Error processing profile check CSV: {e}")
        return None

    # Initialize profile summary html
    profile_summary_html = None
    profile_summary_pie_charts = []

    # Create a reverse lookup dictionary for thumbnails by timestamp and tag
    thumb_lookup = {}
    for thumb_name, (thumb_path, tag_name, timestamp) in sorted_thumbs_dict.items():
        if timestamp and tag_name:  # Only process thumbnails with valid timestamps and tags
            key = (timestamp, tag_name)
            thumb_lookup[key] = (thumb_path, thumb_name)

    # Add JavaScript for toggling tables
    javascript_code = """
    <script>
    function openImage(imgData, caption) {
        var newWindow = window.open('', '_blank');
        newWindow.document.write('<html><head><title>' + caption + '</title></head><body style="margin:0; background:#000; display:flex; align-items:center; justify-content:center; height:100vh;">');
        newWindow.document.write('<img src="' + imgData + '" style="max-width:100%; max-height:100%; object-fit:contain;">');
        newWindow.document.write('</body></html>');
        newWindow.document.close();
    }

    function toggleTable(tagId) {
        var table = document.getElementById('table_' + tagId);
        var link = document.getElementById('link_' + tagId);
        if (table.style.display === 'none') {
            table.style.display = 'block';
            link.textContent = 'Hide all failures ▲';
        } else {
            table.style.display = 'none';
            link.textContent = 'Show all failures ▼';
        }
    }
    </script>
    """

    # Create pie charts for the profile summary
    tag_counter = 0  # Counter to create unique IDs
    for row in profile_data:
        if check_cancelled and check_cancelled():
            logger.warning("HTML report cancelled.")
            return profile_summary_html

        tag = row['Tag']
        failed_frames = int(row['Number of failed frames'])
        percentage = float(row['Percentage of failed frames'])

        if tag != 'Total':
            tag_id = f"tag_{tag_counter}"  # Create unique ID for this tag
            tag_counter += 1
            
            if percentage > 0:
                # Initialize variables for summary data
                failure_entries_html = []

                # Get failure details for this tag
                for timestamp, info_list in failureInfoSummary.items():
                    for info in info_list:
                        if info['tag'] == tag:
                            # Look for thumbnail for this specific timestamp and tag
                            thumb_html = ""
                            lookup_key = (timestamp, tag)
                            
                            if lookup_key in thumb_lookup:
                                thumb_path, thumb_name = thumb_lookup[lookup_key]
                                with open(thumb_path, "rb") as image_file:
                                    encoded_string = b64encode(image_file.read()).decode()
                                # Create caption for the new window
                                caption = f"{tag} at {timestamp} - Value: {info['tagValue']}, Threshold: {info['over']}"
                                # Make thumbnail clickable with JavaScript
                                thumb_html = f'''<img src="data:image/png;base64,{encoded_string}" 
                                                onclick="openImage('data:image/png;base64,{encoded_string}', '{caption}')"
                                                style="width: 100px; height: auto; vertical-align: middle; margin-left: 10px; cursor: pointer; border: 1px solid #ccc;" 
                                                title="Click to enlarge" />'''
                            
                            # Create entry with or without thumbnail
                            entry_html = f'''
                            <div style="margin-bottom: 10px;">
                                <span><b>Timestamp: {timestamp}</b> | <b>Value:</b> {info['tagValue']} | <b>Threshold:</b> {info['over']}</span>
                                {thumb_html}
                            </div>
                            '''
                            failure_entries_html.append(entry_html)

                # Create formatted failure summary with all thumbnails
                formatted_failures = "".join(failure_entries_html)
                
                # Create the full failures table (hidden by default)
                full_table_html = ""
                if tag in all_failures:
                    table_rows = []
                    for failure in all_failures[tag]:
                        table_rows.append(f"""
                        <tr>
                            <td>{failure['timestamp']}</td>
                            <td>{failure['tagValue']}</td>
                            <td>{failure['threshold']}</td>
                        </tr>
                        """)
                    
                    full_table_html = f"""
                    <div id="table_{tag_id}" style="display: none; margin-top: 10px;">
                        <table style="border-collapse: collapse; width: 100%; border: 1px solid #4d2b12;">
                            <tr style="background-color: #fbe4eb;">
                                <th style="border: 1px solid #4d2b12; padding: 8px;">Timestamp</th>
                                <th style="border: 1px solid #4d2b12; padding: 8px;">Value</th>
                                <th style="border: 1px solid #4d2b12; padding: 8px;">Threshold</th>
                            </tr>
                            {''.join(table_rows)}
                        </table>
                    </div>
                    """
                
                summary_html = f"""
                <div style="display: flex; flex-direction: column; align-items: flex-start; background-color: #f5e9e3; padding: 10px; max-height: 400px; overflow-y: auto;">
                    <p><b>Peak Values outside of Threshold for {tag}:</b></p>
                    {formatted_failures}
                    <a id="link_{tag_id}" href="javascript:void(0);" onclick="toggleTable('{tag_id}')" style="color: #378d6a; text-decoration: underline; margin-top: 10px;">Show all failures ▼</a>
                    {full_table_html}
                </div>
                """
            else:
                # For 0% failures, show a simple message without thumbnails
                summary_html = f"""
                <div style="display: flex; flex-direction: column; align-items: flex-start; background-color: #f5e9e3; padding: 10px;">
                    <p><b>All values within specified threshold</b></p>
                </div>
                """

            # Generate Pie chart (same for both cases)
            pie_fig = go.Figure(data=[go.Pie(
                labels=['Failed Frames', 'Other Frames'],
                values=[failed_frames, total_frames - failed_frames],
                hole=.3,
                marker=dict(colors=['#ffbaba', '#d2ffed'])
            )])
            pie_fig.update_layout(
                title=f"{video_id}<br>{tag} - {percentage:.2f}% ({failed_frames} frames)", 
                height=400, 
                width=400,
                paper_bgcolor='#f5e9e3'
            )

            # plotly config values for custo. png file name
            config = {
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'{video_id}_{tag}',
                    'height': 400,
                    'width': 400,
                    'scale': 1
                }
            }

            # Wrap everything in one div
            pie_chart_html = f"""
            <div style="display: flex; flex-direction: column; align-items: start; background-color: #f5e9e3; padding: 10px;"> 
                <div style="width: 400px;">{pie_fig.to_html(full_html=False, include_plotlyjs='cdn', config=config)}</div>
                {summary_html}
            </div>
            """

            profile_summary_pie_charts.append(f"""
            <div style="display:inline-block; margin-right: 10px; padding-bottom: 20px;">  
                {pie_chart_html}
            </div>
            """)

    # Arrange pie charts horizontally with JavaScript
    profile_piecharts_html = ''.join(profile_summary_pie_charts)

    profile_summary_html = f'''
    {javascript_code}
    <div>
        {profile_piecharts_html}
    </div>
    '''

    return profile_summary_html


def make_content_summary_html(qctools_content_check_output, sorted_thumbs_dict, paper_bgcolor='#f5e9e3'):
    with open(qctools_content_check_output, 'r') as file:
        lines = file.readlines()

    # Find the line with content filter results
    content_filter_line_index = None
    for i, line in enumerate(lines):
        if line.startswith("Segments found within thresholds of content filter"):
            content_filter_line_index = i
            break

    if content_filter_line_index is None:
        return "Content filter results not found in CSV."

    content_filter_name = lines[content_filter_line_index].split()[-1].strip(':')
    time_ranges = lines[content_filter_line_index + 1:]

    matching_thumbs = [
        (thumb_name, thumb_path)
        for thumb_name, (thumb_path, profile_name, _) in sorted_thumbs_dict.items()
        if content_filter_name in thumb_path  # Simplified matching
    ]

    # Build HTML table
    table_rows = []
    for i, time_range in enumerate(time_ranges):
        thumbnail_html = ""
        if i < len(matching_thumbs):  
            thumb_name, thumb_path = matching_thumbs[i]
            with open(thumb_path, "rb") as image_file:
                encoded_string = b64encode(image_file.read()).decode()
            thumbnail_html = f"""<img src="data:image/png;base64,{encoded_string}" style="width: 150px; height: auto;" />"""

        table_rows.append(f"""
            <tr>
                <td style="text-align: center; padding: 10px;">{thumbnail_html}</td>
                <td style="padding: 10px; white-space: nowrap;">{time_range}</td>  
            </tr>
        """)

    content_summary_html = f"""
    <table style="background-color: {paper_bgcolor}; margin-top: 20px; border-collapse: collapse;"> 
        <tr>
            <th colspan="2" style="padding: 10px;">Segments found within thresholds of content filter {content_filter_name}:</th>
        </tr>
        {''.join(table_rows)}
    </table>
    """

    return content_summary_html

def generate_final_report(video_id, source_directory, report_directory, destination_directory, video_path=None, check_cancelled=None, signals=None):
    """
    Generate final HTML report if configured.
    
    Args:
        video_id (str): Unique identifier for the video
        source_directory (str): Source directory for the video
        report_directory (str): Directory containing report files
        destination_directory (str): Destination directory for output files
        video_path (str, optional): Path to the video file for thumbnail generation
        
    Returns:
        str or None: Path to the generated HTML report, or None
    """
    
    checks_config = config_mgr.get_config('checks', ChecksConfig)

    if not checks_config.outputs.report:
        return None

    try:
        html_report_path = os.path.join(source_directory, f'{video_id}_avspex_report.html')
        
        # Generate HTML report with video path
        write_html_report(video_id, report_directory, destination_directory, html_report_path, 
                         video_path=video_path, check_cancelled=check_cancelled)
        
        logger.info(f"HTML report generated: {html_report_path}\n")
        if signals:
            signals.step_completed.emit("Generate Report")
        return html_report_path

    except Exception as e:
        logger.critical(f"Error generating HTML report: {e}")
        import traceback
        logger.critical(f"Traceback: {traceback.format_exc()}")
        return None


def write_html_report(video_id, report_directory, destination_directory, html_report_path, video_path=None, check_cancelled=None):
    """
    Modified to include MediaConch policy file in the HTML report.
    """
    qctools_colorbars_duration_output, qctools_bars_eval_check_output, colorbars_values_output, qctools_content_check_outputs, qctools_profile_check_output, profile_fails_csv, tags_check_output, tag_fails_csv, colorbars_eval_fails_csv, difference_csv = find_report_csvs(report_directory)

    if check_cancelled():
        return
    
    # Create thumbPath if it doesn't exist
    thumbPath = os.path.join(report_directory, "ThumbExports")
    if not os.path.exists(thumbPath):
        os.makedirs(thumbPath)
    
    # Generate thumbnails for peak failures (existing code...)
    generated_thumbs = {}
    
    if profile_fails_csv and video_path:
        profile_fails_csv_path = os.path.join(report_directory, profile_fails_csv)
        failureInfoSummary_profile = summarize_failures(profile_fails_csv_path)
        
        # Generate thumbnails for profile failures
        for timestamp, info_list in failureInfoSummary_profile.items():
            for info in info_list:
                thumb_path = generate_thumbnail_for_failure(
                    video_path, 
                    info['tag'], 
                    info['tagValue'], 
                    timestamp, 
                    'threshold_profile',  # or determine from context
                    thumbPath
                )
                if thumb_path:
                    thumb_key = f"Failed frame \n\n{info['tag']}:{info['tagValue']}\n\n{timestamp}"
                    generated_thumbs[thumb_key] = (thumb_path, info['tag'], timestamp)
    
    if tag_fails_csv and video_path:
        tag_fails_csv_path = os.path.join(report_directory, tag_fails_csv)
        failureInfoSummary_tags = summarize_failures(tag_fails_csv_path)
        
        # Generate thumbnails for tag failures
        for timestamp, info_list in failureInfoSummary_tags.items():
            for info in info_list:
                thumb_path = generate_thumbnail_for_failure(
                    video_path, 
                    info['tag'], 
                    info['tagValue'], 
                    timestamp, 
                    'tag_check',
                    thumbPath
                )
                if thumb_path:
                    thumb_key = f"Failed frame \n\n{info['tag']}:{info['tagValue']}\n\n{timestamp}"
                    generated_thumbs[thumb_key] = (thumb_path, info['tag'], timestamp)
    
    if colorbars_eval_fails_csv and video_path:
        colorbars_eval_fails_csv_path = os.path.join(report_directory, colorbars_eval_fails_csv)
        failureInfoSummary_colorbars = summarize_failures(colorbars_eval_fails_csv_path)
        
        # Generate thumbnails for colorbar failures
        for timestamp, info_list in failureInfoSummary_colorbars.items():
            for info in info_list:
                thumb_path = generate_thumbnail_for_failure(
                    video_path, 
                    info['tag'], 
                    info['tagValue'], 
                    timestamp, 
                    'color_bars_evaluation',
                    thumbPath
                )
                if thumb_path:
                    thumb_key = f"Failed frame \n\n{info['tag']}:{info['tagValue']}\n\n{timestamp}"
                    generated_thumbs[thumb_key] = (thumb_path, info['tag'], timestamp)
    
    # Merge with existing thumbs (for things like color bars detection)
    existing_thumbs = find_qct_thumbs(report_directory)
    thumbs_dict = {**existing_thumbs, **generated_thumbs}
    
    # Sort thumbs_dict as before
    sorted_thumbs_dict = {}
    for key in sorted(thumbs_dict.keys(), key=lambda x: (parse_profile(thumbs_dict[x][1]), parse_timestamp(thumbs_dict[x][2]))):
        sorted_thumbs_dict[key] = thumbs_dict[key]

    if check_cancelled():
        return
    
    # Modified to get MediaConch policy content
    (exiftool_output_path, mediainfo_output_path, ffprobe_output_path, 
     mediaconch_csv, fixity_sidecar, mediaconch_policy_content, 
     mediaconch_policy_name) = find_qc_metadata(destination_directory)

    if check_cancelled():
        return

    # Initialize and create html from 
    mc_csv_html, mediaconch_csv_filename = prepare_file_section(mediaconch_csv, lambda path: csv_to_html_table(path, style_mismatched=False, mismatch_color="#ffbaba", match_color="#d2ffed", check_fail=True))
    diff_csv_html, difference_csv_filename = prepare_file_section(difference_csv, lambda path: csv_to_html_table(path, style_mismatched=True, mismatch_color="#ffbaba", match_color="#d2ffed", check_fail=False))
    exif_file_content, exif_file_filename = prepare_file_section(exiftool_output_path)
    mi_file_content, mi_file_filename = prepare_file_section(mediainfo_output_path)
    ffprobe_file_content, ffprobe_file_filename = prepare_file_section(ffprobe_output_path)
    fixity_file_content, fixity_file_filename = prepare_file_section(fixity_sidecar)

    # Get qct-parse thumbs if they exists
    thumbs_dict = find_qct_thumbs(report_directory)

    if profile_fails_csv:
        profile_fails_csv_path = os.path.join(report_directory, profile_fails_csv)
        failureInfoSummary_profile = summarize_failures(profile_fails_csv_path)
    else:
        failureInfoSummary_profile = None

    if tag_fails_csv:
        tag_fails_csv_path = os.path.join(report_directory, tag_fails_csv)
        failureInfoSummary_tags = summarize_failures(tag_fails_csv_path)
    else:
        failureInfoSummary_tags = None

    if colorbars_eval_fails_csv:
        colorbars_eval_fails_csv_path = os.path.join(report_directory, colorbars_eval_fails_csv)
        failureInfoSummary_colorbars = summarize_failures(colorbars_eval_fails_csv_path)
    else:
        failureInfoSummary_colorbars = None

    if check_cancelled():
        return

    # Create graphs for all existing csv files (existing code...)
    if qctools_bars_eval_check_output and failureInfoSummary_colorbars:
        colorbars_eval_html = make_profile_piecharts(qctools_bars_eval_check_output, thumbs_dict, failureInfoSummary_colorbars, video_id, failure_csv_path=colorbars_eval_fails_csv_path, check_cancelled=check_cancelled)
    elif qctools_bars_eval_check_output and failureInfoSummary_colorbars is None:
       color_bars_segment = f"""
        <div style="display: flex; flex-direction: column; align-items: start; background-color: #f5e9e3; padding: 10px;"> 
            <p><b>All QCTools values of the video file are within the peak values of the color bars.</b></p>
        </div>
        """
       colorbars_eval_html = f"""
        <div style="display:inline-block; margin-right: 10px; padding-bottom: 20px;">  
            {color_bars_segment}
        </div>
        """
    else:
        colorbars_eval_html = None

    if colorbars_values_output:
        colorbars_html = make_color_bars_graphs(video_id,qctools_colorbars_duration_output,colorbars_values_output,thumbs_dict)
    else:
         colorbars_html = None

    if qctools_profile_check_output and failureInfoSummary_profile:
        profile_summary_html = make_profile_piecharts(qctools_profile_check_output, thumbs_dict, failureInfoSummary_profile, video_id, failure_csv_path=profile_fails_csv_path, check_cancelled=check_cancelled)
    else:
        profile_summary_html = None

    if qctools_content_check_outputs:
        content_summary_html_list = []
        for output_csv in qctools_content_check_outputs:
            content_summary_html = make_content_summary_html(output_csv, thumbs_dict, paper_bgcolor='#f5e9e3')
            content_summary_html_list.append(content_summary_html)
    else:
        content_summary_html_list = None

    if tags_check_output and failureInfoSummary_tags:
        tags_summary_html = make_profile_piecharts(tags_check_output, thumbs_dict, failureInfoSummary_tags, video_id, failure_csv_path=tag_fails_csv_path, check_cancelled=check_cancelled)
    else:
        tags_summary_html = None

    existing_thumbs = find_qct_thumbs(report_directory)
    no_qct_parse_files = (
        not profile_fails_csv and 
        not tag_fails_csv and 
        not colorbars_eval_fails_csv and 
        not existing_thumbs
    )

    if check_cancelled():
        return

    # Determine the  path to the image file
    logo_image_path = config_mgr.get_logo_path('av_spex_the_logo.png')
    eq_image_path = config_mgr.get_logo_path('germfree_eq.png')

    # HTML template
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AV Spex Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #fcfdff;
                color: #011054;
                margin: 30px;
            }}
            h1 {{
                font-size: 24px;
                text-align: center;
                margin-top: 20px;
                color: #378d6a;
            }}
            h2 {{
                font-size: 20px;
                font-weight: bold;
                margin-top: 30px;
                color: #0a5f1c;
                text-decoration: underline;
            }}
            h3 {{
                font-size: 18px;
                margin-top: 20px;
                color: #bf971b;
            }}
            table {{
                border-collapse: collapse;
                margin-top: 10px;
                margin-bottom: 20px;
                border: 2px solid #4d2b12;
            }}
            th, td {{
                border: 1.5px solid #4d2b12;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #fbe4eb;
                font-weight: bold;
            }}
            pre {{
                background-color: #f5e9e3;
                border: 1px solid #4d2b12;
                padding: 10px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            .xml-content {{
                background-color: #f8f9fa;
                border: 1px solid #6c757d;
                padding: 15px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                max-height: 400px;
                overflow-y: auto;
            }}
        </style>
        <img src="{logo_image_path}" alt="AV Spex Logo" style="display: block; margin-left: auto; margin-right: auto; width: 25%; margin-top: 20px;">
    </head>
    <body>
        <h1>AV Spex Report</h1>
        <h2>{video_id}</h2>
        <img src="{eq_image_path}" alt="AV Spex Graphic EQ Logo" style="width: 10%">
    """

    if check_cancelled():
        return

    if fixity_sidecar:
        html_template += f"""
        <pre>{fixity_file_content}</pre>
        """

    if mediaconch_csv:
        html_template += f"""
        <h3>{mediaconch_csv_filename}</h3>
        {mc_csv_html}
        """

    # Add MediaConch policy section if available
    if mediaconch_policy_content and mediaconch_policy_name:
        html_template += f"""
        <h3>MediaConch Policy File: {mediaconch_policy_name}</h3>
        <div class="xml-content">{mediaconch_policy_content}</div>
        """

    if difference_csv:
        html_template += f"""
        <h3>{difference_csv_filename}</h3>
        {diff_csv_html}
        """

    # Rest of the HTML template remains the same...
    if no_qct_parse_files:
        html_template += """
        <h3>QCT-Parse Analysis</h3>
        <div style="background-color: #fff3cd; padding: 15px; border: 1px solid #856404; margin: 10px 0; border-radius: 5px;">
            <p style="margin: 0; color: #856404;"><strong>Information:</strong> QCT-Parse analysis was not performed for this video. Quality control analysis sections are not available in this report.</p>
        </div>
        """

    if colorbars_html:
        html_template += f"""
        <h3>SMPTE Colorbars vs {video_id} Colorbars</h3>
        {colorbars_html}
        """

    if colorbars_eval_html:
        html_template += f"""
        <h3>Values relative to colorbar's thresholds</h3>
        {colorbars_eval_html}
        """

    if profile_summary_html:
        html_template += f"""
        <h3>qct-parse Profile Summary</h3>
        <div style="white-space: nowrap;">
            {profile_summary_html}
        </div>
        """
    
    if tags_summary_html:
        html_template += f"""
        <h3>qct-parse Tag Check Summary</h3>
        <div style="white-space: nowrap;">
            {tags_summary_html}
        </div>
        """

    if content_summary_html_list:
        for content_summary_html in content_summary_html_list:
            html_template += f"""
            <h3>qct-parse Content Detection</h3>
            <div style="white-space: nowrap;">
                {content_summary_html}
            </div>
            """

    if exiftool_output_path:
        html_template += f"""
        <h3>{exif_file_filename}</h3>
        <pre>{exif_file_content}</pre>
        """

    if mediainfo_output_path:
        html_template += f"""
        <h3>{mi_file_filename}</h3>
        <pre>{mi_file_content}</pre>
        """

    if ffprobe_output_path:
        html_template += f"""
        <h3>{ffprobe_file_filename}</h3>
        <pre>{ffprobe_file_content}</pre>
        """

    if check_cancelled():
        return

    html_template += """
    </body>
    </html>
    """

    # Write the HTML file
    with open(html_report_path, 'w') as f:
        f.write(html_template)

    logger.info("HTML report generated successfully!\n")
