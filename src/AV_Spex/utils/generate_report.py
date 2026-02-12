#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["NUMEXPR_MAX_THREADS"] = "11" # troubleshooting goofy numbpy related error "Note: NumExpr detected 11 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8. # NumExpr defaulting to 8 threads."

import csv
from base64 import b64encode
import json

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

def find_frame_analysis_outputs(source_directory, destination_directory, video_id):
    """
    Find frame analysis output files (border detection, BRNG analysis, signalstats).
    
    Args:
        source_directory (str): Source directory for the video
        destination_directory (str): Destination directory for outputs
        video_id (str): Video identifier
        
    Returns:
        dict: Paths to frame analysis output files
    """
    frame_outputs = {
        'border_visualization': None,
        'border_data': None,
        'brng_analysis': None,
        'brng_thumbnails': [],
        'signalstats_analysis': None,
        'enhanced_frame_analysis': None
    }
    
    # Check for border detection outputs
    border_viz = os.path.join(destination_directory, f"{video_id}_border_detection.jpg")
    if os.path.exists(border_viz):
        frame_outputs['border_visualization'] = border_viz
    
    border_data = os.path.join(destination_directory, f"{video_id}_border_data.json")
    if os.path.exists(border_data):
        frame_outputs['border_data'] = border_data
    
    # Check for BRNG analysis outputs
    brng_analysis = os.path.join(destination_directory, f"{video_id}_active_brng_analysis.json")
    if os.path.exists(brng_analysis):
        frame_outputs['brng_analysis'] = brng_analysis
    
    # Check for BRNG diagnostic thumbnails
    brng_thumbs_dir = os.path.join(destination_directory, "brng_thumbnails")
    if os.path.exists(brng_thumbs_dir):
        for file in os.listdir(brng_thumbs_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                frame_outputs['brng_thumbnails'].append(os.path.join(brng_thumbs_dir, file))
    
    # Check for enhanced frame analysis JSON (contains signalstats data)
    enhanced_analysis = os.path.join(destination_directory, f"{video_id}_enhanced_frame_analysis.json")
    if os.path.exists(enhanced_analysis):
        frame_outputs['enhanced_frame_analysis'] = enhanced_analysis
        # Extract signalstats data from enhanced frame analysis
        try:
            with open(enhanced_analysis, 'r') as f:
                enhanced_data = json.load(f)
            # Check for signalstats in main results or final_signalstats (from refinement)
            if enhanced_data.get('final_signalstats'):
                frame_outputs['signalstats_analysis'] = enhanced_data['final_signalstats']
            elif enhanced_data.get('signalstats'):
                frame_outputs['signalstats_analysis'] = enhanced_data['signalstats']
            
            # Extract BRNG analysis from enhanced JSON if standalone file doesn't exist
            if not frame_outputs['brng_analysis']:
                brng_data = enhanced_data.get('final_brng_analysis') or enhanced_data.get('brng_analysis')
                if brng_data:
                    frame_outputs['brng_analysis'] = brng_data  # Store as dict directly
                    
        except Exception as e:
            logger.warning(f"Could not read signalstats from enhanced frame analysis: {e}")
    
    # Also check for standalone signalstats file (legacy support)
    if not frame_outputs['signalstats_analysis']:
        signalstats = os.path.join(destination_directory, f"{video_id}_signalstats_analysis.json")
        if os.path.exists(signalstats):
            frame_outputs['signalstats_analysis'] = signalstats
    
    return frame_outputs

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
    if not qctools_colorbars_duration_output or not colorbars_values_output:
        logger.warning("Missing required colorbar files - duration or values file is None")
        return None
        
    try:
        if not os.path.isfile(qctools_colorbars_duration_output):
            logger.critical(f"Cannot open color bars duration csv file: {qctools_colorbars_duration_output}")
            return None
    except (TypeError, AttributeError) as e:
        logger.critical(f"Invalid path for color bars duration file: {e}")
        return None
        
    try:
        if not os.path.isfile(colorbars_values_output):
            logger.critical(f"Cannot open color bars values csv file: {colorbars_values_output}")
            return None
    except Exception as e:
        logger.critical(f"Error reading colorbars CSV file: {e}")
        return None
    
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

def _seconds_to_display(seconds):
    """Convert seconds to a human-readable timecode string (HH:MM:SS.s)"""
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes}:{secs:04.1f}"


def generate_frame_analysis_html(frame_outputs, video_id):
    """
    Generate HTML section for frame analysis results.
    
    Args:
        frame_outputs (dict): Dictionary of frame analysis output paths
        video_id (str): Video identifier
        
    Returns:
        str: HTML string for frame analysis section
    """
    if not any(frame_outputs.values()):
        return ""
    
    html = """
    <div class="frame-analysis-section">
        <h2 style="color: #0a5f1c; text-decoration: underline; margin-top: 30px;">Frame Analysis Results</h2>
    """
    
    # Border Detection Section
    if frame_outputs['border_visualization'] or frame_outputs['border_data']:
        html += "<h3 style='color: #bf971b;'>Border Detection</h3>"
        
        if frame_outputs['border_data']:
            try:
                with open(frame_outputs['border_data'], 'r') as f:
                    border_data = json.load(f)
                
                # Display border detection method
                detection_method = border_data.get('detection_method', 'unknown')
                if detection_method == 'simple_fixed':
                    border_size = border_data.get('border_size_used', 25)
                    html += f"<p><strong>Method:</strong> Simple ({border_size}px borders)</p>"
                else:
                    html += f"<p><strong>Method:</strong> Sophisticated (quality-based detection)</p>"
                
                # Display active area
                if border_data.get('active_area'):
                    x, y, w, h = border_data['active_area']
                    video_width = border_data['video_properties']['width']
                    video_height = border_data['video_properties']['height']
                    active_percentage = (w * h) / (video_width * video_height) * 100
                    
                    html += f"""
                    <div style="background-color: #f5e9e3; padding: 10px; margin: 10px 0;">
                        <p><strong>Active Picture Area:</strong> {w}x{h} pixels ({active_percentage:.1f}% of frame)</p>
                        <p><strong>Position:</strong> ({x}, {y})</p>
                        <p><strong>Borders:</strong> Left={x}px, Right={video_width-x-w}px, Top={y}px, Bottom={video_height-y-h}px</p>
                    </div>
                    """
                
                # Display head switching artifacts if detected
                if border_data.get('head_switching_artifacts'):
                    hs_data = border_data['head_switching_artifacts']
                    if hs_data.get('severity') != 'none' and hs_data.get('severity') != 'error':
                        html += f"""
                        <div style="background-color: #ffbaba; padding: 10px; margin: 10px 0;">
                            <p><strong>⚠️ Head Switching Artifacts Detected</strong></p>
                            <p>Severity: {hs_data['severity']}</p>
                            <p>Affected frames: {hs_data['artifact_percentage']:.1f}%</p>
                        </div>
                        """
                        
            except Exception as e:
                logger.error(f"Error reading border data: {e}")
        
        # Display border visualization image
        if frame_outputs['border_visualization']:
            with open(frame_outputs['border_visualization'], "rb") as img_file:
                encoded_img = b64encode(img_file.read()).decode()
            html += f"""
            <div style="margin: 20px 0;">
                <img src="data:image/jpeg;base64,{encoded_img}" 
                     style="max-width: 100%; height: auto; border: 1px solid #4d2b12;"
                     alt="Border detection visualization" />
            </div>
            """
    
    # BRNG Analysis Section
    if frame_outputs['brng_analysis']:
        html += "<h3 style='color: #bf971b;'>BRNG Violation Analysis</h3>"
        
        # Methodology explanation (collapsible)
        html += """
        <a id="link_brng_methodology" href="javascript:void(0);" 
           onclick="toggleContent('brng_methodology', 'What is BRNG analysis? ▼', 'What is BRNG analysis? ▲')" 
           style="color: #378d6a; text-decoration: underline; margin-bottom: 10px; display: block; font-size: 13px;">
           What is BRNG analysis? ▼</a>
        <div id="brng_methodology" style="display: none; background-color: #f8f6f3; padding: 14px 16px; 
             margin: 0 0 16px 0; border: 1px solid #e0d0c0; border-radius: 4px; font-size: 13px; line-height: 1.5;">
            <p style="margin: 0 0 10px 0;">
                <strong>BRNG (Broadcast Range)</strong> measures whether pixel values fall outside the 
                broadcast-legal range (16–235 for luma, 16–240 for chroma in 8-bit video). Pixels outside 
                this range may be clipped during broadcast or indicate issues in the source material.
            </p>
            <p style="margin: 0 0 10px 0; font-weight: bold;">How violations are detected:</p>
            <p style="margin: 0 0 6px 0;">
                AV Spex uses <em>differential detection</em> to identify violations. For each analysis period, 
                two temporary video segments are created from the active picture area:
            </p>
            <ol style="margin: 4px 0 10px 20px; padding: 0;">
                <li style="margin-bottom: 4px;"><strong>Highlighted version</strong> — rendered with FFmpeg's 
                    <code style="background: #eee; padding: 1px 4px; border-radius: 2px;">signalstats=out=brng:color=magenta</code>, 
                    which overlays magenta on out-of-range pixels</li>
                <li style="margin-bottom: 4px;"><strong>Original version</strong> — rendered without the filter 
                    (cropped to active area only)</li>
            </ol>
            <p style="margin: 0 0 6px 0;">
                Frames are then compared pixel-by-pixel using three independent detection methods that vote 
                on whether a pixel is a genuine violation:
            </p>
            <ol style="margin: 4px 0 10px 20px; padding: 0;">
                <li style="margin-bottom: 4px;"><strong>BGR threshold</strong> — checks for magenta color signature 
                    (high red + blue, low green channel differences)</li>
                <li style="margin-bottom: 4px;"><strong>Ratio-based</strong> — verifies that red and blue 
                    channel increases are proportional (characteristic of magenta overlay)</li>
                <li style="margin-bottom: 4px;"><strong>HSV analysis</strong> — confirms magenta hue range with 
                    saturation increase in HSV color space</li>
            </ol>
            <p style="margin: 0 0 10px 0;">
                A pixel is classified as a violation only when <strong>at least 2 of 3 methods agree</strong>. 
                Small isolated pixel clusters (fewer than 10 connected pixels) are filtered out as noise.
            </p>
            <p style="margin: 0 0 6px 0; font-weight: bold;">Violation classification:</p>
            <p style="margin: 0 0 4px 0;">Each frame with detected violations is then classified by spatial pattern:</p>
            <ul style="margin: 4px 0 10px 16px; padding: 0;">
                <li style="margin-bottom: 3px;"><strong>Sub-black</strong> — violations concentrated in low-luma 
                    zones (pixels below broadcast black level)</li>
                <li style="margin-bottom: 3px;"><strong>Highlight clipping</strong> — violations in high-luma 
                    zones (pixels above broadcast white level)</li>
                <li style="margin-bottom: 3px;"><strong>Edge artifacts</strong> — violations concentrated within 
                    15px of frame edges, suggesting border/blanking issues</li>
                <li style="margin-bottom: 3px;"><strong>Linear blanking patterns</strong> — edge violations 
                    forming continuous horizontal or vertical lines</li>
                <li style="margin-bottom: 3px;"><strong>Border adjustment flags</strong> — edge violations severe 
                    enough to suggest the detected active picture area should be expanded</li>
                <li style="margin-bottom: 3px;"><strong>General broadcast range violations</strong> — violations 
                    that don't match a specific spatial pattern</li>
            </ul>
            <p style="margin: 0; color: #777;">
                Frames to analyze are selected by targeting timestamps where QCTools detected BRNG values, 
                supplemented with evenly distributed samples to ensure coverage across each period.
            </p>
        </div>
        """
        
        try:
            # Handle both file path (string) and dict (from enhanced JSON)
            if isinstance(frame_outputs['brng_analysis'], str):
                with open(frame_outputs['brng_analysis'], 'r') as f:
                    brng_data = json.load(f)
            else:
                brng_data = frame_outputs['brng_analysis']
            
            # Get actionable report
            report = brng_data.get('actionable_report', {})
            aggregate = brng_data.get('aggregate_patterns', {})
            violations = brng_data.get('violations', [])
            period_summaries = brng_data.get('period_summaries', [])
            analysis_periods = brng_data.get('analysis_periods', [])
            
            # Overall assessment banner
            assessment = report.get('overall_assessment', 'Analysis complete')
            stats = report.get('summary_statistics', {})
            
            # Determine assessment styling
            edge_pct = aggregate.get('edge_violation_percentage', 0)
            avg_violation = stats.get('average_violation_percentage', 0)
            max_violation = stats.get('max_violation_percentage', 0)
            
            if max_violation > 50 or avg_violation > 10:
                assessment_bg = '#ffbaba'
                assessment_border = '#d32f2f'
                assessment_icon = '⛔'
            elif edge_pct > 50 or avg_violation > 1:
                assessment_bg = '#fff3cd'
                assessment_border = '#bf971b'
                assessment_icon = '⚠️'
            elif len(violations) > 0:
                assessment_bg = '#e8f4fd'
                assessment_border = '#1976d2'
                assessment_icon = 'ℹ️'
            else:
                assessment_bg = '#d2ffed'
                assessment_border = '#2e7d32'
                assessment_icon = '✅'
            
            html += f"""
            <div style="background-color: {assessment_bg}; padding: 12px 16px; margin: 10px 0; 
                        border-left: 4px solid {assessment_border}; border-radius: 0 4px 4px 0;">
                <p style="margin: 0; font-size: 14px;"><strong>{assessment_icon} Assessment:</strong> {assessment}</p>
            </div>
            """
            
            # ── Period-by-Period Analysis ──
            if period_summaries:
                # Calculate overall time range
                all_starts = [p.get('start_time', 0) for p in period_summaries]
                all_ends = [p.get('end_time', 0) for p in period_summaries]
                time_range_start = min(all_starts) if all_starts else 0
                time_range_end = max(all_ends) if all_ends else 0
                total_violations_across = sum(p.get('violations_found', 0) for p in period_summaries)
                
                html += f"""
                <div style="margin: 16px 0;">
                    <p style="font-weight: bold; margin-bottom: 8px; color: #4d2b12;">
                        Analysis Coverage: {len(period_summaries)} periods 
                        ({_seconds_to_display(time_range_start)} – {_seconds_to_display(time_range_end)})
                    </p>
                """
                
                for ps in period_summaries:
                    p_num = ps.get('period_num', '?')
                    p_start = ps.get('start_time', 0)
                    p_end = ps.get('end_time', 0)
                    qct_targeted = ps.get('qctools_frames_targeted', 0)
                    frames_mapped = ps.get('frames_mapped', 0)
                    total_samples = ps.get('total_samples', 0)
                    checked = ps.get('frames_checked', 0)
                    found = ps.get('violations_found', 0)
                    
                    # Bar width as proportion of violations found vs frames checked
                    bar_pct = (found / checked * 100) if checked > 0 else 0
                    bar_color = '#d32f2f' if bar_pct > 50 else '#bf971b' if bar_pct > 20 else '#1976d2'
                    
                    html += f"""
                    <div style="background-color: #f5e9e3; padding: 10px 14px; margin: 6px 0; 
                                border-radius: 4px; border: 1px solid #e0d0c0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: bold; color: #4d2b12;">
                                Period {p_num}: {_seconds_to_display(p_start)} – {_seconds_to_display(p_end)}
                            </span>
                            <span style="font-size: 13px; color: #666;">
                                {found} violation{'s' if found != 1 else ''} / {checked} frames checked
                            </span>
                        </div>
                        <div style="background-color: #e8ddd5; border-radius: 3px; height: 14px; overflow: hidden; margin-bottom: 6px;">
                            <div style="background-color: {bar_color}; height: 100%; width: {bar_pct:.1f}%; 
                                        min-width: {('2px' if found > 0 else '0')}; border-radius: 3px; 
                                        transition: width 0.3s;"></div>
                        </div>
                        <div style="font-size: 12px; color: #777;">
                            QCTools targeted {qct_targeted} frames → {frames_mapped} mapped to period → {total_samples} total samples analyzed
                        </div>
                    </div>
                    """
                
                html += "</div>"
            
            elif analysis_periods:
                # Fall back to showing period ranges without per-period stats
                time_range_start = min(p[0] for p in analysis_periods)
                time_range_end = max(p[0] + p[1] for p in analysis_periods)
                html += f"""
                <p style="color: #666; font-size: 13px;">
                    Analyzed {len(analysis_periods)} periods spanning 
                    {_seconds_to_display(time_range_start)} – {_seconds_to_display(time_range_end)}
                </p>
                """
            
            # ── Violation Types Breakdown ──
            if violations:
                diagnostic_counts = {}
                edge_artifact_edges = set()
                
                for v in violations:
                    diags = v.get('diagnostics', [])
                    if diags:
                        for diag in diags:
                            if diag.startswith("Edge artifacts"):
                                diagnostic_counts["Edge artifacts"] = diagnostic_counts.get("Edge artifacts", 0) + 1
                                if "(" in diag and ")" in diag:
                                    edges_str = diag[diag.find("(")+1:diag.find(")")]
                                    for edge in edges_str.split(", "):
                                        edge_artifact_edges.add(edge.strip())
                            elif diag == "Border adjustment recommended":
                                diagnostic_counts["Border adjustment flags"] = diagnostic_counts.get("Border adjustment flags", 0) + 1
                            else:
                                diagnostic_counts[diag] = diagnostic_counts.get(diag, 0) + 1
                
                if diagnostic_counts:
                    total_v = len(violations)
                    html += """
                    <div style="margin: 16px 0;">
                        <p style="font-weight: bold; margin-bottom: 8px; color: #4d2b12;">Violation Types Detected</p>
                    """
                    
                    # Sort by count (descending)
                    priority_order = ["Sub-black detected", "Highlight clipping", "Edge artifacts", 
                                     "Linear blanking patterns", "Border adjustment flags",
                                     "General broadcast range violations"]
                    
                    sorted_diags = []
                    for diag_type in priority_order:
                        if diag_type in diagnostic_counts:
                            sorted_diags.append((diag_type, diagnostic_counts[diag_type]))
                    for diag_type, count in diagnostic_counts.items():
                        if diag_type not in priority_order:
                            sorted_diags.append((diag_type, count))
                    
                    # Type-specific colors
                    type_colors = {
                        "Sub-black detected": "#5c6bc0",
                        "Highlight clipping": "#ef6c00",
                        "Edge artifacts": "#bf971b",
                        "Linear blanking patterns": "#7b1fa2",
                        "Border adjustment flags": "#795548",
                        "Continuous edge artifacts": "#bf971b",
                        "General broadcast range violations": "#607d8b",
                        "Border detection likely missed blanking": "#d32f2f",
                        "Moderate blanking detected": "#f57c00"
                    }
                    
                    for diag_type, count in sorted_diags:
                        pct = (count / total_v) * 100
                        bar_color = type_colors.get(diag_type, '#90a4ae')
                        
                        # Build label
                        label = diag_type
                        if diag_type == "Edge artifacts" and edge_artifact_edges:
                            label = f"Edge artifacts ({', '.join(sorted(edge_artifact_edges))})"
                        
                        html += f"""
                        <div style="margin: 4px 0;">
                            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 2px;">
                                <span style="color: #333;">{label}</span>
                                <span style="color: #666;">{count} frames ({pct:.1f}%)</span>
                            </div>
                            <div style="background-color: #e8ddd5; border-radius: 3px; height: 10px; overflow: hidden;">
                                <div style="background-color: {bar_color}; height: 100%; width: {pct:.1f}%; 
                                            min-width: 2px; border-radius: 3px;"></div>
                            </div>
                        </div>
                        """
                    
                    # High edge percentage warning
                    if edge_pct > 50:
                        html += f"""
                        <div style="background-color: #fff3cd; padding: 8px 12px; margin-top: 8px; 
                                    border-left: 3px solid #bf971b; border-radius: 0 4px 4px 0; font-size: 13px;">
                            ⚠ High edge percentage ({edge_pct:.1f}%) suggests border detection may need adjustment
                        </div>
                        """
                    
                    html += "</div>"
            
            # ── Violation Statistics ──
            if stats or aggregate:
                continuous_pct = aggregate.get('continuous_edge_percentage', 0)
                linear_pct = aggregate.get('linear_pattern_percentage', 0)
                
                html += """
                <div style="margin: 16px 0;">
                    <p style="font-weight: bold; margin-bottom: 8px; color: #4d2b12;">Violation Statistics</p>
                    <table style="border-collapse: collapse; width: auto; margin: 0;">
                """
                
                stat_rows = [
                    ("Frames with violations", f"{stats.get('total_violations', 0)}"),
                    ("Average BRNG", f"{avg_violation:.2f}%"),
                    ("Maximum BRNG", f"{max_violation:.2f}%"),
                    ("Edge violations (any)", f"{edge_pct:.1f}% of analyzed frames"),
                    ("Edge violations (solid line)", f"{continuous_pct:.1f}% of analyzed frames"),
                ]
                
                if linear_pct > 0:
                    stat_rows.append(("Linear patterns", f"{linear_pct:.1f}% of analyzed frames"))
                
                # Add note about scattered vs solid edges
                if continuous_pct == 0 and edge_pct > 95:
                    stat_rows.append(("", "→ Violations are scattered rather than forming a solid line"))
                
                for label, value in stat_rows:
                    if label:
                        html += f"""
                        <tr>
                            <td style="padding: 4px 12px 4px 0; color: #555; font-size: 13px; border: none; white-space: nowrap;">{label}</td>
                            <td style="padding: 4px 0; font-weight: bold; font-size: 13px; border: none;">{value}</td>
                        </tr>
                        """
                    else:
                        # Note row (spans both columns)
                        html += f"""
                        <tr>
                            <td colspan="2" style="padding: 4px 0; font-size: 12px; color: #888; border: none; font-style: italic;">{value}</td>
                        </tr>
                        """
                
                html += """
                    </table>
                </div>
                """
            
            # ── Content Start Detection ──
            skip_info = brng_data.get('skip_info', {})
            if skip_info and skip_info.get('total_skipped_seconds', 0) > 0:
                html += f"""
                <div style="background-color: #f5e9e3; padding: 10px; margin: 10px 0; border-radius: 4px;">
                    <p style="margin: 0; font-size: 13px;"><strong>Content Start Detection:</strong> 
                    Skipped first {skip_info['total_skipped_seconds']:.1f} seconds (test patterns/color bars)</p>
                </div>
                """
            
            # ── Recommendations ──
            if report.get('recommendations'):
                html += "<div style='margin: 16px 0;'><p style='font-weight: bold; margin-bottom: 8px; color: #4d2b12;'>Recommendations</p>"
                for rec in report['recommendations']:
                    severity = rec.get('severity', 'low')
                    if severity == 'high':
                        rec_bg = '#ffbaba'
                        rec_border = '#d32f2f'
                    elif severity == 'medium':
                        rec_bg = '#fff3cd'
                        rec_border = '#bf971b'
                    else:
                        rec_bg = '#e8f4fd'
                        rec_border = '#1976d2'
                    
                    html += f"""
                    <div style="background-color: {rec_bg}; padding: 8px 12px; margin: 4px 0; 
                                border-left: 3px solid {rec_border}; border-radius: 0 4px 4px 0; font-size: 13px;">
                        <strong>{rec.get('issue', 'Unknown issue')}</strong>
                    """
                    if rec.get('description'):
                        html += f"<br><span style='color: #555;'>{rec['description']}</span>"
                    html += "</div>"
                html += "</div>"
                
        except Exception as e:
            logger.error(f"Error reading BRNG analysis: {e}")
    
    # BRNG Diagnostic Thumbnails
    if frame_outputs['brng_thumbnails']:
        # Try to get violations data for thumbnail metadata
        thumb_violations_data = []
        try:
            if isinstance(frame_outputs['brng_analysis'], str):
                with open(frame_outputs['brng_analysis'], 'r') as f:
                    _brng_data = json.load(f)
            elif isinstance(frame_outputs['brng_analysis'], dict):
                _brng_data = frame_outputs['brng_analysis']
            else:
                _brng_data = {}
            thumb_violations_data = _brng_data.get('violations', [])
        except Exception:
            pass
        
        num_thumbs = len(frame_outputs['brng_thumbnails'])
        
        html += f"""
        <h3 style='color: #bf971b;'>BRNG Diagnostic Thumbnails</h3>
        <a id="link_brng_thumb_info" href="javascript:void(0);" 
           onclick="toggleContent('brng_thumb_info', 'How are thumbnails selected? ▼', 'How are thumbnails selected? ▲')" 
           style="color: #378d6a; text-decoration: underline; margin-bottom: 10px; display: block; font-size: 13px;">
           How are thumbnails selected? ▼</a>
        <div id="brng_thumb_info" style="display: none; background-color: #f8f6f3; padding: 12px 14px; margin: 0 0 12px 0; 
                    border: 1px solid #e0d0c0; border-radius: 4px; font-size: 13px; line-height: 1.5;">
            <p style="margin: 0 0 6px 0;">
                <strong>Thumbnail selection:</strong> Up to 5 diagnostic thumbnails are chosen from the 
                frames with detected violations. The highest-scoring violation frame is always included. 
                Additional frames are selected in descending order of violation score, subject to a 
                <strong>minimum 10-second temporal separation</strong> from all previously selected frames. 
                This ensures thumbnails represent different moments in the video rather than clustering 
                around a single event. If fewer than 5 frames meet the separation requirement, remaining 
                slots are filled by the next highest-scoring frames regardless of spacing.
            </p>
            <p style="margin: 0; color: #777;">
                Each thumbnail is a 4-quadrant diagnostic image: <strong>Original</strong> (top-left), 
                <strong>BRNG Highlighted</strong> (top-right, magenta = out-of-range pixels), 
                <strong>Violations Only</strong> (bottom-left, yellow on black), and 
                <strong>Analysis Data</strong> (bottom-right, frame number, timestamp, BRNG %, pixel count, 
                and diagnostic classification).
            </p>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0;">
        """
        
        # Sort thumbnails by filename (which includes timecode)
        sorted_thumbs = sorted(frame_outputs['brng_thumbnails'])
        
        for thumb_idx, thumb_path in enumerate(sorted_thumbs[:6]):  # Limit to 6 thumbnails in report
            # Extract timecode from filename if possible
            filename = os.path.basename(thumb_path)
            parts = filename.split('_')
            timecode = "Unknown"
            timestamp_seconds = None
            if len(parts) >= 3:
                timecode = parts[-1].replace('.jpg', '').replace('.png', '').replace('-', ':')
                # Try to parse as seconds for matching
                try:
                    timestamp_seconds = float(timecode.replace('s', ''))
                except (ValueError, AttributeError):
                    pass
            
            # Try to find matching violation data for this thumbnail
            thumb_score = None
            thumb_diags = None
            thumb_brng_pct = None
            if timestamp_seconds is not None and thumb_violations_data:
                for v in thumb_violations_data:
                    v_ts = v.get('timestamp', 0)
                    if abs(v_ts - timestamp_seconds) < 0.2:  # Match within 0.2s
                        thumb_score = v.get('violation_score')
                        thumb_brng_pct = v.get('violation_percentage')
                        thumb_diags = v.get('diagnostics', [])
                        break
            
            with open(thumb_path, "rb") as img_file:
                encoded_thumb = b64encode(img_file.read()).decode()
            
            # Build caption
            caption_parts = [f"Frame at {timecode}"]
            if thumb_score is not None:
                caption_parts.append(f"score: {thumb_score:.4f}")
            if thumb_brng_pct is not None:
                caption_parts.append(f"BRNG: {thumb_brng_pct:.2f}%")
            
            caption_line1 = caption_parts[0]
            caption_line2 = ", ".join(caption_parts[1:]) if len(caption_parts) > 1 else ""
            
            # Diagnostic badge
            diag_html = ""
            if thumb_diags:
                diag_labels = [d for d in thumb_diags if d != "Border adjustment recommended"][:2]
                if diag_labels:
                    badges = ""
                    for dl in diag_labels:
                        badges += f'<span style="display: inline-block; background: #e8ddd5; padding: 1px 6px; border-radius: 3px; font-size: 11px; margin: 1px 2px;">{dl}</span>'
                    diag_html = f'<div style="margin-top: 3px;">{badges}</div>'
            
            html += f"""
            <div style="text-align: center; margin: 5px;">
                <img src="data:image/jpeg;base64,{encoded_thumb}" 
                     style="width: 300px; height: auto; border: 1px solid #4d2b12; cursor: pointer;"
                     onclick="openImage('data:image/jpeg;base64,{encoded_thumb}', 'BRNG Diagnostic - {caption_line1}')"
                     title="Click to enlarge" />
                <p style="font-size: 12px; margin: 4px 0 0 0;">{caption_line1}</p>
            """
            if caption_line2:
                html += f'<p style="font-size: 11px; margin: 1px 0 0 0; color: #777;">{caption_line2}</p>'
            html += diag_html
            html += "</div>"
        
        html += "</div>"
        
        if num_thumbs > 6:
            html += f"<p style='font-style: italic; font-size: 13px;'>Showing 6 of {num_thumbs} diagnostic thumbnails</p>"
    
    # Signalstats Analysis Section
    if frame_outputs['signalstats_analysis']:
        html += "<h3 style='color: #bf971b;'>Signalstats Analysis</h3>"
        
        try:
            # Handle both dict (from enhanced_frame_analysis.json) and file path (legacy)
            if isinstance(frame_outputs['signalstats_analysis'], dict):
                signalstats_data = frame_outputs['signalstats_analysis']
            else:
                with open(frame_outputs['signalstats_analysis'], 'r') as f:
                    signalstats_data = json.load(f)
            
            diagnosis = signalstats_data.get('diagnosis', 'Analysis complete')
            
            # Determine background color based on diagnosis
            if 'broadcast-compliant' in diagnosis.lower() or 'acceptable' in diagnosis.lower():
                bg_color = '#d2ffed'
            elif 'significant' in diagnosis.lower() or 'severe' in diagnosis.lower():
                bg_color = '#ffbaba'
            else:
                bg_color = '#f5e9e3'
            
            html += f"""
            <div style="background-color: {bg_color}; padding: 10px; margin: 10px 0;">
                <p><strong>Diagnosis:</strong> {diagnosis}</p>
            </div>
            """
            
            # Display violation percentage if available
            violation_pct = signalstats_data.get('violation_percentage')
            max_brng = signalstats_data.get('max_brng')
            avg_brng = signalstats_data.get('avg_brng')
            
            if violation_pct is not None or max_brng is not None:
                html += f"""
                <div style="background-color: #f5e9e3; padding: 10px; margin: 10px 0;">
                    <p><strong>Overall Results:</strong></p>
                    <ul>
                """
                if violation_pct is not None:
                    html += f"<li>Frames with violations: {violation_pct:.1f}%</li>"
                if avg_brng is not None:
                    html += f"<li>Average BRNG: {avg_brng:.4f}%</li>"
                if max_brng is not None:
                    html += f"<li>Maximum BRNG: {max_brng:.4f}%</li>"
                html += "</ul></div>"
            
            # Display results for active area (legacy format)
            if signalstats_data.get('results', {}).get('active_area'):
                active_area = signalstats_data['results']['active_area']
                html += f"""
                <div style="background-color: #f5e9e3; padding: 10px; margin: 10px 0;">
                    <p><strong>Active Area Results:</strong></p>
                    <ul>
                        <li>Frames with violations: {active_area['frames_with_violations']}/{active_area['frames_analyzed']} ({active_area['violation_percentage']:.1f}%)</li>
                        <li>Average BRNG: {active_area['avg_brng']:.4f}%</li>
                        <li>Maximum BRNG: {active_area['max_brng']:.4f}%</li>
                    </ul>
                </div>
                """
            
            # Display analysis periods if available
            if signalstats_data.get('analysis_periods'):
                periods = signalstats_data['analysis_periods']
                html += f"""
                <div style="background-color: #f5e9e3; padding: 10px; margin: 10px 0;">
                    <p><strong>Analysis Periods:</strong> {len(periods)} periods analyzed</p>
                    <ul>
                """
                for i, period in enumerate(periods[:5]):  # Show up to 5 periods
                    if isinstance(period, (list, tuple)) and len(period) >= 2:
                        start, duration = period[0], period[1]
                        html += f"<li>Period {i+1}: {start:.1f}s - {start + duration:.1f}s ({duration}s duration)</li>"
                    elif isinstance(period, dict):
                        start = period.get('start_time', period.get('start', 0))
                        duration = period.get('duration', 60)
                        html += f"<li>Period {i+1}: {start:.1f}s - {start + duration:.1f}s ({duration}s duration)</li>"
                html += "</ul></div>"
                
        except Exception as e:
            logger.error(f"Error reading signalstats analysis: {e}")
    
    html += "</div>"
    return html


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

def generate_final_report(video_id, source_directory, report_directory, destination_directory, 
                         video_path=None, check_cancelled=None, signals=None):
    """
    Generate final HTML report if configured.
    
    Args:
        video_id (str): Unique identifier for the video
        source_directory (str): Source directory for the video
        report_directory (str): Directory containing report files
        destination_directory (str): Destination directory for output files
        video_path (str, optional): Path to the video file for thumbnail generation
        check_cancelled (callable, optional): Function to check if cancelled
        signals (object, optional): Signals for GUI updates
        
    Returns:
        str or None: Path to the generated HTML report, or None
    """
    
    checks_config = config_mgr.get_config('checks', ChecksConfig)

    if not checks_config.outputs.report:
        return None

    try:
        html_report_path = os.path.join(source_directory, f'{video_id}_avspex_report.html')
        
        # Generate HTML report with video path (no frame_analysis parameter needed)
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
    
    # Find frame analysis outputs
    frame_outputs = find_frame_analysis_outputs(
        os.path.dirname(html_report_path),  # source_directory
        destination_directory,
        video_id
    )
    
    # Generate frame analysis HTML section
    frame_analysis_html = generate_frame_analysis_html(frame_outputs, video_id) if frame_outputs else ""

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

    # HTML template with JavaScript functions
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
            .metadata-content {{
                background-color: #f5e9e3;
                border: 1px solid #4d2b12;
                padding: 10px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: monospace;
                max-height: 400px;
                overflow-y: auto;
            }}
        </style>
        <script>
        function openImage(imgData, caption) {{
            var newWindow = window.open('', '_blank');
            newWindow.document.write('<html><head><title>' + caption + '</title></head><body style="margin:0; background:#000; display:flex; align-items:center; justify-content:center; height:100vh;">');
            newWindow.document.write('<img src="' + imgData + '" style="max-width:100%; max-height:100%; object-fit:contain;">');
            newWindow.document.write('</body></html>');
            newWindow.document.close();
        }}

        function toggleTable(tagId) {{
            var table = document.getElementById('table_' + tagId);
            var link = document.getElementById('link_' + tagId);
            if (table.style.display === 'none') {{
                table.style.display = 'block';
                link.textContent = 'Hide all failures ▲';
            }} else {{
                table.style.display = 'none';
                link.textContent = 'Show all failures ▼';
            }}
        }}

        function toggleContent(contentId, showText, hideText) {{
            var content = document.getElementById(contentId);
            var link = document.getElementById('link_' + contentId);
            if (content.style.display === 'none') {{
                content.style.display = 'block';
                link.textContent = hideText;
            }} else {{
                content.style.display = 'none';
                link.textContent = showText;
            }}
        }}
        </script>
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

    # Add MediaConch policy section if available - NOW WITH COLLAPSIBLE FUNCTIONALITY
    if mediaconch_policy_content and mediaconch_policy_name:
        html_template += f"""
        <h3>MediaConch Policy File: {mediaconch_policy_name}</h3>
        <a id="link_mediaconch_policy" href="javascript:void(0);" onclick="toggleContent('mediaconch_policy', 'Show policy content ▼', 'Hide policy content ▲')" style="color: #378d6a; text-decoration: underline; margin-bottom: 10px; display: block;">Show policy content ▼</a>
        <div id="mediaconch_policy" class="xml-content" style="display: none;">{mediaconch_policy_content}</div>
        """

    if frame_analysis_html:
        html_template += frame_analysis_html

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

    if difference_csv:
        html_template += f"""
        <h3>{difference_csv_filename}</h3>
        {diff_csv_html}
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

    # Modified sections with collapsible functionality
    if exiftool_output_path:
        html_template += f"""
        <h3>{exif_file_filename}</h3>
        <a id="link_exiftool" href="javascript:void(0);" onclick="toggleContent('exiftool', 'Show content ▼', 'Hide content ▲')" style="color: #378d6a; text-decoration: underline; margin-bottom: 10px; display: block;">Show content ▼</a>
        <div id="exiftool" class="metadata-content" style="display: none;">{exif_file_content}</div>
        """

    if mediainfo_output_path:
        html_template += f"""
        <h3>{mi_file_filename}</h3>
        <a id="link_mediainfo" href="javascript:void(0);" onclick="toggleContent('mediainfo', 'Show content ▼', 'Hide content ▲')" style="color: #378d6a; text-decoration: underline; margin-bottom: 10px; display: block;">Show content ▼</a>
        <div id="mediainfo" class="metadata-content" style="display: none;">{mi_file_content}</div>
        """

    if ffprobe_output_path:
        html_template += f"""
        <h3>{ffprobe_file_filename}</h3>
        <a id="link_ffprobe" href="javascript:void(0);" onclick="toggleContent('ffprobe', 'Show content ▼', 'Hide content ▲')" style="color: #378d6a; text-decoration: underline; margin-bottom: 10px; display: block;">Show content ▼</a>
        <div id="ffprobe" class="metadata-content" style="display: none;">{ffprobe_file_content}</div>
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