#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The majority of this code is derived from the open source project qct-parse
# which is licensed under the GNU Version 3 License. You may obtain a copy of the license at: https://github.com/FutureDays/qct-parse/blob/master/LICENSE
# Original code is here: https://github.com/FutureDays/qct-parse  

# The original code from the qct-parse was written by Brendan Coates and Morgan Morel as part of the 2016 AMIA "Hack Day"
# Summary of that event here: https://wiki.curatecamp.org/index.php/Association_of_Moving_Image_Archivists_%26_Digital_Library_Federation_Hack_Day_2016

import gzip
import math
import os
import subprocess
import shutil
import sys
import re
import operator
import collections      # for circular buffer
import csv
import datetime as dt
import io
from dataclasses import asdict, dataclass, field
from collections import defaultdict

from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ChecksConfig, SpexConfig
from AV_Spex.utils.config_manager import ConfigManager

config_mgr = ConfigManager()

def load_etree():
    """Helper function to load lxml.etree with error handling"""
    try:
        from lxml import etree
        return etree
    except ImportError as e:
        logger.critical(f"Error importing lxml.etree: {e}")
        return None


def safe_gzip_open_with_encoding_fallback(file_path):
    """
    Opens a gzipped file with encoding fallback handling.
    Returns the raw bytes with encoding information for logging.
    
    Parameters:
        file_path (str): Path to the .gz file
        
    Returns:
        tuple: (raw_bytes, encoding_used) or (None, None) if failed
    """
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    try:
        # Read the raw bytes first
        with gzip.open(file_path, 'rb') as gz_file:
            raw_content = gz_file.read()
    except Exception as e:
        logger.error(f"Error reading gzipped file {file_path}: {e}\n")
        return None, None
    
    # Try to decode with different encodings to find the right one
    for encoding in encodings_to_try:
        try:
            # Test if we can decode successfully
            decoded_content = raw_content.decode(encoding)
            # logger.debug(f"Successfully decoded {file_path} using {encoding} encoding\n")
            return raw_content, encoding
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try with error handling
    try:
        decoded_content = raw_content.decode('utf-8', errors='replace')
        logger.warning(f"Used utf-8 with error replacement for {file_path}\n")
        return raw_content, 'utf-8-replace'
    except Exception as e:
        logger.critical(f"Failed to decode {file_path} with any encoding method: {e}\n")
        return None, None


def safe_gzip_iterparse(file_path, etree_module):
    """
    Safely parse gzipped XML with encoding fallback.
    
    Parameters:
        file_path (str): Path to the .gz file
        etree_module: The lxml.etree module
        
    Returns:
        iterator: XML parser iterator or None if failed
    """
    raw_content, encoding_used = safe_gzip_open_with_encoding_fallback(file_path)
    
    if raw_content is None:
        return None
    
    try:
        # Create a BytesIO object from the raw content
        # lxml.etree.iterparse expects a file-like object that returns bytes
        bytes_io = io.BytesIO(raw_content)
        
        # Create the iterparse iterator
        parser_iter = etree_module.iterparse(bytes_io, events=('end',), tag='frame')
        return parser_iter
        
    except Exception as e:
        logger.error(f"Error creating XML parser for {file_path}: {e}")
        return None


# Dictionary to map the string to the corresponding operator function
operator_mapping = {
    'lt': operator.lt,
    'gt': operator.gt,
}

def getFullTagList():
    """Heler function for retrieving tag list"""
    spex_config = config_mgr.get_config('spex', SpexConfig)
    # init variable for config list of QCTools tags
    fullTagList = asdict(spex_config.qct_parse_values.fullTagList)
    return fullTagList

# Creates timestamp for pkt_dts_time
def dts2ts(frame_pkt_dts_time):
    """
    Converts a time in seconds to a formatted time string in HH:MM:SS.ssss format.

    Parameters:
        frame_pkt_dts_time (str): The time in seconds as a string.

    Returns:
        str: The formatted time string.
    """

    seconds = float(frame_pkt_dts_time)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours < 10:
        hours = "0" + str(int(hours))
    else:
        hours = str(int(hours))  
    if minutes < 10:
        minutes = "0" + str(int(minutes))
    else:
        minutes = str(int(minutes))
    secondsStr = str(round(seconds, 4))
    if int(seconds) < 10:
        secondsStr = "0" + secondsStr
    else:
        seconds = str(minutes)
    while len(secondsStr) < 7:
        secondsStr = secondsStr + "0"
    timeStampString = hours + ":" + minutes + ":" + secondsStr
    return timeStampString


# finds stuff over/under threshold
def threshFinder(qct_parse, video_path, inFrame, startObj, pkt, tag, over, thumbPath, thumbDelay, thumbExportDelay, profile_name, failureInfo):
    """
    Compares tagValue in frameDict (from qctools.xml.gz) with threshold from config

    Parameters:
        qct_parse (dict): qct-parse dictionary from command_config.yaml 
        video_path (file): Path to the video file.
        inFrame (dict): The most recent frameDict in framesList
        startObj (qctools.xml.gz): Starting object or reference, used in logging or naming.
        pkt (str): The attribute key used to extract timestamps from <frame> tag in qctools.xml.gz.
        tag (str): Attribute tag from <frame> tag in qctools.xml.gz, is checked against the threshold.
        over (float): The threshold value to compare against, from config
        comp_op (callable): The comparison operator function (e.g., operator.lt, operator.gt).
        thumbPath (str): Path where thumbnails are saved.
        thumbDelay (int): Current delay count between thumbnails.
        thumbExportDelay (int): Required delay count between exporting thumbnails.
        profile_name (str): The name of the profile being checked against, used in naming thumbnail images
        failureInfo (dict): Dictionary that stores tag, tagValue and threshold value (over) for each failed timestamp

    Returns:
        tuple: (bool indicating if threshold was met, updated thumbDelay, updated failureInfo dictionary)
    """

    tagValue = float(inFrame[tag])
    frame_pkt_dts_time = inFrame[pkt]

    if "MIN" in tag or "LOW" in tag:
        comparision = operator.lt
    else:
        comparision = operator.gt
	
    if comparision(float(tagValue), float(over)): # if the attribute is over usr set threshold
        timeStampString = dts2ts(frame_pkt_dts_time)
        # Store failure information in the dictionary (update the existing dictionary, not create a new one)
        if timeStampString not in failureInfo:  # If timestamp not in dict, initialize an empty list
            failureInfo[timeStampString] = []

        failureInfo[timeStampString].append({  # Add failure details to the list
            'tag': tag,
            'tagValue': tagValue,
            'over': over
        })

        # Remove the thumbnail generation - just reset thumbDelay if conditions met
        if qct_parse['thumbExport'] and (thumbDelay > int(thumbExportDelay)):
            # Previously generated thumbnail here, now just reset the delay counter
            thumbDelay = 0
        
        return True, thumbDelay, failureInfo # return true because it was over and thumbDelay
    else:
        return False, thumbDelay, failureInfo # return false because it was NOT over and thumbDelay


#  print thumbnail images of overs/unders        
def printThumb(video_path, tag, profile_name, startObj, thumbPath, tagValue, timeStampString):
    """
    Exports a thumbnail image for a specific frame 

    Parameters:
        video_path (str): Path to the video file.
        tag (str): Attribute tag of the frame, used for naming the thumbnail.
        startObj
    """
    if os.path.isfile(video_path):
        video_basename = os.path.basename(video_path)
        video_id = os.path.splitext(video_basename)[0]
        outputFramePath = os.path.join(thumbPath, video_id + "." + profile_name + "." + tag + "." + str(tagValue) + "." + timeStampString + ".png")
        ffoutputFramePath = outputFramePath.replace(":", ".")
        # for windows we gotta see if that first : for the drive has been replaced by a dot and put it back
        match = ''
        match = re.search(r"[A-Z]\.\/", ffoutputFramePath) # matches pattern R./ which should be R:/ on windows
        if match:
            ffoutputFramePath = ffoutputFramePath.replace(".", ":", 1) # replace first instance of "." in string ffoutputFramePath
        if tag == "TOUT":
            ffmpegString = "ffmpeg -ss " + timeStampString + ' -i "' + video_path +  '" -vf signalstats=out=tout:color=yellow -vframes 1 -s 720x486 -y "' + ffoutputFramePath + '"' # Hardcoded output frame size to 720x486 for now, need to infer from input eventually
        elif tag == "VREP":
            ffmpegString = "ffmpeg -ss " + timeStampString + ' -i "' + video_path +  '" -vf signalstats=out=vrep:color=pink -vframes 1 -s 720x486 -y "' + ffoutputFramePath + '"' # Hardcoded output frame size to 720x486 for now, need to infer from input eventually
        else:
            ffmpegString = "ffmpeg -ss " + timeStampString + ' -i "' + video_path +  '" -vf signalstats=out=brng:color=cyan -vframes 1 -s 720x486 -y "' + ffoutputFramePath + '"' # Hardcoded output frame size to 720x486 for now, need to infer from input eventually
        # Removing logging statement for now - too much clutter in output
        # logger.warning(f"Exporting thumbnail image of {video_id} to {os.path.basename(ffoutputFramePath)}\n")
        output = subprocess.Popen(ffmpegString, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    else:
        logger.critical("Input video file not found when attempting to create thumbnail for report. Ensure video file is in the '_qc_metadata' directory as the QCTools report and report file name contains video file extension.")
        exit()
    return


# detect bars    
def detectBars(startObj, pkt, durationStart, durationEnd, framesList, buffSize, bit_depth_10, signals=None, total_duration=None):
    """
    Detects color bars in a video by analyzing frames within a buffered window and logging the start and end times of the bars.

    This function iterates through the frames in a QCTools report, parses each frame, 
    and analyzes specific tags (YMAX, YMIN, YDIF) to detect the presence of color bars. 
    The detection checks a frame each time the buffer reaches the specified size (`buffSize`) and ends when the frame tags no longer match the expected bar values.

    Args:
    args (argparse.Namespace): Parsed command-line arguments.
    startObj (str): Path to the QCTools report file (.qctools.xml.gz)
    pkt (str): Key used to identify the packet timestamp (pkt_*ts_time) in the XML frames.
    durationStart (str): The timestamp when the bars start, initially an empty string.
    durationEnd (str): The timestamp when the bars end, initially an empty string.
    framesList (list): List of dictionaries storing the parsed frame data.
    buffSize (int): The size of the frame buffer to hold frames for analysis.

    Returns:
    tuple:
    float: The timestamp (`durationStart`) when the bars were first detected.
    float: The timestamp (`durationEnd`) when the bars were last detected.

    Behavior:
    - Parses the input XML file frame by frame.
    - Each frame's timestamp (`pkt_*ts_time`) and key-value pairs are stored in a dictionary (`frameDict`).
    - Once the buffer reaches the specified size (`buffSize`), it checks the middle frame's attributes:
    - Color bars are detected if `YMAX > 210`, `YMIN < 10`, and `YDIF < 3.0`.
    - Logs the start and end times of the bars and stops detection once the bars end.
    - Clears the memory of parsed elements to avoid excessive memory usage during parsing.

    Example log outputs:
    - "Bars start at [timestamp] ([formatted timestamp])"
    - "Bars ended at [timestamp] ([formatted timestamp])"
    """
    etree = load_etree()
    if etree is None:
        return "", "", None, None
    
    if bit_depth_10:
        YMAX_thresh = 800
        YMIN_thresh = 10
        YDIF_thresh = 8.0     # Allow up to 8.0 for 10-bit (your real bars have ~6.5)
        SATMAX_thresh = 250   # Key discriminator - real color bars have high saturation
        URANGE_min = 500      # Real color bars have large chroma ranges
        VRANGE_min = 500
    else:
        YMAX_thresh = 210
        YMIN_thresh = 10
        YDIF_thresh = 3.0
        SATMAX_thresh = 100   # Scaled for 8-bit
        URANGE_min = 150
        VRANGE_min = 150

    barsStartString = None
    barsEndString = None
    consecutive_non_bar_frames = 0
    max_consecutive_non_bar_frames = 10  # Allow up to 10 consecutive bad frames before ending
    
    # Add time limit for searching (5 minutes = 300 seconds)
    search_time_limit = 300.0
    
    # Add tolerance for occasional frames that don't meet criteria
    consecutive_failures = 0
    failure_tolerance = 15  # Allow up to 15 consecutive frames to fail before ending

    # Confirmation window: require consecutive passing frames before confirming bars start
    # This prevents false detection on unstable/skewed bars from analog artifacts
    bars_confirmation_count = 0
    bars_confirmation_threshold = 30  # ~1 second at 30fps must pass before confirming bars
    bars_candidate_start = None       # timestamp of potential bars start (beginning of passing run)

    # Use the safe parser with encoding fallback
    parser_iter = safe_gzip_iterparse(startObj, etree)
    if parser_iter is None:
        logger.error(f"Failed to parse {startObj} for bars detection")
        return "", "", None, None

    try:
        bars_frame_count = 0
        bars_progress_interval = 500  # Emit progress every 500 frames
        bars_last_progress_pct = 5
        for event, elem in parser_iter: #iterparse the xml doc
            if elem.attrib['media_type'] == "video": #get just the video frames
                frame_pkt_dts_time = elem.attrib[pkt] #get the timestamps for the current frame we're looking at
                
                # Emit progress during bars detection (maps into 5→12% range)
                bars_frame_count += 1
                if (signals and hasattr(signals, 'qctparse_progress') and 
                    total_duration and bars_frame_count % bars_progress_interval == 0):
                    pct = 5 + int((float(frame_pkt_dts_time) / total_duration) * 7)
                    pct = min(12, max(5, pct))
                    if pct > bars_last_progress_pct:
                        signals.qctparse_progress.emit(pct)
                        bars_last_progress_pct = pct
                
                # Stop searching after 5 minutes if no bars found yet
                if durationStart == "" and float(frame_pkt_dts_time) > search_time_limit:
                    logger.debug(f"Stopped searching for color bars after {search_time_limit} seconds")
                    break
                
                frameDict = {}  #start an empty dict for the new frame
                frameDict[pkt] = frame_pkt_dts_time  #give the dict the timestamp, which we have now
                for t in list(elem):    #iterating through each attribute for each element
                    keySplit = t.attrib['key'].split(".")   #split the names by dots 
                    keyName = str(keySplit[-1])             #get just the last word for the key name
                    frameDict[keyName] = t.attrib['value']	#add each attribute to the frame dictionary
                framesList.append(frameDict)
                middleFrame = int(round(float(len(framesList))/2))	# get the middle index of the list
                if len(framesList) == buffSize:	# wait till the buffer is full to start detecting bars
                ## This is where the bars detection actually happens
                    # Check conditions - including saturation check
                    if (float(framesList[middleFrame]['YMAX']) > YMAX_thresh and 
                        float(framesList[middleFrame]['YMIN']) < YMIN_thresh and 
                        float(framesList[middleFrame]['YDIF']) < YDIF_thresh and
                        float(framesList[middleFrame].get('SATMAX', 0)) > SATMAX_thresh):  # Add saturation check
                        
                        consecutive_failures = 0  # Reset failure counter when bars are detected
                        
                        if durationStart == "":
                            # Track confirmation window before committing to bars start
                            bars_confirmation_count += 1
                            if bars_candidate_start is None:
                                bars_candidate_start = float(framesList[middleFrame][pkt])
                            
                            if bars_confirmation_count >= bars_confirmation_threshold:
                                durationStart = bars_candidate_start
                                barsStartString = dts2ts(str(bars_candidate_start))
                                logger.debug("Bars start at " + str(bars_candidate_start) + " (" + barsStartString + ")")
                        
                        if durationStart != "":
                            durationEnd = float(framesList[middleFrame][pkt])
                    else:
                        if durationStart == "":
                            # Reset confirmation window - unstable bars interrupted the run
                            bars_confirmation_count = 0
                            bars_candidate_start = None
                        else:
                            # Only count as failure if we've already confirmed bars start
                            consecutive_failures += 1
                            
                            # Only end if we've had enough consecutive failures AND minimum duration
                            if (consecutive_failures >= failure_tolerance and 
                                durationEnd - durationStart > 2):
                                logger.debug("Bars ended at " + str(framesList[middleFrame][pkt]) + " (" + dts2ts(framesList[middleFrame][pkt]) + ")\n")
                                barsEndString = dts2ts(framesList[middleFrame][pkt])
                                break
                elem.clear() # we're done with that element so let's get it outta memory
    except Exception as e:
        logger.error(f"Error during bars detection parsing: {e}")
        
    return durationStart, durationEnd, barsStartString, barsEndString

def validateEntireVideoAsBars(startObj, pkt, durationStart, framesList, buffSize, bit_depth_10):
    """
    Validates if the entire video consists of color bars by checking TOUT, UMAX, UDIF, VMAX, and VDIF values.
    This is used when bars start is detected but no end is found.
    
    Parameters:
        startObj (str): Path to the QCTools report file (.qctools.xml.gz)
        pkt (str): The attribute key used to extract timestamps from <frame> tag in qctools.xml.gz.
        durationStart (float): Initial timestamp marking the start of detected bars.
        framesList (list): List of frameDict dictionaries
        buffSize (int): The size of the frame buffer
        bit_depth_10 (bool): Whether the video is 10-bit depth
    
    Returns:
        tuple: (is_entire_video_bars (bool), video_end_time (float or None))
               Returns (True, end_time) if entire video is bars, (False, None) otherwise
    """
    etree = load_etree()
    if etree is None:
        return False, None
    
    # Threshold for validation
    TOUT_MAX = 0.020  # TOUT should be below 0.020 for color bars
    
    # Thresholds for basic color bar characteristics
    if bit_depth_10:
        YMAX_thresh = 800
        YMIN_thresh = 10
        YDIF_thresh = 25
        SATMAX_thresh = 300
        UMAX_thresh = 1020  
        UDIF_thresh = 20   
        VMAX_thresh = 970  
        VDIF_thresh = 15   
    else:
        YMAX_thresh = 199
        YMIN_thresh = 3
        YDIF_thresh = 2
        SATMAX_thresh = 75
        UMAX_thresh = 255  
        UDIF_thresh = 5
        VMAX_thresh = 243   
        VDIF_thresh = 4
    
    logger.debug(f"No end duration found for color bars - validating if entire video is color bars using TOUT (threshold: {TOUT_MAX}), UMAX, UDIF, VMAX, VDIF")
    
    # Counters for validation
    total_frames_checked = 0
    frames_meeting_criteria = 0
    video_end_time = None
    
    # Use the safe parser with encoding fallback
    parser_iter = safe_gzip_iterparse(startObj, etree)
    if parser_iter is None:
        logger.error(f"Failed to parse {startObj} for entire video validation")
        return False, None
    
    try:
        for event, elem in parser_iter:
            if elem.attrib['media_type'] == "video":
                frame_pkt_dts_time = elem.attrib[pkt]
                video_end_time = float(frame_pkt_dts_time)  # Update with each frame
                
                frameDict = {}
                frameDict[pkt] = frame_pkt_dts_time
                for t in list(elem):
                    keySplit = t.attrib['key'].split(".")
                    keyName = str(keySplit[-1])
                    frameDict[keyName] = t.attrib['value']
                
                framesList.append(frameDict)
                
                if len(framesList) == buffSize:
                    middleFrame = int(round(float(len(framesList))/2))
                    total_frames_checked += 1
                    
                    # Check if frame meets all color bar criteria
                    try:
                        tout_value = float(framesList[middleFrame].get('TOUT', 999))
                        ymax_value = float(framesList[middleFrame].get('YMAX', 0))
                        ymin_value = float(framesList[middleFrame].get('YMIN', 999))
                        ydif_value = float(framesList[middleFrame].get('YDIF', 999))
                        satmax_value = float(framesList[middleFrame].get('SATMAX', 0))
                        # New checks
                        umax_value = float(framesList[middleFrame].get('UMAX', 9999))
                        udif_value = float(framesList[middleFrame].get('UDIF', 9999))
                        vmax_value = float(framesList[middleFrame].get('VMAX', 9999))
                        vdif_value = float(framesList[middleFrame].get('VDIF', 9999))
                        
                        # Check TOUT, basic bar characteristics, and U/V values
                        if (tout_value <= TOUT_MAX and 
                            ymax_value > YMAX_thresh and 
                            ymin_value < YMIN_thresh and 
                            ydif_value < YDIF_thresh and
                            satmax_value > SATMAX_thresh and
                            umax_value < UMAX_thresh and
                            udif_value < UDIF_thresh and
                            vmax_value < VMAX_thresh and
                            vdif_value < VDIF_thresh):
                            frames_meeting_criteria += 1
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Error checking frame values: {e}")
                        pass
                
                elem.clear()
    except Exception as e:
        logger.error(f"Error during entire video validation parsing: {e}")
        return False, None
    
    # Calculate percentage of frames meeting criteria
    if total_frames_checked > 0:
        percentage_valid = (frames_meeting_criteria / total_frames_checked) * 100
        logger.debug(f"Validation results: {frames_meeting_criteria}/{total_frames_checked} frames ({percentage_valid:.1f}%) meet color bar criteria")
        
        # If at least 95% of frames meet the criteria, consider entire video as bars
        if percentage_valid >= 95.0:
            logger.info(f"Confirmed: Entire video consists of color bars (TOUT <= {TOUT_MAX}, UMAX < {UMAX_thresh}, UDIF < {UDIF_thresh}, VMAX < {VMAX_thresh}, VDIF < {VDIF_thresh})")
            return True, video_end_time
        else:
            logger.debug(f"Only {percentage_valid:.1f}% of frames meet criteria - not confirming as entire video bars")
            return False, None
    else:
        logger.debug("No frames were checked during validation")
        return False, None
    

def evalBars(startObj,pkt,durationStart,durationEnd,framesList,buffSize):
    """
    Find maximum or minimum values for specific QCTools keys inside the duration of the color bars. 

    Parameters:
        pkt (str): The attribute key used to extract timestamps from <frame> tag in qctools.xml.gz.
        durationStart (float): Initial timestamp marking the potential start of detected bars.
        durationEnd (float): Timestamp marking the end of detected bars.
        framesList (list): List of frameDict dictionaries

    Returns:
        maxBarsDict (dict): Returns dictionary of max or min value of corresponding QCTools keys
    """
    
    etree = load_etree()
    if etree is None:
        return None
    
    # Define the keys for which you want to calculate the average
    keys_to_check = ['YMAX', 'YMIN', 'UMIN', 'UMAX', 'VMIN', 'VMAX', 'SATMAX', 'SATMIN']
    # Initialize a dictionary to store the highest values for each key
    maxBarsDict = {}
    # adds the list keys_to_check as keys to a dictionary
    for key_being_checked in keys_to_check:
        # assign 'dummy' threshold to be overwritten
        if "MAX" in key_being_checked:
            maxBarsDict[key_being_checked] = 0
        elif "MIN" in key_being_checked:
            maxBarsDict[key_being_checked] = 1023

    # Use the safe parser with encoding fallback
    parser_iter = safe_gzip_iterparse(startObj, etree)
    if parser_iter is None:
        logger.error(f"Failed to parse {startObj} for bars evaluation")
        return None

    try:
        for event, elem in parser_iter: # iterparse the xml doc
            if elem.attrib['media_type'] == "video": # get just the video frames
                frame_pkt_dts_time = elem.attrib[pkt] # get the timestamps for the current frame we're looking at
                if frame_pkt_dts_time >= str(durationStart): 	# only work on frames that are after the start time   # only work on frames that are after the start time
                    if float(frame_pkt_dts_time) > durationEnd:        # only work on frames that are before the end time
                        break
                    frameDict = {}  # start an empty dict for the new frame
                    frameDict[pkt] = frame_pkt_dts_time  # give the dict the timestamp, which we have now
                    for t in list(elem):    # iterating through each attribute for each element
                        keySplit = t.attrib['key'].split(".")   # split the names by dots 
                        keyName = str(keySplit[-1])             # get just the last word for the key name
                        frameDict[keyName] = t.attrib['value']	# add each attribute to the frame dictionary
                    framesList.append(frameDict)
                    if len(framesList) == buffSize:	# wait till the buffer is full to start detecting bars
                        ## This is where the bars detection magic actually happens
                        for colorbar_key in keys_to_check:
                            if colorbar_key in frameDict:
                                if "MAX" in colorbar_key:
                                    # Convert the value to float and compare it with the current highest value
                                    value = float(frameDict[colorbar_key])
                                    if value > maxBarsDict[colorbar_key]:
                                        maxBarsDict[colorbar_key] = value
                                elif "MIN" in colorbar_key:
                                    # Convert the value to float and compare it with the current highest value
                                    value = float(frameDict[colorbar_key])
                                    if value < maxBarsDict[colorbar_key]:
                                        maxBarsDict[colorbar_key] = value
                                # Convert highest values to integer
                                maxBarsDict = {colorbar_key: int(value) for colorbar_key, value in maxBarsDict.items()}
    except Exception as e:
        logger.error(f"Error during bars evaluation parsing: {e}")
							
    return maxBarsDict


def getCompFromConfig(qct_parse, profile, tag):
   """
   Determines the comparison operator based on profile and tag.

    Args:
        qct_parse (dict): qct-parse configuration.
        profile (dict): Profile data.
        tag (str): Tag to check.

    Returns:
        callable: Comparison operator (e.g., operator.lt, operator.gt).
   """
   
   spex_config = config_mgr.get_config('spex', SpexConfig)
   
   smpte_color_bars_keys = asdict(spex_config.qct_parse_values.smpte_color_bars).keys()

   if set(profile) == set(smpte_color_bars_keys):
       return operator.lt if "MIN" in tag else operator.gt

   raise ValueError(f"No matching comparison operator found for profile and tag: {profile}, {tag}")


def analyzeIt(qct_parse, video_path, profile, profile_name, startObj, pkt, durationStart, durationEnd, thumbPath, thumbDelay, thumbExportDelay, framesList, frameCount=0, overallFrameFail=0, adhoc_tag=False, check_cancelled=None, signals=None, total_duration=None):
    """
    Analyzes video frames from the QCTools report to detect threshold exceedances for specified tags or profiles and logs frame failures.

    This function iteratively parses video frames from a QCTools report (`.qctools.xml.gz`) and checks whether the frame attributes exceed user-defined thresholds 
    (either single tags or profiles). Threshold exceedances are logged, and frames can be flagged for further analysis. Optionally, thumbnails of failing frames can be generated.

    Args:
        args (argparse.Namespace): Parsed command-line arguments, including tag thresholds and options for profile, thumbnail export, etc.
        profile (dict): A dictionary of key-value pairs of tag names and their corresponding threshold values.
        startObj (str): Path to the QCTools report file (.qctools.xml.gz)
        pkt (str): Key used to identify the pkt_*ts_time in the XML frames.
        durationStart (float): The starting time for analyzing frames (in seconds).
        durationEnd (float): The ending time for analyzing frames (in seconds). Can be `None` to process until the end.
        thumbPath (str): Path to save the thumbnail images of frames exceeding thresholds.
        thumbDelay (int): Delay counter between consecutive thumbnail generations to prevent spamming.
        framesList (list): A circular buffer to hold dictionaries of parsed frame attributes.
        frameCount (int, optional): The total number of frames analyzed (defaults to 0).
        overallFrameFail (int, optional): A count of how many frames failed threshold checks across all tags (defaults to 0).

    Returns:
        tuple: 
            - kbeyond (dict): A dictionary where each tag is associated with a count of how many times its threshold was exceeded.
            - frameCount (int): The total number of frames analyzed.
            - overallFrameFail (int): The total number of frames that exceeded thresholds across all tags.
            - failureInfo (dict): Dictionary containing failure information.

    Behavior:
        - Iteratively parses the input XML file and analyzes frames after `durationStart` and before `durationEnd`.
        - Frames are stored in a circular buffer (`framesList`), and attributes (tags) are extracted into dictionaries.
        - For each frame, checks whether specified tags exceed user-defined thresholds (from `args.o`, `args.u`, or `profile`).
        - Logs threshold exceedances and updates the count of failed frames.
        - Optionally, generates thumbnails for frames that exceed thresholds, ensuring a delay between consecutive thumbnails.

    Example usage:
        - Analyzing frames using a single tag threshold: `analyzeIt(args, {}, startObj, pkt, durationStart, durationEnd, thumbPath, thumbDelay, framesList)`
        - Analyzing frames using a profile: `analyzeIt(args, profile, startObj, pkt, durationStart, durationEnd, thumbPath, thumbDelay, framesList)`
    """
    etree = load_etree()
    if etree is None:
        return {}, 0, 0, {}
    
    kbeyond = {} # init a dict for each key which we'll use to track how often a given key is over
    fots = "" # init frame over threshold to avoid counting the same frame more than once in the overallFrameFail count
    failureInfo = {}  # Initialize a new dictionary to store failure information
    for k,v in profile.items(): 
        kbeyond[k] = 0

    # Progress emission setup: emit every ~1000 frames to avoid overhead
    # Maps frameCount against total_frames_estimate into the 20→88% range
    # (reserves 88-98% for audio analysis)
    progress_emit_interval = 1000
    last_progress_pct = 0

    # Use the safe parser with encoding fallback
    parser_iter = safe_gzip_iterparse(startObj, etree)
    if parser_iter is None:
        logger.error(f"Failed to parse {startObj} for analysis")
        return {}, 0, 0, {}

    try:
        for event, elem in parser_iter: #iterparse the xml doc
            if elem.attrib['media_type'] == "video": 	#get just the video frames
                frameCount = frameCount + 1
                frame_pkt_dts_time = elem.attrib[pkt] 	#get the timestamps for the current frame we're looking at
                
                # Emit progress periodically
                if (signals and hasattr(signals, 'qctparse_progress') and
                    total_duration and frameCount % progress_emit_interval == 0):
                    pct = 20 + int((float(frame_pkt_dts_time) / total_duration) * 68)
                    pct = min(88, max(20, pct))
                    if pct > last_progress_pct:
                        signals.qctparse_progress.emit(pct)
                        last_progress_pct = pct
                
                if frame_pkt_dts_time >= str(durationStart): 	#only work on frames that are after the start time
                    if check_cancelled():
                        return kbeyond, frameCount, overallFrameFail, failureInfo
                    if durationEnd:
                        if float(frame_pkt_dts_time) > durationEnd:		#only work on frames that are before the end time
                            print("started at " + str(durationStart) + " seconds and stopped at " + str(frame_pkt_dts_time) + " seconds (" + dts2ts(frame_pkt_dts_time) + ") or " + str(frameCount) + " frames!")
                            break
                    frameDict = {}  								#start an empty dict for the new frame
                    frameDict[pkt] = frame_pkt_dts_time  			#make a key for the timestamp, which we have now
                    for t in list(elem):    						#iterating through each attribute for each element
                        keySplit = t.attrib['key'].split(".")   	#split the names by dots 
                        keyName = str(keySplit[-1])             	#get just the last word for the key name
                        if len(keyName) == 1:						#if it's psnr or mse, keyName is gonna be a single char
                            keyName = '.'.join(keySplit[-2:])		#full attribute made by combining last 2 parts of split with a period in btw
                        frameDict[keyName] = t.attrib['value']		#add each attribute to the frame dictionary
                    framesList.append(frameDict)					#add this dict to our circular buffer
                    # Now we can parse the frame data from the buffer!	
                    for k,v in profile.items():
                        tag = k
                        over = float(v)
                        # ACTUALLY DO THE THING ONCE FOR EACH TAG
                        frameOver, thumbDelay, failureInfo = threshFinder(qct_parse, video_path, framesList[-1], startObj, pkt, tag, over, thumbPath, thumbDelay, thumbExportDelay, profile_name, failureInfo)
                        if frameOver is True:
                            kbeyond[k] = kbeyond[k] + 1 # note the over in the key over dict
                            if not frame_pkt_dts_time in fots: # make sure that we only count each over frame once
                                overallFrameFail = overallFrameFail + 1
                                fots = frame_pkt_dts_time # set it again so we don't dupe
                    thumbDelay = thumbDelay + 1				
                elem.clear() #we're done with that element so let's get it outta memory
    except Exception as e:
        logger.error(f"Error during analysis parsing: {e}")

    return kbeyond, frameCount, overallFrameFail, failureInfo


def print_color_bar_values(video_id, smpte_color_bars, maxBarsDict, colorbars_values_output):
    """
    Writes color bar values to a CSV file.

    Compares SMPTE color bar values with those extracted from a video using QCTools.
    The output CSV includes the attribute name, the expected SMPTE value, and the value detected in the video.

    Args:
        video_id (str): Identifier for the video being analyzed.
        smpte_color_bars (dict): Dictionary of expected SMPTE color bar values, from config.yaml
        maxBarsDict (dict): Dictionary of color bar values extracted from the video.
        colorbars_values_output (str): Path to the output CSV file.
    """

    with open(colorbars_values_output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header
        writer.writerow(["QCTools Fields", "SMPTE Colorbars", f"{video_id} Colorbars"])
        
        # Write the data
        for key in smpte_color_bars:
            smpte_value = smpte_color_bars.get(key, "")
            maxbars_value = maxBarsDict.get(key, "")
            writer.writerow([key, smpte_value, maxbars_value])


def printresults(profile, kbeyond, frameCount, overallFrameFail, qctools_check_output):
    """
    Writes the analyzeIt results into a summary file, detailing the count and percentage of frames that exceeded the thresholds.

    Parameters:
        kbeyond (dict): Dictionary mapping tags to the count of frames exceeding the thresholds.
        frameCount (int): Total number of frames analyzed.
        overallFrameFail (int): Total number of frames with at least one threshold exceedance.
        qctools_check_output (str): File path to write the output summary.

    Returns:
        None
    """

    
    spex_config = config_mgr.get_config('spex', SpexConfig)
    fullTagList = getFullTagList()
    
    def format_percentage(value):
        percent = value * 100
        if percent == 100:
            return "100"
        elif percent == 0:
            return "0"
        elif percent < 0.01:
            return "0"
        else:
            return f"{percent:.2f}"

    color_bar_dict = asdict(spex_config.qct_parse_values.smpte_color_bars)
    color_bar_keys = color_bar_dict.keys()

    with open(qctools_check_output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["**************************"])

        if profile == fullTagList:
            writer.writerow(["qct-parse evaluation of user specified tags summary"])
        elif set(profile.keys()) == set(color_bar_keys):
            writer.writerow(["qct-parse color bars evaluation summary"])
        else:
            writer.writerow(["qct-parse profile results summary"])

        if frameCount == 0:
            writer.writerow(["TotalFrames", "0"])
            return

        writer.writerow(["TotalFrames", frameCount])
        writer.writerow(["Tag", "Number of failed frames", "Percentage of failed frames"])

        for tag, count in kbeyond.items():
            percentOverString = format_percentage(count / frameCount)
            writer.writerow([tag, count, percentOverString])

        percentOverallString = format_percentage(overallFrameFail / frameCount)
        writer.writerow(["Total", overallFrameFail, percentOverallString])


def print_color_bar_keys(qctools_colorbars_values_output, profile, color_bar_keys):
    """
    Writes color bar keys and their threshold values to a CSV file.

    If the provided `profile` keys match the expected `color_bar_keys`, 
    the function writes a header indicating the thresholds are based on peak QCTools filter values.
    Then, it writes each key and its corresponding threshold value from the `profile`.

    Args:
    qctools_colorbars_values_output (str): Path to the output CSV file.
    profile (dict): Dictionary containing color bar keys and their threshold values.
    color_bar_keys (list): List of expected color bar keys.
    """

    with open(qctools_colorbars_values_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        if set(profile.keys()) == set(color_bar_keys):
            writer.writerow(["The thresholds defined by the peak values of QCTools filters in the identified color bars are:"])
            for key, value in profile.items():
                writer.writerow([key, value])


def print_timestamps(qctools_timestamp_output, summarized_timestamps, descriptor):
    """
    Writes timestamps of frames with failures to a CSV file.

    If `summarized_timestamps` is not empty, it writes a header indicating the timestamps correspond to frames
    with at least one failure during the qct-parse process, along with the provided `descriptor`.
    Then, for each start and end timestamp pair in `summarized_timestamps`, it writes either a single timestamp 
    (if start and end are the same) or a range of timestamps in the format "HH:MM:SS.mmm, HH:MM:SS.mmm".

    Args:
        qctools_timestamp_output (str): Path to the output CSV file.
        summarized_timestamps (list of tuples): List of (start, end) timestamp pairs.
        descriptor (str): Description of the analysis or filter applied.
    """

    with open(qctools_timestamp_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        if summarized_timestamps:
            writer.writerow([f"Times stamps of frames with at least one fail during qct-parse {descriptor}"])
        for start, end in summarized_timestamps:
            if start == end:
                writer.writerow([start.strftime("%H:%M:%S.%f")[:-3]])
            else:
                writer.writerow([f"{start.strftime('%H:%M:%S.%f')[:-3]}, {end.strftime('%H:%M:%S.%f')[:-3]}"])


def print_bars_durations(qctools_check_output, barsStartString, barsEndString):
    """
    Writes color bar duration information to a CSV file.

    If both `barsStartString` and `barsEndString` are provided, it writes a header indicating color bars were found
    and then writes the start and end timestamps on separate rows.
    If either timestamp is missing, it writes a message indicating no color bars were found.

    Args:
        qctools_check_output (str): Path to the output CSV file.
        barsStartString (str or None): Start timestamp of the color bars.
        barsEndString (str or None): End timestamp of the color bars.
    """
    with open(qctools_check_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        if barsStartString and barsEndString:
            writer.writerow(["qct-parse color bars found:"])
            writer.writerow([barsStartString, barsEndString])
        else:
            writer.writerow(["qct-parse found no color bars"])


# blatant copy paste from https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
def uniquify(path):
    if os.path.isdir(path):
        original_path = path.rstrip(os.sep)  # Remove trailing separator if it exists
        counter = 1
        while os.path.exists(path):
            path = original_path + " (" + str(counter) + ")"
            counter += 1
        return path
    else:
        filename, extension = os.path.splitext(path)
        counter = 1
        while os.path.exists(path):
            path = filename + " (" + str(counter) + ")" + extension
            counter += 1
        return path


def archiveThumbs(thumbPath):
    """
    Archives thumbnail images in a dated subdirectory.

    Checks if the specified `thumbPath` contains any files. If so, it creates a new subdirectory 
    named `archivedThumbs_YYYY_MM_DD` (where YYYY_MM_DD is the creation date of `thumbPath`) 
    and moves all files (except '.DS_Store') from `thumbPath` into this archive directory.
    If a file with the same name already exists in the archive, it's renamed to ensure uniqueness.

    Args:
        thumbPath (str): The path to the directory containing thumbnail images.

    Returns:
        str or None: The path to the newly created archive directory if thumbnails were archived, 
                        otherwise None if `thumbPath` was empty.
    """

    # Check if thumbPath contains any files
    has_files = False
    for entry in os.scandir(thumbPath):
        if entry.is_file():
            has_files = True
            break

    # If thumbPath contains files, create the archive directory
    if has_files:
        # Get the creation time of the thumbPath directory
        creation_time = os.path.getctime(thumbPath)
        creation_date = dt.datetime.fromtimestamp(creation_time)

        # Format the date as YYYY_MM_DD
        date_str = creation_date.strftime('%Y_%m_%d')

        # Create the new directory name
        archive_dir = os.path.join(thumbPath, f'archivedThumbs_{date_str}')

        if os.path.exists(archive_dir):
            archive_dir = archive_dir
        else:
            # Create the archive directory
            os.makedirs(archive_dir)

        # Move all files from thumbPath to archive_dir
        for entry in os.scandir(thumbPath):
            # if an item in the ThumbExports directory is a file, and is no .DS_Store, then:
            if entry.is_file() and entry.name != '.DS_Store':
                # define the new path of the thumbnail, once it has been moved to archive_dir
                entry_archive_path = os.path.join(archive_dir, os.path.basename(entry))
                # But if the new path for that thumbnail is already taken:
                if os.path.exists(entry_archive_path):
                    # Create a unique path for the archived thumb (original name plus sequential number in parentheses (1), (2), etc.)
                    unique_file_path = uniquify(entry_archive_path)
                    # Rename the existing thumb to match the unique path (also moves the file)
                    os.rename(entry, unique_file_path)
                else:
                    shutil.move(entry, archive_dir)

        return archive_dir
    else:
        return None


def save_failures_to_csv(failureInfo, failure_csv_path):
    """Saves the failure information to a CSV file.

    Args:
        failureInfo (dict): A dictionary where keys are timestamps and values are lists of failure details.
        failure_csv_path (str, optional): The path to the CSV file. Defaults to 'failures.csv'.
    """
    with open(failure_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Timestamp', 'Tag', 'Tag Value', 'Threshold']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for timestamp, info_list in failureInfo.items():
            for info in info_list:
                writer.writerow({'Timestamp': timestamp, 'Tag': info['tag'], 'Tag Value': info['tagValue'], 'Threshold': info['over']})

def extract_report_mkv(startObj, qctools_output_path):
    etree = load_etree()
    if etree is None:
        return None
    
    report_file_output = qctools_output_path.replace(".qctools.mkv", ".qctools.xml.gz")

    # If previous report exists, remove it
    if os.path.isfile(report_file_output):
        os.remove(report_file_output)

    # Run ffmpeg command to extract xml.gz report
    full_command = [
        'ffmpeg', 
        '-hide_banner', 
        '-loglevel', 'panic', 
        '-dump_attachment:t:0', report_file_output, 
        '-i', qctools_output_path
    ]
    logger.info(f'Extracting qctools.xml.gz report from {os.path.basename(qctools_output_path)}\n')
    logger.debug(f'Running command: {" ".join(full_command)}\n')
    subprocess.run(full_command)

    if os.path.isfile(report_file_output):
        startObj = report_file_output
    else:
        logger.critical(f'Unable to extract XML from QCTools mkv report file\n')
        startObj = None
    
    return startObj
    

def detectBitdepth(startObj,pkt,framesList,buffSize):
    etree = load_etree()
    if etree is None:
        return False

    bit_depth_10 = False
    
    # Use the safe parser with encoding fallback
    parser_iter = safe_gzip_iterparse(startObj, etree)
    if parser_iter is None:
        logger.error(f"Failed to parse {startObj} for bit depth detection")
        return False

    try:
        for event, elem in parser_iter: # iterparse the xml doc
            if elem.attrib['media_type'] == "video": # get just the video frames
                frame_pkt_dts_time = elem.attrib[pkt] # get the timestamps for the current frame we're looking at
                frameDict = {}  # start an empty dict for the new frame
                frameDict[pkt] = frame_pkt_dts_time  # give the dict the timestamp, which we have now
                for t in list(elem):    # iterating through each attribute for each element
                    keySplit = t.attrib['key'].split(".")   # split the names by dots 
                    keyName = str(keySplit[-1])             # get just the last word for the key name
                    frameDict[keyName] = t.attrib['value']	# add each attribute to the frame dictionary
                framesList.append(frameDict)
                middleFrame = int(round(float(len(framesList))/2))	# i hate this calculation, but it gets us the middle index of the list as an integer
                if len(framesList) == buffSize:	# wait till the buffer is full to start detecting bars
                    ## This is where the bars detection magic actually happens
                    bufferRange = list(range(0, buffSize))
                    if float(framesList[middleFrame]['YMAX']) > 250:
                        bit_depth_10 = True
                        break
                elem.clear() # we're done with that element so let's get it outta memory
    except Exception as e:
        logger.error(f"Error during bit depth detection parsing: {e}")

    return bit_depth_10


AUDIO_CLIPPING_THRESHOLD_DB = -0.5
SILENCE_THRESHOLD_DB = -60.0

# ---------------------------------------------------------------------------
# Audible timecode detection thresholds
# ---------------------------------------------------------------------------

# R128 thresholds (derived from analysis of known TC files)
_TC_R128_WINDOW_SEC = 30          # rolling window size in seconds
_TC_R128_MIN_WINDOWS = 6          # minimum consecutive windows to confirm TC (~3 min)

# Criterion A: dual-channel TC (both channels carry TC)
_TC_R128_A_M_STDEV_MAX = 2.0
_TC_R128_A_M_MEAN_MIN = -25.0
_TC_R128_A_LRA_MEDIAN_MAX = 5.0

# Criterion B: TC + silence (one channel TC, other near-silent)
_TC_R128_B_LRA_HIGH_MIN = -30.0
_TC_R128_B_LRA_HIGH_MAX = -10.0
_TC_R128_B_LRA_HIGH_STDEV_MAX = 2.0
_TC_R128_B_M_I_DIFF_MIN = 20.0
_TC_R128_B_LRA_MIN = 13.0

# Criterion C: TC + program audio
_TC_R128_C_MS_DIFF_MEDIAN_MIN = 1.5
_TC_R128_C_M_STDEV_MIN = 8.0
_TC_R128_C_LRA_HIGH_MIN = -30.0
_TC_R128_C_LRA_HIGH_MAX = -10.0
_TC_R128_C_LRA_HIGH_STDEV_MAX = 3.0
_TC_R128_C_LRA_HIGH_UPPER = -24.0
_TC_R128_C_M_MIN = -60.0

# astats thresholds (per-channel detection)
_TC_ASTATS_WINDOW_FRAMES = 7     # ~7 frames at ~4.6s each ~ 30s window
_TC_ASTATS_MIN_WINDOWS = 2       # minimum consecutive windows to confirm TC

# Per-channel TC indicators in astats
_TC_ASTATS_RMS_LEVEL_MIN = -30.0
_TC_ASTATS_RMS_LEVEL_MAX = -5.0
_TC_ASTATS_RMS_STDEV_MAX = 3.0
_TC_ASTATS_CREST_FACTOR_MAX = 2.0
_TC_ASTATS_DYNAMIC_RANGE_MAX = 20.0
_TC_ASTATS_ZERO_CROSSINGS_RATE_MIN = 0.06
_TC_ASTATS_ZERO_CROSSINGS_RATE_MAX = 0.15
_TC_ASTATS_ENTROPY_MAX = 0.35

# ---------------------------------------------------------------------------
# Audio dropout detection thresholds
# ---------------------------------------------------------------------------

DROPOUT_ROLLING_WINDOW_SIZE = 7       # ~11s of audio at ~1.6s/frame
DROPOUT_RMS_DROP_THRESHOLD_DB = 40.0  # dB drop below rolling median to trigger
DROPOUT_SILENCE_FLOOR_DB = -55.0      # ignore frames where median is below this (natural silence)
DROPOUT_DIFF_DROP_FACTOR = 0.1        # Max/RMS_difference must drop below 10% of rolling median
DROPOUT_ZCR_SPIKE_FACTOR = 3.0        # Zero_crossings_rate must exceed 3x rolling median
DROPOUT_MERGE_GAP_SEC = 3.5           # merge candidates within this gap on same channel
DROPOUT_LONG_EVENT_SEC = 2.0          # events longer than this require high confidence
DROPOUT_LONG_EVENT_MIN_CORR = 2       # minimum corroborating metrics for long events


@dataclass
class _TCDetection:
    """A detected region of audible timecode."""
    start_time: float = 0.0
    end_time: float = 0.0
    criterion: str = ""
    channel: str = ""
    confidence: str = ""
    details: str = ""


@dataclass
class _DropoutCandidate:
    """A single-frame dropout candidate before merging."""
    time: float = 0.0
    channel: int = 0
    rms_level: float = 0.0
    median_rms: float = 0.0
    corroborating: list = field(default_factory=list)


@dataclass
class _DropoutEvent:
    """A merged dropout event (one or more consecutive candidates)."""
    start_time: float = 0.0
    end_time: float = 0.0
    channel: int = 0
    worst_rms_level: float = 0.0
    median_rms_level: float = 0.0
    confidence: str = ""
    corroborating: list = field(default_factory=list)


def analyzeAudio(startObj, pkt, report_directory, detect_clipping=False, detect_imbalance=False, detect_timecode=False, detect_dropout=False, signals=None, total_duration=None):
    """
    Analyzes audio frames in a QCTools report in a single pass. Optionally detects
    audio clipping (Peak_level >= threshold), channel imbalance (comparing
    per-channel RMS_level values), audible timecode (LTC artifacts), and/or audio
    dropout (sudden RMS drops indicative of tape dropout).

    Parameters:
        startObj (str): Path to the QCTools report file (.qctools.xml.gz)
        pkt (str): The attribute key used to extract timestamps (pkt_dts_time or pkt_pts_time).
        report_directory (str): Path to {video_id}_report_csvs directory for CSV output.
        detect_clipping (bool): Whether to run audio clipping detection.
        detect_imbalance (bool): Whether to run channel imbalance detection.
        detect_timecode (bool): Whether to run audible timecode detection.
        detect_dropout (bool): Whether to run audio dropout detection.
        signals: Optional signals object for emitting progress updates.
        total_duration (float or None): Total video duration in seconds for progress reporting.

    Returns:
        tuple: (clipping_results, imbalance_results, timecode_results, dropout_results)
            Each is a dict with analysis results, or None if that analysis was
            not requested or no audio frames were found.
    """
    etree = load_etree()
    if etree is None:
        return None, None, None, None

    parser_iter = safe_gzip_iterparse(startObj, etree)
    if parser_iter is None:
        logger.error(f"Failed to parse {startObj} for audio analysis")
        return None, None, None, None

    total_audio_frames = 0

    # Progress emission setup: maps into the 90→98% range of qctparse_progress
    audio_progress_interval = 500
    last_audio_progress_pct = 90

    # Clipping state
    clipping_events = []
    clipped_frames = 0
    max_peak_level = None
    max_flat_factor = None

    # Channel imbalance state — dict keyed by channel number (int), values are lists of RMS levels
    channel_rms_values = {}

    # Timecode detection state — per-frame data collected during parse
    tc_frames = []  # list of dicts: {time, tags}
    tc_metric_type = 'unknown'

    # Dropout detection state — per-channel rolling windows and candidates
    dropout_candidates = []  # list of _DropoutCandidate
    # Per-channel rolling windows: {ch_num: deque of values}
    dropout_rms_windows = {}
    dropout_max_diff_windows = {}
    dropout_rms_diff_windows = {}
    dropout_zcr_windows = {}

    try:
        for event, elem in parser_iter:
            if elem.attrib.get('media_type') == "audio":
                total_audio_frames += 1
                frame_pkt_dts_time = elem.attrib.get(pkt, "0")

                # Emit incremental progress during audio analysis (90→98% range)
                if (signals and hasattr(signals, 'qctparse_progress') and
                    total_duration and total_audio_frames % audio_progress_interval == 0):
                    pct = 90 + int((float(frame_pkt_dts_time) / total_duration) * 8)
                    pct = min(98, max(90, pct))
                    if pct > last_audio_progress_pct:
                        signals.qctparse_progress.emit(pct)
                        last_audio_progress_pct = pct

                # Parse audio frame attributes in one loop
                peak_level = None
                flat_factor = None
                frame_channel_rms = {}
                tc_frame_tags = {} if detect_timecode else None
                # Dropout per-frame state: {ch_num: {metric: value}}
                dropout_frame_data = {} if detect_dropout else None

                for t in list(elem):
                    key = t.attrib.get('key', '')
                    try:
                        val = float(t.attrib['value'])
                    except (ValueError, KeyError):
                        continue

                    if detect_clipping:
                        if key.endswith('.Peak_level') and 'Overall' in key:
                            peak_level = val
                        elif key.endswith('.Flat_factor') and 'Overall' in key:
                            flat_factor = val

                    if detect_imbalance:
                        # Match lavfi.astats.N.RMS_level for any channel number N
                        if key.endswith('.RMS_level') and 'Overall' not in key:
                            match = re.search(r'\.(\d+)\.RMS_level$', key)
                            if match:
                                ch_num = int(match.group(1))
                                frame_channel_rms[ch_num] = val

                    if detect_timecode:
                        # Collect r128 and astats tags for timecode detection
                        if key.startswith('lavfi.r128') or key.startswith('lavfi.astats'):
                            tc_frame_tags[key] = val
                            if tc_metric_type == 'unknown':
                                if key.startswith('lavfi.r128'):
                                    tc_metric_type = 'r128'
                                elif key.startswith('lavfi.astats'):
                                    tc_metric_type = 'astats'

                    if detect_dropout and 'Overall' not in key:
                        # Extract per-channel dropout metrics
                        for metric in ('RMS_level', 'Max_difference', 'RMS_difference', 'Zero_crossings_rate'):
                            if key.endswith(f'.{metric}'):
                                match = re.search(r'\.(\d+)\.' + metric + '$', key)
                                if match:
                                    ch_num = int(match.group(1))
                                    if ch_num not in dropout_frame_data:
                                        dropout_frame_data[ch_num] = {}
                                    dropout_frame_data[ch_num][metric] = val
                                break

                # Clipping analysis
                if detect_clipping and peak_level is not None:
                    if max_peak_level is None or peak_level > max_peak_level:
                        max_peak_level = peak_level
                    if flat_factor is not None and (max_flat_factor is None or flat_factor > max_flat_factor):
                        max_flat_factor = flat_factor

                    if peak_level >= AUDIO_CLIPPING_THRESHOLD_DB:
                        clipped_frames += 1
                        timeStampString = dts2ts(frame_pkt_dts_time)
                        clipping_events.append({
                            'timestamp': timeStampString,
                            'timestamp_seconds': float(frame_pkt_dts_time),
                            'peak_level': peak_level,
                            'flat_factor': flat_factor
                        })

                # Channel imbalance collection
                if detect_imbalance:
                    for ch_num, rms_val in frame_channel_rms.items():
                        if ch_num not in channel_rms_values:
                            channel_rms_values[ch_num] = []
                        channel_rms_values[ch_num].append(rms_val)

                # Timecode frame collection
                if detect_timecode and tc_frame_tags:
                    try:
                        frame_time = float(frame_pkt_dts_time)
                    except ValueError:
                        frame_time = 0.0
                    tc_frames.append({'time': frame_time, 'tags': tc_frame_tags})

                # Dropout detection — rolling window analysis per channel
                if detect_dropout and dropout_frame_data:
                    try:
                        frame_time = float(frame_pkt_dts_time)
                    except ValueError:
                        frame_time = 0.0

                    for ch_num, metrics in dropout_frame_data.items():
                        rms = metrics.get('RMS_level')
                        if rms is None:
                            continue

                        # Initialize rolling windows for this channel
                        if ch_num not in dropout_rms_windows:
                            dropout_rms_windows[ch_num] = collections.deque(maxlen=DROPOUT_ROLLING_WINDOW_SIZE)
                            dropout_max_diff_windows[ch_num] = collections.deque(maxlen=DROPOUT_ROLLING_WINDOW_SIZE)
                            dropout_rms_diff_windows[ch_num] = collections.deque(maxlen=DROPOUT_ROLLING_WINDOW_SIZE)
                            dropout_zcr_windows[ch_num] = collections.deque(maxlen=DROPOUT_ROLLING_WINDOW_SIZE)

                        rms_win = dropout_rms_windows[ch_num]
                        max_diff_win = dropout_max_diff_windows[ch_num]
                        rms_diff_win = dropout_rms_diff_windows[ch_num]
                        zcr_win = dropout_zcr_windows[ch_num]

                        # Only analyze once window is full
                        if len(rms_win) >= DROPOUT_ROLLING_WINDOW_SIZE:
                            median_rms = sorted(rms_win)[len(rms_win) // 2]

                            # Skip if content is near-silent (avoids false positives)
                            if median_rms >= DROPOUT_SILENCE_FLOOR_DB:
                                rms_drop = median_rms - rms
                                if rms_drop >= DROPOUT_RMS_DROP_THRESHOLD_DB:
                                    # Check corroborating metrics.
                                    # During dropout the signal drops to near-silence, so:
                                    #   - Max_difference DROPS (tiny sample-to-sample jumps)
                                    #   - RMS_difference DROPS (tiny sample-to-sample variation)
                                    #   - Zero_crossings_rate INCREASES (noise hovers around zero)
                                    corroborating = []
                                    max_diff = metrics.get('Max_difference')
                                    if max_diff is not None and len(max_diff_win) >= DROPOUT_ROLLING_WINDOW_SIZE:
                                        median_max_diff = sorted(max_diff_win)[len(max_diff_win) // 2]
                                        if median_max_diff > 0 and max_diff < median_max_diff * DROPOUT_DIFF_DROP_FACTOR:
                                            corroborating.append('Max_difference drop')

                                    rms_diff = metrics.get('RMS_difference')
                                    if rms_diff is not None and len(rms_diff_win) >= DROPOUT_ROLLING_WINDOW_SIZE:
                                        median_rms_diff = sorted(rms_diff_win)[len(rms_diff_win) // 2]
                                        if median_rms_diff > 0 and rms_diff < median_rms_diff * DROPOUT_DIFF_DROP_FACTOR:
                                            corroborating.append('RMS_difference drop')

                                    zcr = metrics.get('Zero_crossings_rate')
                                    if zcr is not None and len(zcr_win) >= DROPOUT_ROLLING_WINDOW_SIZE:
                                        median_zcr = sorted(zcr_win)[len(zcr_win) // 2]
                                        if median_zcr > 0 and zcr > median_zcr * DROPOUT_ZCR_SPIKE_FACTOR:
                                            corroborating.append('Zero_crossings_rate spike')

                                    dropout_candidates.append(_DropoutCandidate(
                                        time=frame_time,
                                        channel=ch_num,
                                        rms_level=rms,
                                        median_rms=median_rms,
                                        corroborating=corroborating,
                                    ))

                        # Update rolling windows (after analysis, so current frame doesn't influence its own detection)
                        rms_win.append(rms)
                        max_diff = metrics.get('Max_difference')
                        if max_diff is not None:
                            max_diff_win.append(max_diff)
                        rms_diff = metrics.get('RMS_difference')
                        if rms_diff is not None:
                            rms_diff_win.append(rms_diff)
                        zcr = metrics.get('Zero_crossings_rate')
                        if zcr is not None:
                            zcr_win.append(zcr)

            elem.clear()
    except Exception as e:
        logger.error(f"Error during audio analysis: {e}")

    if total_audio_frames == 0:
        logger.warning("No audio frames found in QCTools report for audio analysis\n")
        return None, None, None, None

    # Build clipping results
    clipping_results = None
    if detect_clipping:
        clipping_results = _write_clipping_results(
            report_directory, total_audio_frames, clipped_frames,
            max_peak_level, max_flat_factor, clipping_events
        )

    # Build channel imbalance results
    imbalance_results = None
    if detect_imbalance:
        imbalance_results = _write_imbalance_results(
            report_directory, total_audio_frames, channel_rms_values
        )

    # Run audible timecode detection
    timecode_results = None
    if detect_timecode:
        timecode_results = _detect_and_write_timecode_results(
            tc_frames, tc_metric_type, report_directory
        )

    # Run dropout detection
    dropout_results = None
    if detect_dropout:
        dropout_results = _detect_and_write_dropout_results(
            dropout_candidates, report_directory, total_audio_frames
        )

    return clipping_results, imbalance_results, timecode_results, dropout_results


def _write_clipping_results(report_directory, total_audio_frames, clipped_frames, max_peak_level, max_flat_factor, clipping_events):
    """Write audio clipping detection results to CSV and log summary."""
    clipping_detected = clipped_frames > 0
    pct = (clipped_frames / total_audio_frames) * 100 if total_audio_frames > 0 else 0

    results = {
        'clipping_detected': clipping_detected,
        'clipping_events': clipping_events,
        'total_audio_frames': total_audio_frames,
        'clipped_frames': clipped_frames,
        'max_peak_level': max_peak_level if max_peak_level is not None else 0.0,
        'max_flat_factor': max_flat_factor if max_flat_factor is not None else 0.0,
        'threshold_db': AUDIO_CLIPPING_THRESHOLD_DB
    }

    audio_clipping_csv = os.path.join(report_directory, "qct-parse_audio_clipping.csv")
    with open(audio_clipping_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Audio Clipping Detection Results"])
        writer.writerow(["Threshold (dBFS)", AUDIO_CLIPPING_THRESHOLD_DB])
        writer.writerow(["Total Audio Frames", total_audio_frames])
        writer.writerow(["Clipped Frames", clipped_frames])
        writer.writerow(["Clipped Frames (%)", f"{pct:.2f}"])
        writer.writerow(["Max Peak Level (dBFS)", f"{max_peak_level:.1f}" if max_peak_level is not None else "N/A"])
        writer.writerow(["Max Flat Factor", f"{max_flat_factor:.0f}" if max_flat_factor is not None else "N/A"])
        writer.writerow(["Clipping Detected", "Yes" if clipping_detected else "No"])
        writer.writerow([])

        if clipping_events:
            writer.writerow(["Timestamp", "Peak Level (dBFS)", "Flat Factor"])
            for event in clipping_events:
                ff = event.get('flat_factor')
                ff_str = f"{ff:.0f}" if ff is not None else "N/A"
                writer.writerow([event['timestamp'], f"{event['peak_level']:.1f}", ff_str])

    if clipping_detected:
        logger.warning(f"Audio clipping detected: {clipped_frames} frames ({pct:.2f}%) exceeded {AUDIO_CLIPPING_THRESHOLD_DB} dBFS threshold. Max peak: {max_peak_level:.1f} dBFS\n")
    else:
        logger.debug(f"No audio clipping detected. Max peak level: {max_peak_level:.1f} dBFS\n")

    return results


def _characterize_imbalance(abs_diff):
    """Return a human-readable characterization for a given absolute dB difference."""
    if abs_diff < 1.0:
        return "Balanced"
    elif abs_diff < 3.0:
        return "Slight imbalance"
    elif abs_diff < 6.0:
        return "Moderate imbalance"
    else:
        return "Significant imbalance"


def _write_imbalance_results(report_directory, total_audio_frames, channel_rms_values):
    """Compute channel imbalance from per-channel RMS levels and write results to CSV.

    Args:
        report_directory (str): Path to report CSV directory.
        total_audio_frames (int): Total audio frames encountered.
        channel_rms_values (dict): {channel_num: [rms_values...]} for each channel found.
    """
    num_channels = len(channel_rms_values)

    if num_channels == 0:
        logger.warning("No per-channel RMS level data found in QCTools report for channel imbalance analysis\n")
        return None

    sorted_channels = sorted(channel_rms_values.keys())

    if num_channels == 1:
        # Mono — no imbalance to compute, but report the single channel's mean RMS
        ch = sorted_channels[0]
        vals = channel_rms_values[ch]
        mean_rms = sum(vals) / len(vals)
        results = {
            'total_audio_frames': total_audio_frames,
            'num_channels': 1,
            'channel_means': {ch: mean_rms},
            'pairwise': [],
            'overall_characterization': "Mono (single channel)",
        }
        imbalance_csv = os.path.join(report_directory, "qct-parse_channel_imbalance.csv")
        with open(imbalance_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Channel Imbalance Analysis Results"])
            writer.writerow(["Total Audio Frames", total_audio_frames])
            writer.writerow(["Number of Channels", 1])
            writer.writerow([f"Channel {ch} Mean RMS (dBFS)", f"{mean_rms:.1f}"])
            writer.writerow(["Overall Characterization", "Mono (single channel)"])
        logger.debug(f"Channel imbalance analysis: Mono — single channel detected (Ch{ch}: {mean_rms:.1f} dBFS)\n")
        return results

    # Multi-channel: compute per-channel means and pairwise comparisons
    frames_with_all = min(len(channel_rms_values[ch]) for ch in sorted_channels)
    if frames_with_all == 0:
        logger.warning("No per-channel RMS level data found in QCTools report for channel imbalance analysis\n")
        return None

    channel_means = {}
    for ch in sorted_channels:
        vals = channel_rms_values[ch][:frames_with_all]
        channel_means[ch] = sum(vals) / len(vals)

    # Pairwise comparisons between all channel pairs
    pairwise = []
    for i, ch_a in enumerate(sorted_channels):
        for ch_b in sorted_channels[i + 1:]:
            mean_diff = channel_means[ch_a] - channel_means[ch_b]
            abs_diff = abs(mean_diff)
            characterization = _characterize_imbalance(abs_diff)
            if abs_diff >= 1.0:
                louder = f"Channel {ch_a}" if mean_diff > 0 else f"Channel {ch_b}"
            else:
                louder = "Neither"
            pairwise.append({
                'channel_a': ch_a,
                'channel_b': ch_b,
                'mean_difference_db': mean_diff,
                'abs_mean_difference_db': abs_diff,
                'characterization': characterization,
                'louder_channel': louder,
            })

    # Overall characterization: worst pairwise result
    worst = max(pairwise, key=lambda p: p['abs_mean_difference_db'])
    overall_characterization = worst['characterization']

    # Silent channel detection — only when significant imbalance is found
    silent_channels = []
    if overall_characterization == "Significant imbalance":
        for ch in sorted_channels:
            mean_rms = channel_means[ch]
            if mean_rms == float('-inf') or mean_rms < SILENCE_THRESHOLD_DB:
                silent_channels.append(ch)

    results = {
        'total_audio_frames': total_audio_frames,
        'num_channels': num_channels,
        'frames_analyzed': frames_with_all,
        'channel_means': channel_means,
        'pairwise': pairwise,
        'overall_characterization': overall_characterization,
        'silent_channels': silent_channels,
    }

    # Write CSV
    imbalance_csv = os.path.join(report_directory, "qct-parse_channel_imbalance.csv")
    with open(imbalance_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Channel Imbalance Analysis Results"])
        writer.writerow(["Total Audio Frames", total_audio_frames])
        writer.writerow(["Number of Channels", num_channels])
        writer.writerow(["Frames Analyzed", frames_with_all])
        for ch in sorted_channels:
            writer.writerow([f"Channel {ch} Mean RMS (dBFS)", f"{channel_means[ch]:.1f}"])
        writer.writerow([])
        if num_channels == 2:
            # For stereo, keep a simple summary like before
            p = pairwise[0]
            writer.writerow(["Mean Difference (dB)", f"{p['mean_difference_db']:+.1f}"])
            writer.writerow(["Characterization", p['characterization']])
            writer.writerow(["Louder Channel", p['louder_channel']])
        else:
            # For >2 channels, write pairwise comparison table
            writer.writerow(["Pairwise Comparisons"])
            writer.writerow(["Channel A", "Channel B", "Mean Difference (dB)", "Characterization", "Louder Channel"])
            for p in pairwise:
                writer.writerow([
                    f"Channel {p['channel_a']}", f"Channel {p['channel_b']}",
                    f"{p['mean_difference_db']:+.1f}", p['characterization'], p['louder_channel']
                ])
        writer.writerow([])
        writer.writerow(["Overall Characterization", overall_characterization])
        if silent_channels:
            writer.writerow(["Silent Channels", ", ".join(f"Channel {ch}" for ch in silent_channels)])

    # Log summary
    ch_summary = ", ".join(f"Ch{ch}: {channel_means[ch]:.1f} dBFS" for ch in sorted_channels)
    if silent_channels:
        silent_str = ", ".join(f"Ch{ch}" for ch in silent_channels)
        logger.warning(f"Channel imbalance analysis: {overall_characterization} — silent channel(s) detected: {silent_str} ({ch_summary})\n")
    else:
        logger.debug(f"Channel imbalance analysis: {overall_characterization} ({ch_summary})\n")

    return results


# ---------------------------------------------------------------------------
# Audible timecode detection
# ---------------------------------------------------------------------------

def _tc_mean(values):
    if not values:
        return float('nan')
    return sum(values) / len(values)


def _tc_median(values):
    if not values:
        return float('nan')
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def _tc_stdev(values):
    if len(values) < 2:
        return 0.0
    m = _tc_mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _tc_format_time(seconds):
    """Format seconds as HH:MM:SS.s"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f'{h:d}:{m:02d}:{s:04.1f}'
    return f'{m:d}:{s:04.1f}'


def _tc_filter_by_consecutive(detections, min_consecutive):
    """Keep only detections that are part of a consecutive run of sufficient length."""
    if len(detections) < min_consecutive:
        return []

    by_crit = defaultdict(list)
    for d in detections:
        by_crit[d.criterion].append(d)

    result = []
    for crit, dets in by_crit.items():
        dets.sort(key=lambda d: d.start_time)
        run = [dets[0]]
        for d in dets[1:]:
            gap = d.start_time - run[-1].end_time
            if gap <= _TC_R128_WINDOW_SEC * 1.5:
                run.append(d)
            else:
                if len(run) >= min_consecutive:
                    result.extend(run)
                run = [d]
        if len(run) >= min_consecutive:
            result.extend(run)

    return result


def _tc_merge_detections(detections):
    """Merge overlapping detections with the same criterion into spans."""
    if not detections:
        return []

    by_crit = defaultdict(list)
    for d in detections:
        by_crit[d.criterion].append(d)

    merged = []
    for crit, dets in by_crit.items():
        dets.sort(key=lambda d: d.start_time)
        current = _TCDetection(
            start_time=dets[0].start_time,
            end_time=dets[0].end_time,
            criterion=dets[0].criterion,
            channel=dets[0].channel,
            confidence=dets[0].confidence,
            details=dets[0].details
        )
        for d in dets[1:]:
            if d.start_time <= current.end_time + _TC_R128_WINDOW_SEC:
                current.end_time = max(current.end_time, d.end_time)
                if d.confidence == 'high':
                    current.confidence = 'high'
            else:
                merged.append(current)
                current = _TCDetection(
                    start_time=d.start_time,
                    end_time=d.end_time,
                    criterion=d.criterion,
                    channel=d.channel,
                    confidence=d.confidence,
                    details=d.details
                )
        merged.append(current)

    merged.sort(key=lambda d: d.start_time)
    return merged


def _detect_r128_timecode(frames):
    """Detect audible timecode using EBU R128 measurements."""
    if not frames:
        return []

    times = [f['time'] for f in frames]
    m_vals = [f['tags'].get('lavfi.r128.M', float('nan')) for f in frames]
    s_vals = [f['tags'].get('lavfi.r128.S', float('nan')) for f in frames]
    lra_vals = [f['tags'].get('lavfi.r128.LRA', float('nan')) for f in frames]
    lra_high_vals = [f['tags'].get('lavfi.r128.LRA.high', float('nan')) for f in frames]

    # Get integrated loudness from last valid frame
    i_val = float('nan')
    for frame in reversed(frames):
        candidate = frame['tags'].get('lavfi.r128.I', float('nan'))
        if not math.isnan(candidate) and candidate < -1.0:
            i_val = candidate
            break

    # Determine frame interval
    if len(times) >= 2:
        dt_sec = times[1] - times[0]
    else:
        dt_sec = 0.1
    window_size = max(1, int(_TC_R128_WINDOW_SEC / dt_sec))

    detections = []
    detections.extend(_detect_r128_criterion_a(times, m_vals, lra_vals, window_size))
    detections.extend(_detect_r128_criterion_b(times, m_vals, lra_vals, lra_high_vals, i_val, window_size))
    detections.extend(_detect_r128_criterion_c(times, m_vals, s_vals, lra_high_vals, window_size))

    return _tc_merge_detections(detections)


def _detect_r128_criterion_a(times, m_vals, lra_vals, window_size):
    """Criterion A: detect dual-channel TC via rolling windows."""
    detections = []
    for i in range(0, len(times) - window_size + 1, window_size // 2):
        end = min(i + window_size, len(times))
        w_m = [v for v in m_vals[i:end] if not math.isnan(v)]
        w_lra = [v for v in lra_vals[i:end] if not math.isnan(v)]

        if not w_m or not w_lra:
            continue

        m_std = _tc_stdev(w_m)
        m_mean = _tc_mean(w_m)
        lra_med = _tc_median(w_lra)

        if (m_std < _TC_R128_A_M_STDEV_MAX
                and m_mean > _TC_R128_A_M_MEAN_MIN
                and lra_med < _TC_R128_A_LRA_MEDIAN_MAX):
            detections.append(_TCDetection(
                start_time=times[i],
                end_time=times[min(end - 1, len(times) - 1)],
                criterion='R128-A (dual-channel TC)',
                channel='both',
                confidence='high' if m_std < 1.0 else 'medium',
                details=f'M_stdev={m_std:.2f}, M_mean={m_mean:.1f}, LRA_med={lra_med:.1f}'
            ))

    return _tc_filter_by_consecutive(detections, _TC_R128_MIN_WINDOWS)


def _detect_r128_criterion_b(times, m_vals, lra_vals, lra_high_vals, i_val, window_size):
    """Criterion B: TC + silence."""
    detections = []
    for i in range(0, len(times) - window_size + 1, window_size // 2):
        end = min(i + window_size, len(times))
        w_m = [v for v in m_vals[i:end] if not math.isnan(v)]
        w_lra = [v for v in lra_vals[i:end] if not math.isnan(v)]
        w_lra_h = [v for v in lra_high_vals[i:end] if not math.isnan(v)]

        if not w_m or not w_lra_h or not w_lra:
            continue

        lra_h_mean = _tc_mean(w_lra_h)
        lra_h_std = _tc_stdev(w_lra_h)
        m_med = _tc_median(w_m)
        lra_med = _tc_median(w_lra)

        if math.isnan(i_val):
            m_i_diff = 0
        else:
            m_i_diff = abs(m_med - i_val)

        if (_TC_R128_B_LRA_HIGH_MIN <= lra_h_mean <= _TC_R128_B_LRA_HIGH_MAX
                and lra_h_std < _TC_R128_B_LRA_HIGH_STDEV_MAX
                and m_i_diff > _TC_R128_B_M_I_DIFF_MIN
                and lra_med > _TC_R128_B_LRA_MIN):
            detections.append(_TCDetection(
                start_time=times[i],
                end_time=times[min(end - 1, len(times) - 1)],
                criterion='R128-B (TC + silence)',
                channel='one channel',
                confidence='high' if lra_h_std < 0.5 else 'medium',
                details=f'LRA.high={lra_h_mean:.1f}(+/-{lra_h_std:.2f}), |M-I|={m_i_diff:.1f}, LRA={lra_med:.1f}'
            ))

    return _tc_filter_by_consecutive(detections, _TC_R128_MIN_WINDOWS)


def _detect_r128_criterion_c(times, m_vals, s_vals, lra_high_vals, window_size):
    """Criterion C: TC + program audio."""
    detections = []
    for i in range(0, len(times) - window_size + 1, window_size // 2):
        end = min(i + window_size, len(times))
        w_m = [v for v in m_vals[i:end] if not math.isnan(v)]
        w_s = [v for v in s_vals[i:end] if not math.isnan(v)]
        w_lra_h = [v for v in lra_high_vals[i:end] if not math.isnan(v)]

        if not w_m or not w_s or not w_lra_h:
            continue

        ms_diffs = [abs(m - s) for m, s in zip(w_m, w_s)]
        ms_diff_med = _tc_median(ms_diffs)
        m_std = _tc_stdev(w_m)
        lra_h_mean = _tc_mean(w_lra_h)
        lra_h_std = _tc_stdev(w_lra_h)
        m_min = min(w_m)

        if (ms_diff_med > _TC_R128_C_MS_DIFF_MEDIAN_MIN
                and m_std > _TC_R128_C_M_STDEV_MIN
                and _TC_R128_C_LRA_HIGH_MIN <= lra_h_mean <= _TC_R128_C_LRA_HIGH_UPPER
                and lra_h_std < _TC_R128_C_LRA_HIGH_STDEV_MAX
                and m_min > _TC_R128_C_M_MIN):
            detections.append(_TCDetection(
                start_time=times[i],
                end_time=times[min(end - 1, len(times) - 1)],
                criterion='R128-C (TC + program audio)',
                channel='one channel',
                confidence='high' if ms_diff_med > 2.0 else 'medium',
                details=f'|M-S|_med={ms_diff_med:.2f}, M_stdev={m_std:.1f}, LRA.high={lra_h_mean:.1f}(+/-{lra_h_std:.2f})'
            ))

    return _tc_filter_by_consecutive(detections, _TC_R128_MIN_WINDOWS)


def _detect_astats_timecode(frames):
    """Detect audible timecode using FFmpeg astats per-channel measurements."""
    if not frames:
        return []

    # Determine available channels
    channels = set()
    for frame in frames[:5]:
        for key in frame['tags']:
            if key.startswith('lavfi.astats.1.'):
                channels.add('1')
            elif key.startswith('lavfi.astats.2.'):
                channels.add('2')

    if not channels:
        return []

    times = [f['time'] for f in frames]
    detections = []

    for ch in sorted(channels):
        detections.extend(_detect_astats_channel_tc(frames, times, ch))

    # Label dual-channel if both channels flagged at the same time
    ch1 = [d for d in detections if d.channel == 'ch1']
    ch2 = [d for d in detections if d.channel == 'ch2']
    for d1 in ch1:
        for d2 in ch2:
            if d1.start_time <= d2.end_time and d2.start_time <= d1.end_time:
                d1.channel = 'both (ch1)'
                d2.channel = 'both (ch2)'

    return _tc_merge_detections(detections)


def _detect_astats_channel_tc(frames, times, channel):
    """Detect TC on a single channel using rolling windows of astats data."""
    prefix = f'lavfi.astats.{channel}.'
    detections = []
    window = _TC_ASTATS_WINDOW_FRAMES

    for i in range(0, len(frames) - window + 1, max(1, window // 2)):
        end = min(i + window, len(frames))
        w_frames = frames[i:end]

        rms_levels = []
        crest_factors = []
        zcr_rates = []
        entropies = []

        for f in w_frames:
            tags = f['tags']
            rms = tags.get(prefix + 'RMS_level', float('nan'))
            crest = tags.get(prefix + 'Crest_factor', float('nan'))
            zcr = tags.get(prefix + 'Zero_crossings_rate', float('nan'))
            ent = tags.get(prefix + 'Entropy', float('nan'))

            if not math.isnan(rms):
                rms_levels.append(rms)
            if not math.isnan(crest):
                crest_factors.append(crest)
            if not math.isnan(zcr):
                zcr_rates.append(zcr)
            if not math.isnan(ent):
                entropies.append(ent)

        if not rms_levels:
            continue

        rms_mean = _tc_mean(rms_levels)
        rms_std = _tc_stdev(rms_levels)
        crest_med = _tc_median(crest_factors) if crest_factors else float('nan')
        zcr_med = _tc_median(zcr_rates) if zcr_rates else float('nan')
        ent_med = _tc_median(entropies) if entropies else float('nan')

        reasons = []
        passes = True

        if _TC_ASTATS_RMS_LEVEL_MIN <= rms_mean <= _TC_ASTATS_RMS_LEVEL_MAX and rms_std < _TC_ASTATS_RMS_STDEV_MAX:
            reasons.append(f'RMS={rms_mean:.1f}dB(+/-{rms_std:.1f})')
        else:
            passes = False

        if not math.isnan(crest_med) and crest_med < _TC_ASTATS_CREST_FACTOR_MAX:
            reasons.append(f'Crest={crest_med:.1f}dB')
        else:
            passes = False

        if not math.isnan(ent_med) and ent_med < _TC_ASTATS_ENTROPY_MAX:
            reasons.append(f'Entropy={ent_med:.3f}')
        else:
            passes = False

        if not math.isnan(zcr_med) and _TC_ASTATS_ZERO_CROSSINGS_RATE_MIN <= zcr_med <= _TC_ASTATS_ZERO_CROSSINGS_RATE_MAX:
            reasons.append(f'ZCR={zcr_med:.3f}')
        else:
            passes = False

        if passes:
            confidence = 'high' if crest_med < 1.5 and ent_med < 0.25 else 'medium'
            detections.append(_TCDetection(
                start_time=times[i],
                end_time=times[min(end - 1, len(times) - 1)],
                criterion=f'astats (ch{channel})',
                channel=f'ch{channel}',
                confidence=confidence,
                details=', '.join(reasons)
            ))

    return _tc_filter_by_consecutive(detections, _TC_ASTATS_MIN_WINDOWS)


def _detect_and_write_timecode_results(tc_frames, tc_metric_type, report_directory):
    """Run audible timecode detection and write results to CSV."""
    if not tc_frames:
        logger.debug("No audio frame data available for audible timecode detection\n")
        return None

    if tc_metric_type == 'r128':
        detections = _detect_r128_timecode(tc_frames)
    elif tc_metric_type == 'astats':
        detections = _detect_astats_timecode(tc_frames)
    else:
        logger.debug("No recognized audio metrics (r128 or astats) found for timecode detection\n")
        return None

    tc_detected = len(detections) > 0
    duration = tc_frames[-1]['time'] if tc_frames else 0

    results = {
        'timecode_detected': tc_detected,
        'metric_type': tc_metric_type,
        'total_audio_frames': len(tc_frames),
        'duration': duration,
        'detections': [
            {
                'start_time': d.start_time,
                'end_time': d.end_time,
                'criterion': d.criterion,
                'channel': d.channel,
                'confidence': d.confidence,
                'details': d.details,
            }
            for d in detections
        ],
    }

    # Write CSV
    tc_csv = os.path.join(report_directory, "qct-parse_audible_timecode.csv")
    with open(tc_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Audible Timecode Detection Results"])
        writer.writerow(["Metric Type", tc_metric_type])
        writer.writerow(["Total Audio Frames", len(tc_frames)])
        writer.writerow(["Duration", _tc_format_time(duration)])
        writer.writerow(["Audible Timecode Detected", "Yes" if tc_detected else "No"])
        writer.writerow(["Regions Detected", len(detections)])
        writer.writerow([])

        if detections:
            writer.writerow(["Start Time", "End Time", "Criterion", "Channel", "Confidence", "Details"])
            for d in detections:
                writer.writerow([
                    _tc_format_time(d.start_time),
                    _tc_format_time(d.end_time),
                    d.criterion,
                    d.channel,
                    d.confidence,
                    d.details,
                ])

    # Log summary
    if tc_detected:
        regions = "; ".join(
            f"{_tc_format_time(d.start_time)}-{_tc_format_time(d.end_time)} [{d.confidence}] {d.criterion}"
            for d in detections
        )
        logger.warning(f"Audible timecode detected: {len(detections)} region(s) — {regions}\n")
    else:
        logger.debug("No audible timecode detected\n")

    return results


def _get_video_duration(video_path):
    """Get video duration in seconds using ffprobe.

    Returns:
        float or None: Duration in seconds, or None if unavailable.
    """
    try:
        command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def _merge_dropout_candidates(candidates):
    """Merge consecutive dropout candidates into events.

    Candidates on the same channel within DROPOUT_MERGE_GAP_SEC of each other
    are merged into a single _DropoutEvent.
    """
    if not candidates:
        return []

    # Sort by channel then time
    sorted_cands = sorted(candidates, key=lambda c: (c.channel, c.time))

    events = []
    current = sorted_cands[0]
    event_start = current.time
    event_end = current.time
    event_channel = current.channel
    worst_rms = current.rms_level
    best_median = current.median_rms
    all_corroborating = set(current.corroborating)
    frames_in_event = 1

    for cand in sorted_cands[1:]:
        if cand.channel == event_channel and (cand.time - event_end) <= DROPOUT_MERGE_GAP_SEC:
            # Merge into current event
            event_end = cand.time
            if cand.rms_level < worst_rms:
                worst_rms = cand.rms_level
            best_median = max(best_median, cand.median_rms)
            all_corroborating.update(cand.corroborating)
            frames_in_event += 1
        else:
            # Finalize current event
            n_corr = len(all_corroborating)
            if n_corr >= 2:
                confidence = 'high'
            elif n_corr == 1:
                confidence = 'medium'
            else:
                confidence = 'low'
            events.append(_DropoutEvent(
                start_time=event_start,
                end_time=event_end,
                channel=event_channel,
                worst_rms_level=worst_rms,
                median_rms_level=best_median,
                confidence=confidence,
                corroborating=sorted(all_corroborating),
            ))
            # Start new event
            event_start = cand.time
            event_end = cand.time
            event_channel = cand.channel
            worst_rms = cand.rms_level
            best_median = cand.median_rms
            all_corroborating = set(cand.corroborating)
            frames_in_event = 1

    # Finalize last event
    n_corr = len(all_corroborating)
    if n_corr >= 2:
        confidence = 'high'
    elif n_corr == 1:
        confidence = 'medium'
    else:
        confidence = 'low'
    events.append(_DropoutEvent(
        start_time=event_start,
        end_time=event_end,
        channel=event_channel,
        worst_rms_level=worst_rms,
        median_rms_level=best_median,
        confidence=confidence,
        corroborating=sorted(all_corroborating),
    ))

    return events


def _detect_and_write_dropout_results(dropout_candidates, report_directory, total_audio_frames):
    """Merge dropout candidates into events, write CSV, and return results dict."""

    events = _merge_dropout_candidates(dropout_candidates)

    # Filter out long events without strong corroboration.
    # Real tape dropout is very brief; multi-second events with no
    # corroborating metrics are almost certainly false positives.
    events = [
        ev for ev in events
        if (ev.end_time - ev.start_time) <= DROPOUT_LONG_EVENT_SEC
        or len(ev.corroborating) >= DROPOUT_LONG_EVENT_MIN_CORR
    ]

    dropout_detected = len(events) > 0
    frames_flagged = len(dropout_candidates)

    results = {
        'dropout_detected': dropout_detected,
        'dropout_events': len(events),
        'frames_flagged': frames_flagged,
        'total_audio_frames': total_audio_frames,
        'events': events,
    }

    audio_dropout_csv = os.path.join(report_directory, "qct-parse_audio_dropout.csv")
    with open(audio_dropout_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Audio Dropout Detection Results"])
        writer.writerow(["Rolling Window Size (frames)", DROPOUT_ROLLING_WINDOW_SIZE])
        writer.writerow(["RMS Drop Threshold (dB)", DROPOUT_RMS_DROP_THRESHOLD_DB])
        writer.writerow(["Silence Floor (dBFS)", DROPOUT_SILENCE_FLOOR_DB])
        writer.writerow(["Total Audio Frames", total_audio_frames])
        writer.writerow(["Dropout Events Detected", len(events)])
        writer.writerow(["Frames Flagged", frames_flagged])
        writer.writerow(["Dropout Detected", "Yes" if dropout_detected else "No"])
        writer.writerow([])

        if events:
            writer.writerow(["Timestamp Start", "Timestamp End", "Channel",
                             "Worst RMS (dBFS)", "Median RMS (dBFS)", "Drop (dB)",
                             "Confidence", "Corroborating Metrics"])
            for ev in events:
                drop_db = ev.median_rms_level - ev.worst_rms_level
                writer.writerow([
                    dts2ts(str(ev.start_time)),
                    dts2ts(str(ev.end_time)),
                    ev.channel,
                    f"{ev.worst_rms_level:.1f}",
                    f"{ev.median_rms_level:.1f}",
                    f"{drop_db:.1f}",
                    ev.confidence,
                    ", ".join(ev.corroborating) if ev.corroborating else "None",
                ])

    if dropout_detected:
        high_count = sum(1 for e in events if e.confidence == 'high')
        med_count = sum(1 for e in events if e.confidence == 'medium')
        low_count = sum(1 for e in events if e.confidence == 'low')
        logger.warning(
            f"Audio dropout detected: {len(events)} event(s) "
            f"({high_count} high, {med_count} medium, {low_count} low confidence), "
            f"{frames_flagged} frame(s) flagged\n"
        )
    else:
        logger.debug("No audio dropout detected\n")

    return results


def run_qctparse(video_path, qctools_output_path, report_directory, check_cancelled=None, signals=None):
    """
    Executes the qct-parse analysis on a given video file, exporting relevant data and thumbnails based on specified thresholds and profiles.

    Parameters:
        video_path (str): Path to the video file being analyzed.
        qctools_output_path (str): Path to the QCTools XML report output.
        report_directory (str): Path to {video_id}_report_csvs directory.
        check_cancelled (callable): Optional function to check if processing was cancelled.
        signals: Optional signals object for emitting progress updates.

    """
    
    checks_config = config_mgr.get_config('checks', ChecksConfig)
    spex_config = config_mgr.get_config('spex', SpexConfig)

    fullTagList = getFullTagList()

    # Check if we can load required library
    etree = load_etree()
    if etree is None:
        logger.critical("Cannot proceed with qct-parse: required library lxml.etree is not available")
        return None
    
    logger.info("Starting qct-parse\n")

    # Estimate total video frames for progress reporting
    total_duration = _get_video_duration(video_path)

    ###### Initialize variables ######
    startObj = qctools_output_path
    
    qct_parse = asdict(checks_config.tools.qct_parse)

    # Get the extension from the actual qctools file path
    if '.qctools.mkv' in qctools_output_path:
        qctools_ext = 'qctools.mkv'
    elif '.qctools.xml.gz' in qctools_output_path:
        qctools_ext = 'qctools.xml.gz'
    else:
        # Fallback to config if we can't determine from path
        qctools_ext = checks_config.outputs.qctools_ext

    if qctools_ext.lower().endswith('mkv'):
        startObj = extract_report_mkv(startObj, qctools_output_path)

    # Initalize circular buffer for efficient xml parsing
    buffSize = int(11)
    framesList = collections.deque(maxlen=buffSize) # init framesList

    # Set parentDir and baseName
    parentDir = os.path.dirname(startObj)
    baseName = (os.path.basename(startObj)).split('.')[0]

    # Initialize thumbExport delay, will be updated per use case
    thumbDelay = 9000
    thumbExportDelay = thumbDelay

    # initialize the start and end duration times variables
    durationStart = 0
    durationEnd = 99999999

    # set the path for the thumbnail export
    thumbPath = os.path.join(report_directory, "ThumbExports")
    if qct_parse['thumbExport']:
        if not os.path.exists(thumbPath):
            os.makedirs(thumbPath)
        else:
            archive_result = archiveThumbs(thumbPath)
            if archive_result:
                logger.debug(f"Archived thumbnails to {archive_result}\n")

    profile = {}  # init a dictionary where we'll store reference values from config.yaml file

    # init a list of every tag available in a QCTools Report from the fullTagList in the config.yaml
    tagList = list(fullTagList.keys())

    # open qctools report 
    # determine if report stores pkt_dts_time or pkt_pts_time
    with gzip.open(startObj) as xml:    
        for event, elem in etree.iterparse(xml, events=('end',), tag='frame'):  # iterparse the xml doc
            if elem.attrib['media_type'] == "video":  # get just the video frames
                # we gotta find out if the qctools report has pkt_dts_time or pkt_pts_time ugh
                match = re.search(r"pkt_.ts_time", etree.tostring(elem).decode('utf-8'))
                if match:
                    pkt = match.group()
                    break

    # Determine if video values are 10 bit depth
    bit_depth_10 = detectBitdepth(startObj,pkt,framesList,buffSize)

    if signals and hasattr(signals, 'qctparse_progress'):
        signals.qctparse_progress.emit(5)

    if check_cancelled():
        return None

    ######## Iterate Through the XML for Bars detection ########
    if qct_parse['barsDetection']:
        durationStart = ""                            # if bar detection is turned on then we have to calculate this
        durationEnd = ""                            # if bar detection is turned on then we have to calculate this
        logger.debug(f"Starting Bars Detection on {baseName}\n")
        qctools_colorbars_duration_output = os.path.join(report_directory, "qct-parse_colorbars_durations.csv")
        durationStart, durationEnd, barsStartString, barsEndString = detectBars(startObj,pkt,durationStart,durationEnd,framesList,buffSize,bit_depth_10,signals=signals,total_duration=total_duration)
        
        # Handle case where bars start was found but no end transition was detected
        # This happens when the entire video is color bars
        if barsStartString and not barsEndString:
            logger.debug("Color bars start detected but no end found - checking if entire video is color bars\n")
            # Reset framesList for validation
            framesList.clear()
            is_entire_video_bars, video_end_time = validateEntireVideoAsBars(startObj, pkt, durationStart, framesList, buffSize, bit_depth_10)
            
            if is_entire_video_bars and video_end_time is not None:
                durationEnd = video_end_time
                barsEndString = dts2ts(str(video_end_time))
                logger.info(f"Entire video confirmed as color bars - setting end duration to {barsEndString}\n")
            else:
                logger.warning("Could not confirm entire video as color bars\n")
        
        if durationStart == "" and durationEnd == "":
            logger.error("No color bars detected\n")
            print_bars_durations(qctools_colorbars_duration_output, barsStartString, barsEndString)
        if barsStartString and barsEndString:
            print_bars_durations(qctools_colorbars_duration_output, barsStartString, barsEndString)
            if qct_parse['thumbExport']:
                barsStampString = dts2ts(durationStart)
                printThumb(video_path, "bars_found", "color_bars_detection", startObj,thumbPath, "first_frame", barsStampString)

    if check_cancelled():
        return None

    if signals and hasattr(signals, 'qctparse_progress'):
        signals.qctparse_progress.emit(13)

    ######## Iterate Through the XML for Bars Evaluation ########
    if qct_parse['evaluateBars']:
        bars_fallback = False
        
        if qct_parse['barsDetection'] and durationStart == "" and durationEnd == "":
            logger.warning(f"No color bars found - falling back to SMPTE color bars values from config.\n")
            maxBarsDict = asdict(spex_config.qct_parse_values.smpte_color_bars)
            bars_fallback = True
        elif qct_parse['barsDetection'] and durationStart != "" and durationEnd != "":
            if signals and hasattr(signals, 'qctparse_progress'):
                signals.qctparse_progress.emit(14)
            maxBarsDict = evalBars(startObj,pkt,durationStart,durationEnd,framesList,buffSize)
            if signals and hasattr(signals, 'qctparse_progress'):
                signals.qctparse_progress.emit(16)
            if maxBarsDict is None:
                logger.critical("Something went wrong - Cannot run evaluate color bars\n")
        else:
            logger.critical("Cannot run color bars evaluation without running Bars Detection.")

        if maxBarsDict is not None:
            logger.debug(f"Starting qct-parse color bars evaluation on {baseName}\n")
            smpte_color_bars = asdict(spex_config.qct_parse_values.smpte_color_bars)
            colorbars_values_output = os.path.join(report_directory, "qct-parse_colorbars_values.csv")
            
            if bars_fallback:
                with open(colorbars_values_output, 'w') as f:
                    f.write("SMPTE_FALLBACK\n")
            else:
                print_color_bar_values(baseName, smpte_color_bars, maxBarsDict, colorbars_values_output)
            
            durationStart = 0
            durationEnd = 99999999
            profile = maxBarsDict
            profile_name = 'color_bars_evaluation'
            thumbExportDelay = 9000
            if signals and hasattr(signals, 'qctparse_progress'):
                signals.qctparse_progress.emit(18)
            kbeyond, frameCount, overallFrameFail, failureInfo = analyzeIt(qct_parse, video_path, profile, profile_name, startObj, pkt, durationStart, durationEnd, thumbPath, thumbDelay, thumbExportDelay, framesList, frameCount=0, overallFrameFail=0, adhoc_tag=False, check_cancelled=check_cancelled, signals=signals, total_duration=total_duration)
            colorbars_eval_fails_csv_path = os.path.join(report_directory, "qct-parse_colorbars_eval_failures.csv")
            if failureInfo:
                save_failures_to_csv(failureInfo, colorbars_eval_fails_csv_path)
            qctools_bars_eval_check_output = os.path.join(report_directory, "qct-parse_colorbars_eval_summary.csv")
            printresults(profile, kbeyond, frameCount, overallFrameFail, qctools_bars_eval_check_output)
            logger.debug(f"qct-parse bars evaluation complete. qct-parse summary written to {qctools_bars_eval_check_output}\n")

    if check_cancelled():
        return None

    ######## Audio Analysis (Clipping Detection / Channel Imbalance / Audible Timecode) ########
    do_audio_analysis = qct_parse.get('audio_analysis', False)
    if do_audio_analysis:
        logger.debug(f"Starting audio analysis on {baseName}\n")
        clipping_results, imbalance_results, timecode_results, dropout_results = analyzeAudio(
            startObj, pkt, report_directory,
            detect_clipping=True,
            detect_imbalance=True,
            detect_timecode=True,
            detect_dropout=True,
            signals=signals, total_duration=total_duration
        )
        if clipping_results is None:
            logger.warning("Audio clipping detection could not be performed\n")
        if imbalance_results is None:
            logger.warning("Channel imbalance analysis could not be performed\n")
        if timecode_results is None:
            logger.warning("Audible timecode detection could not be performed\n")
        if dropout_results is None:
            logger.warning("Audio dropout detection could not be performed\n")

    if check_cancelled():
        return None

    logger.info(f"qct-parse finished processing file: {os.path.basename(startObj)} \n")

    if signals and hasattr(signals, 'qctparse_progress'):
        signals.qctparse_progress.emit(100)

    return


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #    print("Usage: python qct-parse.py <input_video> <qctools_report>")
    #    sys.exit(1)
    video_path = sys.argv[1]
    report_path = sys.argv[2]
    qctools_check_output = os.path.dirname(video_path)
    if not os.path.isfile(report_path):
        print(f"Error: {report_path} is not a valid file.")
        sys.exit(1)
    run_qctparse(video_path, report_path, qctools_check_output)