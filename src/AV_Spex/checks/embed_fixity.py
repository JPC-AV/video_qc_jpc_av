import xml.etree.ElementTree as ET
import subprocess
import tempfile
import os
import time
import re
import json

from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ChecksConfig
from AV_Spex.utils.config_manager import ConfigManager

def get_stream_hash_algorithm():
    """Get the configured stream hash algorithm from config."""
    config_mgr = ConfigManager()
    checks_config = config_mgr.get_config('checks', ChecksConfig)
    return getattr(checks_config.fixity, 'stream_hash_algorithm', 'md5').lower()


def detect_hash_algorithm(hash_string):
    """
    Detect the hash algorithm used based on the hash string length.
    
    Args:
        hash_string: The hash string to analyze
        
    Returns:
        str: 'md5', 'sha256', or None if unknown
    """
    if hash_string is None:
        return None
    
    hash_length = len(hash_string)
    if hash_length == 32:
        return 'md5'
    elif hash_length == 64:
        return 'sha256'
    else:
        return None

def get_total_frames(video_path):
    """
    Get the total number of video frames using the fastest available method.
    Uses metadata if available, falls back to duration × framerate,
    and only uses count_packets as a last resort.
    """
    # Method 1: Try to get nb_frames metadata directly
    metadata_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames',
        '-of', 'csv=p=0',
        video_path
    ]
    
    result = subprocess.run(metadata_cmd, stdout=subprocess.PIPE, text=True)
    frames = result.stdout.strip()
    
    # If frames metadata exists and is valid
    if frames and frames.isdigit() and int(frames) > 0:
        return int(frames)
    
    # Method 2: Calculate using duration and framerate
    duration_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=duration,r_frame_rate',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(duration_cmd, stdout=subprocess.PIPE, text=True)
    try:
        data = json.loads(result.stdout)
        stream = data['streams'][0]
        
        # Some files might not have duration in the stream info
        if 'duration' in stream:
            duration = float(stream['duration'])
        else:
            # Fallback to format duration
            format_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                video_path
            ]
            format_result = subprocess.run(format_cmd, stdout=subprocess.PIPE, text=True)
            duration = float(format_result.stdout.strip())
        
        # Parse framerate (often in the format "num/den")
        framerate_str = stream['r_frame_rate']
        if '/' in framerate_str:
            num, den = map(int, framerate_str.split('/'))
            framerate = num / den
        else:
            framerate = float(framerate_str)
        
        # Calculate frame count
        if duration > 0 and framerate > 0:
            return int(duration * framerate)
    except (json.JSONDecodeError, KeyError, ValueError, ZeroDivisionError):
        pass  # Fall through to the slow method if calculation fails
    
    # Method 3 (slowest): Fall back to counting packets
    count_cmd = [
        'ffprobe',
        '-v', 'error',
        '-threads', '0',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(count_cmd, stdout=subprocess.PIPE)
    try:
        total_frames = int(result.stdout.decode().strip())
        return total_frames
    except (ValueError, UnicodeDecodeError):
        # If all methods fail, return a reasonable default
        return 1000  # A reasonable guess to allow progress to be shown


def make_stream_hash(video_path, check_cancelled=None, signals=None):
    """Calculate checksum of video and audio streams using ffmpeg."""   
    
    algorithm = get_stream_hash_algorithm()

    if not signals:
        print(f"\rFFmpeg 'streamhash' Progress: Initializing...", end='', flush=True)
    total_frames = get_total_frames(video_path)
    video_hash = None
    audio_hash = None
    last_update_time = time.time()
    update_interval = 0.1  # Update every 100ms

    # Use compiled regex patterns for faster matching
    frame_pattern = re.compile(r'frame=(\d+)')
    
    # Multi-threading and efficient buffer handling
    ffmpeg_command = [
        'ffmpeg',
        '-hide_banner', '-progress', 'pipe:1', '-nostats', '-loglevel', 'error',
        '-threads', '0',  # Use all available CPU cores
        '-i', video_path,
        '-map', '0',
        '-f', 'streamhash',
        '-hash', algorithm,
        '-'
    ]
    
    # Constants for parsing - adjust based on algorithm
    frame_prefix = 'frame='
    if algorithm == 'sha256':
        video_hash_prefix = '0,v,SHA256'
        audio_hash_prefix = '1,a,SHA256'
    else:
        video_hash_prefix = '0,v,MD5'
        audio_hash_prefix = '1,a,MD5'
    
    # Update progress less frequently (every 10 frames or so)
    update_frequency = max(1, total_frames // 100)
    last_update_frame = 0
    
    try:
        with subprocess.Popen(
            ffmpeg_command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            bufsize=1  # Line buffering
        ) as ffmpeg_process:
            for line in ffmpeg_process.stdout:
                # Check for cancellation
                if check_cancelled and check_cancelled():
                    ffmpeg_process.terminate()
                    return None
                    
                # Extract frame number and update progress
                if line.startswith(frame_prefix):
                    current_frame = int(line[len(frame_prefix):])
                    
                    # Only update GUI when needed (not every frame)
                    if current_frame - last_update_frame >= update_frequency:
                        percent_complete = min(100, int((current_frame * 100) / total_frames))
                        
                        current_time = time.time()
                        if current_time - last_update_time > update_interval:
                            if signals:
                                signals.stream_hash_progress.emit(percent_complete)
                            else:
                                print(f"\rFFmpeg 'streamhash' Progress:                         ", end='', flush=True)
                                print(f"\rFFmpeg 'streamhash' Progress: {percent_complete:.2f}%", end='', flush=True)
                            last_update_time = current_time
                            last_update_frame = current_frame
                        
                # Extract hash values efficiently
                elif line.startswith(video_hash_prefix):
                    video_hash = line.split('=')[1].strip()
                elif line.startswith(audio_hash_prefix):
                    audio_hash = line.split('=')[1].strip()
                    
                # Early termination if we have both hashes and not needing to track progress
                if video_hash and audio_hash and not signals:
                    break
    finally:
        # Final progress update if using console output
        if not signals:
            print(f"\rFFmpeg 'streamhash' Progress: 100.00%", end='', flush=True)
    
    return video_hash, audio_hash


def extract_tags(video_path):
    command = f"mkvextract tags {video_path}"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout


def remove_existing_stream_hashes(xml_tags):
    """
    Remove any existing VIDEO_STREAM_HASH and AUDIO_STREAM_HASH tags from the XML.
    """
    root = ET.fromstring(xml_tags)

    for tag in root.findall('.//Tag'):
        for simple in list(tag.findall('.//Simple')):
            name_element = simple.find('Name')
            if name_element is not None and name_element.text in ["VIDEO_STREAM_HASH", "AUDIO_STREAM_HASH"]:
                tag.remove(simple)

    return ET.tostring(root, encoding="unicode")


def add_stream_hash_tag(xml_tags, video_hash, audio_hash):
    root = ET.fromstring(xml_tags)

    # Find 'Tag' elements
    tags = root.findall('.//Tag')

    # The tag describing the whole file do not contain the element "Targets", loop through tags to find whole file <Tag>
    tags_without_target = []
    for tag in tags:
        if tag.find('Targets'):
            continue
        else:
            tags_without_target.append(tag)

    # Assigns last_tag to last tag not containing element "Targets"
    last_tag = tags_without_target[-1]

    # Create a new 'Simple' element
    video_md5_tag = ET.Element("Simple")
    name = ET.SubElement(video_md5_tag, "Name")
    name.text = "VIDEO_STREAM_HASH"
    string = ET.SubElement(video_md5_tag, "String")
    string.text = video_hash
    tag_language = ET.SubElement(video_md5_tag, "TagLanguageIETF")
    tag_language.text = "und"

    # Create a new 'Simple' element
    audio_md5_tag = ET.Element("Simple")
    name = ET.SubElement(audio_md5_tag, "Name")
    name.text = "AUDIO_STREAM_HASH"
    string = ET.SubElement(audio_md5_tag, "String")
    string.text = audio_hash
    tag_language = ET.SubElement(audio_md5_tag, "TagLanguageIETF")
    tag_language.text = "und"

    # insert new stream_hash subelement into last_tag
    last_tag.insert(-1, video_md5_tag)
    # insert new stream_hash subelement into last_tag
    last_tag.insert(-1, audio_md5_tag)
    # remove last tag from XML
    root.remove(last_tag)
    # insert last_tag to top of <Tags> tree
    root.insert(0, last_tag)

    return ET.tostring(root, encoding="unicode")


def write_tags_to_temp_file(xml_tags):
    # Create a temporary XML file
    with tempfile.NamedTemporaryFile(mode='w+', suffix=".xml", delete=False) as temp_file:
        temp_file.write(xml_tags)
        temp_file_path = temp_file.name
    return temp_file_path


def write_tags_to_mkv(mkv_file, temp_xml_file):
    command = f'mkvpropedit --tags "global:{temp_xml_file}" "{mkv_file}"'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Check if there's any output
    if stdout:
        # Modify the output as needed
        modified_output = stdout.decode('utf-8').replace('Done.', '')
        logger.info(f'Running mkvpropedit:\n{modified_output}')  # Or do something else with the modified output
    if stderr:
        logger.critical(f"Running mkvpropedit:\n{stderr.decode('utf-8')}")  # Print any errors if they occur


def extract_hashes(xml_tags):
    video_hash = None
    audio_hash = None

    root = ET.fromstring(xml_tags)

    # Find 'video_stream_hash' element
    v_stream_element = root.find('.//Simple[Name="VIDEO_STREAM_HASH"]/String')
    if v_stream_element is not None:
        # Assign MD5 in VIDEO_STREAM_HASH to video_hash
        video_hash = v_stream_element.text

    # Find 'video_stream_hash' element
    a_stream_element = root.find('.//Simple[Name="AUDIO_STREAM_HASH"]/String')
    if a_stream_element is not None:
        # Assign MD5 in AUDIO_STREAM_HASH to audio_hash
        audio_hash = a_stream_element.text

    return video_hash, audio_hash


def compare_hashes(existing_video_hash, existing_audio_hash, video_hash, audio_hash):
    """
    Compare stored hashes with newly computed hashes and log results.
    """
    # Detect algorithm for logging purposes
    algorithm = detect_hash_algorithm(existing_video_hash)
    algorithm_display = algorithm.upper() if algorithm else "Unknown"
    
    logger.info(f"Comparing {algorithm_display} hashes:\n")
    
    if existing_video_hash == video_hash:
        logger.info("✓ Video hashes match.")
    else:
        logger.critical(f"✗ Video hashes do not match.\n"
                       f"  Stored:    {existing_video_hash}\n"
                       f"  Computed:  {video_hash}\n")

    if existing_audio_hash == audio_hash:
        logger.info("✓ Audio hashes match.\n")
    else:
        logger.critical(f"✗ Audio hashes do not match.\n"
                       f"  Stored:    {existing_audio_hash}\n"
                       f"  Computed:  {audio_hash}\n")


def embed_fixity(video_path, check_cancelled=None, signals=None):
    """
    Generates new stream hashes and embeds them in the MKV file.
    Handles removal of old hashes if overwriting.
    """
    config_mgr = ConfigManager()
    checks_config = config_mgr.get_config('checks', ChecksConfig)
    
    logger.debug('Generating video and audio stream hashes. This may take a moment...')
    hash_result = make_stream_hash(video_path, check_cancelled=check_cancelled, signals=signals)
    if hash_result is None:
        return None
    video_hash, audio_hash = hash_result
    logger.debug('')
    logger.info(f'Video hash = {video_hash}\nAudio hash = {audio_hash}\n')

    if check_cancelled():
        return None

    existing_tags = extract_tags(video_path)

    if check_cancelled():
        return None

    if existing_tags:
        existing_video_hash, existing_audio_hash = extract_hashes(existing_tags)
        checks_config = config_mgr.get_config('checks', ChecksConfig)

        if existing_video_hash or existing_audio_hash:
            # Now using boolean check
            if checks_config.fixity.overwrite_stream_fixity:
                logger.debug('Removing old stream hashes before embedding new ones.')
                cleaned_tags = remove_existing_stream_hashes(existing_tags)
                updated_tags = add_stream_hash_tag(cleaned_tags, video_hash, audio_hash)
            else:
                logger.error("embed_fixity() called but stream hashes already exist and overwrite is disabled. Aborting embedding.")
                return
        else:
            # No existing stream hashes, so just add new ones
            updated_tags = add_stream_hash_tag(existing_tags, video_hash, audio_hash)
    else:
        logger.critical("mkvextract unable to extract MKV tags! Unable to embed stream hashes.\n")
        return

    if check_cancelled():
        return None

    temp_xml_file = write_tags_to_temp_file(updated_tags)

    logger.debug('Embedding video and audio stream hashes to XML in MKV file.')
    write_tags_to_mkv(video_path, temp_xml_file)

    os.remove(temp_xml_file)



def validate_embedded_md5(video_path, check_cancelled=None, signals=None):
    """
    Validates embedded stream hashes against newly computed hashes.
    Checks for algorithm mismatch before computing to avoid wasted effort.
    """
    if check_cancelled():
        return None
    
    configured_algorithm = get_stream_hash_algorithm()

    logger.debug('Extracting existing video and audio stream hashes\n')
    existing_tags = extract_tags(video_path)
    
    if not existing_tags:
        logger.critical("mkvextract unable to extract MKV tags! Cannot validate stream hashes.\n")
        return
    
    existing_video_hash, existing_audio_hash = extract_hashes(existing_tags)
    
    # Check if hashes exist
    if existing_video_hash is None:
        logger.warning('No video stream hash found\n')
        embed_fixity(video_path, check_cancelled=check_cancelled, signals=signals)
        return
    
    if existing_audio_hash is None:
        logger.warning('No audio stream hash found\n')
        embed_fixity(video_path, check_cancelled=check_cancelled, signals=signals)
        return
    
    # Detect algorithms from stored hashes
    video_algorithm = detect_hash_algorithm(existing_video_hash)
    audio_algorithm = detect_hash_algorithm(existing_audio_hash)
    
    # Display found hashes and detected algorithms
    logger.info(f'Video stream hash found: {existing_video_hash}')
    if video_algorithm:
        logger.info(f'  Detected algorithm: {video_algorithm.upper()}')
    else:
        logger.warning(f'  Could not detect algorithm from hash length ({len(existing_video_hash)})')
    
    logger.info(f'Audio stream hash found: {existing_audio_hash}')
    if audio_algorithm:
        logger.info(f'  Detected algorithm: {audio_algorithm.upper()}')
    else:
        logger.warning(f'  Could not detect algorithm from hash length ({len(existing_audio_hash)})')
    
    logger.info(f'Configured algorithm: {configured_algorithm.upper()}\n')
    
    # Check for algorithm mismatch - stop before computing if mismatch detected
    if video_algorithm and video_algorithm != configured_algorithm:
        logger.error(f'ALGORITHM MISMATCH: Stored video hash uses {video_algorithm.upper()}, '
                    f'but configured algorithm is {configured_algorithm.upper()}')
        logger.error('Validation stopped. Please update configuration or re-embed with correct algorithm.\n')
        return
    
    if audio_algorithm and audio_algorithm != configured_algorithm:
        logger.error(f'ALGORITHM MISMATCH: Stored audio hash uses {audio_algorithm.upper()}, '
                    f'but configured algorithm is {configured_algorithm.upper()}')
        logger.error('Validation stopped. Please update configuration or re-embed with correct algorithm.\n')
        return
    
    # Edge case: video and audio use different algorithms
    if video_algorithm and audio_algorithm and video_algorithm != audio_algorithm:
        logger.error(f'ALGORITHM MISMATCH: Video hash uses {video_algorithm.upper()}, '
                    f'but audio hash uses {audio_algorithm.upper()}')
        logger.error('Validation stopped. Hashes must use the same algorithm.\n')
        return
    
    if check_cancelled():
        return None
    
    # Algorithms match - proceed with validation
    logger.debug('Generating video and audio stream hashes. This may take a moment...')
    hash_result = make_stream_hash(video_path, check_cancelled=check_cancelled, signals=signals)
    if hash_result is None:
        return None
    
    video_hash, audio_hash = hash_result
    logger.debug(f"\n")
    logger.debug('Validating stream fixity\n')
    compare_hashes(existing_video_hash, existing_audio_hash, video_hash, audio_hash)

    if check_cancelled():
        return None


def process_embedded_fixity(video_path, check_cancelled=None, signals=None):
    """
    Handles embedding stream fixity tags in the video file.
    Decides whether to embed for the first time, overwrite, or skip.
    """
    config_mgr = ConfigManager()
    checks_config = config_mgr.get_config('checks', ChecksConfig)

    existing_tags = extract_tags(video_path)

    if existing_tags:
        existing_video_hash, existing_audio_hash = extract_hashes(existing_tags)
    else:
        existing_video_hash = None
        existing_audio_hash = None

    # Decide what to do:
    if existing_video_hash is None or existing_audio_hash is None:
        # No stream hashes yet → always embed them
        logger.info("No existing stream hashes found. Embedding new stream hashes.")
        embed_fixity(video_path, check_cancelled=check_cancelled, signals=signals)
    else:
        logger.critical("Existing stream hashes found!")
        # Now using boolean check
        if checks_config.fixity.overwrite_stream_fixity:
            logger.critical('New stream hashes will be generated and old hashes will be overwritten!\n')
            embed_fixity(video_path, check_cancelled=check_cancelled, signals=signals)
        else:
            logger.error('Not writing new stream hashes to MKV. Overwrite is disabled.\n')