import subprocess
import os
import sys
from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ChecksConfig
from AV_Spex.utils.config_manager import ConfigManager

def get_duration(video_path):
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    duration = result.stdout.decode().strip()
    return duration


def make_access_file(video_path, output_path, check_cancelled=None, signals=None):
    """Create access file using ffmpeg."""

    logger.debug(f'Running ffmpeg on {os.path.basename(video_path)} to create access copy {os.path.basename(output_path)}\n')

    duration_str = get_duration(video_path)

    ffmpeg_command = [
        'ffmpeg',
        '-n', '-vsync', '0',
        '-hide_banner', '-progress', 'pipe:1', '-nostats', '-loglevel', 'error',
        '-i', video_path,
        '-movflags', 'faststart', '-map', '0:v:0', '-map', '0:a?', '-c:v', 'libx264', 
        '-vf', 'yadif=1,format=yuv420p', '-crf', '18', '-preset', 'fast', '-maxrate', '1000k', '-bufsize', '1835k', 
        '-c:a', 'aac', '-strict', '-2', '-b:a', '192k', '-f', 'mp4', output_path
    ]

    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            ff_output = ffmpeg_process.stdout.readline()
            if not ff_output:
                break
            duration_prefix = 'out_time_ms='
            # define prefix of ffmpeg microsecond progress output
            duration = float(duration_str)
            # Convert string integer
            duration_ms = (duration * 1000000)
            # Calculate the total duration in microseconds
            if ff_output.startswith(duration_prefix):
                if check_cancelled and check_cancelled():
                    ffmpeg_process.terminate()  # Fixed: added parentheses
                    ffmpeg_process.wait()  # Wait for process to finish
                    return False  # Return False to indicate cancellation
                current_frame_str = ff_output.split(duration_prefix)[1]
                current_frame_ms = float(current_frame_str)
                percent_complete = (current_frame_ms / duration_ms) * 100
                if signals:
                    # Make doubly sure we're emitting an integer percentage in range 0-100
                    safe_percent = min(100, max(0, int(percent_complete)))
                    signals.access_file_progress.emit(safe_percent)
                else:
                    print(f"\rFFmpeg Access Copy Progress: {percent_complete:.2f}%", end='', flush=True)
        
        # Wait for process to complete
        ffmpeg_process.wait()
        
        ffmpeg_stderr = ffmpeg_process.stderr.read()
        if ffmpeg_stderr:
            logger.error(f"ffmpeg stderr: {ffmpeg_stderr.strip()}")
            
        # Check if process completed successfully
        if ffmpeg_process.returncode == 0:
            return True
        else:
            logger.error(f"ffmpeg exited with code {ffmpeg_process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Error during ffmpeg process: {str(e)}")
        return False
    finally:
        print("\n")


def process_access_file(video_path, source_directory, video_id, check_cancelled=None, signals=None):
    """
    Generate access file if configured and not already existing.
    
    Args:
        video_path (str): Path to the input video file
        source_directory (str): Source directory for the video
        video_id (str): Unique identifier for the video
        check_cancelled: Function to check if operation was cancelled
        signals: Signal object for progress updates
        
    Returns:
        str or None: Path to the created access file, or None
    """
    config_mgr = ConfigManager()
    checks_config = config_mgr.get_config('checks', ChecksConfig)
    
    # Check if access file should be generated
    if checks_config.outputs.access_file != 'yes':
        return None

    access_output_path = os.path.join(source_directory, f'{video_id}_access.mp4')

    try:
        # Check if access file already exists
        if os.path.isfile(access_output_path):
            # Check file size to ensure it's not a partial file from interrupted processing
            file_size = os.path.getsize(access_output_path)
            if file_size > 100000:  # More than 100KB (adjust based on your typical file sizes)
                logger.critical(f"Access file already exists, not running ffmpeg\n")
                if signals:
                    signals.step_completed.emit("Generate Access File")
                return access_output_path
            else:
                # Remove incomplete file
                logger.info(f"Removing incomplete access file from previous run (size: {file_size} bytes)")
                os.remove(access_output_path)

        # Store the access file path in context for cleanup on pause
        if hasattr(check_cancelled, '__self__'):
            processor = check_cancelled.__self__
            if hasattr(processor, '_processing_context') and processor._processing_context:
                processor._processing_context['current_access_file'] = access_output_path

        # Generate access file
        success = make_access_file(video_path, access_output_path, check_cancelled=check_cancelled, signals=signals)
        
        # Clear the current access file from context after completion
        if hasattr(check_cancelled, '__self__'):
            processor = check_cancelled.__self__
            if hasattr(processor, '_processing_context') and processor._processing_context:
                processor._processing_context.pop('current_access_file', None)
        
        if success:
            if signals:
                signals.step_completed.emit("Generate Access File")
            return access_output_path
        else:
            # Clean up incomplete file if creation failed or was cancelled
            if os.path.exists(access_output_path):
                logger.info("Removing incomplete access file after cancellation/failure")
                os.remove(access_output_path)
            return None

    except Exception as e:
        logger.critical(f"Error creating access file: {e}")
        # Clean up on error
        if os.path.exists(access_output_path):
            try:
                os.remove(access_output_path)
            except:
                pass
        return None