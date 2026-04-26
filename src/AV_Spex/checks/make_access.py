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


def make_access_file(video_path, output_path, check_cancelled=None, signals=None, start_time=None, crop_area=None):
    """Create access file using ffmpeg.

    If start_time (seconds) is provided and > 0, the input is seeked past that
    point so head content (e.g. color bars detected by qct-parse) is excluded
    from the access copy.

    If crop_area (x, y, w, h) is provided, the active picture area is cropped
    out of the access copy. Width/height are forced even for yuv420p.
    """

    logger.debug(f'Running ffmpeg on {os.path.basename(video_path)} to create access copy {os.path.basename(output_path)}\n')

    duration_str = get_duration(video_path)

    ffmpeg_command = ['ffmpeg', '-n', '-vsync', '0',
        '-hide_banner', '-progress', 'pipe:1', '-nostats', '-loglevel', 'error']

    if start_time and start_time > 0:
        logger.info(f'Trimming first {start_time:.2f}s from access copy (color bars detected by qct-parse)\n')
        ffmpeg_command.extend(['-ss', f'{start_time:.3f}'])

    vf_filters = []
    if crop_area:
        x, y, w, h = (int(v) for v in crop_area)
        # yuv420p requires even dimensions
        if w % 2:
            w -= 1
        if h % 2:
            h -= 1
        vf_filters.append(f'crop={w}:{h}:{x}:{y}')
        logger.info(f'Cropping access copy to active area {w}x{h} at offset ({x},{y})\n')
    vf_filters.extend(['yadif=1', 'format=yuv420p'])

    ffmpeg_command.extend([
        '-i', video_path,
        '-movflags', 'faststart', '-map', '0:v:0', '-map', '0:a?', '-c:v', 'libx264',
        '-vf', ','.join(vf_filters), '-crf', '18', '-preset', 'fast', '-maxrate', '1000k', '-bufsize', '1835k',
        '-c:a', 'aac', '-strict', '-2', '-b:a', '192k', '-f', 'mp4', output_path
    ])

    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            ff_output = ffmpeg_process.stdout.readline()
            if not ff_output:
                break
            duration_prefix = 'out_time_ms='
            # define prefix of ffmpeg microsecond progress output
            duration = float(duration_str)
            if start_time and start_time > 0:
                duration = max(0.001, duration - start_time)
            # Convert string integer
            duration_ms = (duration * 1000000)
            # Calculate the total duration in microseconds
            if ff_output.startswith(duration_prefix):
                if check_cancelled():
                    ffmpeg_process.terminate
                    return
                current_frame_str = ff_output.split(duration_prefix)[1]
                current_frame_ms = float(current_frame_str)
                percent_complete = (current_frame_ms / duration_ms) * 100
                if signals:
                    # Make doubly sure we're emitting an integer percentage in range 0-100
                    safe_percent = min(100, max(0, int(percent_complete)))
                    signals.access_file_progress.emit(safe_percent)
                else:
                    print(f"\rFFmpeg Access Copy Progress: {percent_complete:.2f}%", end='', flush=True)
        ffmpeg_stderr = ffmpeg_process.stderr.read()
        if ffmpeg_stderr:
            logger.error(f"ffmpeg stderr: {ffmpeg_stderr.strip()}")
    except Exception as e:
        logger.error(f"Error during ffmpeg process: {str(e)}")
    print("\n")


def process_access_file(video_path, source_directory, video_id, check_cancelled=None, signals=None, color_bars_end_time=None, crop_area=None):
    """
    Generate access file if configured and not already existing.

    Args:
        video_path (str): Path to the input video file
        source_directory (str): Source directory for the video
        video_id (str): Unique identifier for the video
        check_cancelled (callable, optional): Function to check if operation was cancelled
        signals (object, optional): Signal object for progress updates
        color_bars_end_time (float, optional): End time of color bars in seconds, as
            detected by qct-parse. When provided, the access copy is trimmed to start
            after the bars.
        crop_area (tuple, optional): (x, y, w, h) active picture area from sophisticated
            border detection (BRNG-refined when available). When provided, borders are
            cropped off the access copy.

    Returns:
        str or None: Path to the created access file, or None
    """
    config_mgr = ConfigManager()
    checks_config = config_mgr.get_config('checks', ChecksConfig)

    # Check if access file should be generated
    if not checks_config.outputs.access_file:
        return None

    access_output_path = os.path.join(source_directory, f'{video_id}_access.mp4')

    try:
        # Check if access file already exists
        for filename in os.listdir(source_directory):
            if filename.lower().endswith('mp4'):
                logger.critical(f"Access file already exists, not running ffmpeg\n")
                if signals:
                    signals.step_completed.emit("Generate Access File")
                return None
        if os.path.isfile(access_output_path):
            logger.critical(f"Access file already exists, not running ffmpeg\n")
            if signals:
                signals.step_completed.emit("Generate Access File")
            return None

        # Generate access file
        make_access_file(
            video_path, access_output_path,
            check_cancelled=check_cancelled, signals=signals,
            start_time=color_bars_end_time,
            crop_area=crop_area
        )
        if signals:
            signals.step_completed.emit("Generate Access File")
        return access_output_path

    except Exception as e:
        logger.critical(f"Error creating access file: {e}")
        return None