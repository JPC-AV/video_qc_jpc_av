import os
import time
import subprocess
from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ChecksConfig, is_mkv_extension
from AV_Spex.utils.config_manager import ConfigManager

config_mgr = ConfigManager()

# mkvalidator emits no parseable progress (just a run of dots), so its progress bar
# is a time estimate. Across the JPC_AV sample files (3–37 GiB, n=8) runtime grows
# *super-linearly* with file size — the per-GiB cost rises from ~2.6 s/GiB at 3 GiB to
# ~4.8 s/GiB at 37 GiB — so a flat per-GiB rate underestimates large files and pins
# the bar at 99%. A power-law fit tracks the whole range:
#     estimated_seconds ≈ MKVALIDATOR_EST_COEFF * size_in_GiB ** MKVALIDATOR_EST_EXP
# Coefficients are lightly biased to overestimate (worst observed underestimate ≈ 0%),
# so the bar errs toward finishing early — it snaps to 100% rather than sitting at 99%.
# Recalibrate from more sample _avspex_processing.log timings if the bias drifts.
MKVALIDATOR_EST_COEFF = 2.3
MKVALIDATOR_EST_EXP = 1.2
_GIB = 1024 ** 3

def run_command(command, input_path, output_type, output_path):
    '''
    Run a shell command with 4 variables: command name, path to the input file, output type (often '>'), path to the output file
    '''

    # Get the current PATH environment variable.
    # /opt/homebrew/bin first so Apple Silicon native binaries win over any
    # leftover x86_64 ones in /usr/local/bin (which would run under Rosetta).
    env = os.environ.copy()
    env['PATH'] = '/opt/homebrew/bin:/usr/local/bin:' + env.get('PATH', '')

    full_command = f"{command} \"{input_path}\" {output_type} \"{output_path}\""

    logger.debug(f'Running command: {full_command}\n')
    subprocess.run(full_command, shell=True, env=env)


def run_tool_command(tool_name, video_path, destination_directory, video_id, signals=None):
    """
    Run a specific metadata extraction tool and generate its output file.

    Args:
        tool_name (str): Name of the tool to run (e.g., 'exiftool', 'mediainfo')
        video_path (str): Path to the input video file
        destination_directory (str): Directory to store output files
        video_id (str): Unique identifier for the video
        signals: Optional ProcessingSignals; used by mkvalidator to emit progress.

    Returns:
        str or None: Path to the output file, or None if tool is not run
    """
    # get config
    checks_config = config_mgr.get_config('checks', ChecksConfig)

    # Define tool-specific commands
    tool_commands = {
        'exiftool': 'exiftool -j',
        'mediainfo': 'mediainfo -f --Output=JSON',
        'mediatrace': 'mediainfo --Details=1 --Output=XML',
        'ffprobe': 'ffprobe -v error -hide_banner -show_format -show_streams -print_format json'
    }

    # Construct output path
    output_path = os.path.join(destination_directory, f'{video_id}_{tool_name}_output.{_get_file_extension(tool_name)}')

    # mkvalidator is a Matroska conformance validator (not a metadata extractor), so
    # it is not in tool_commands. It writes its WRN/ERR diagnostics to stderr, so the
    # generic '>' redirect used by run_command() would drop them. Handle it separately
    # (before the tool_commands lookup) to capture stdout+stderr, and skip non-MKV
    # input (it only applies to Matroska).
    if tool_name == 'mkvalidator':
        tool = getattr(checks_config.tools, tool_name)
        if tool.run_tool:
            ext = getattr(checks_config, 'video_file_extension', 'mkv')
            if not is_mkv_extension(ext):
                logger.warning(
                    f"Input extension '{ext}' is not MKV; mkvalidator only applies to "
                    "Matroska. Skipping mkvalidator.\n"
                )
                return None
            run_mkvalidator_command(video_path, output_path, signals=signals)
        return output_path

    # Check if the tool is configured
    command = tool_commands.get(tool_name)
    if not command:
        logger.error(f"tool command is not configured correctly: {tool_name}")
        return None

    if tool_name != "mediaconch":
        # Check if tool should be run based on configuration
        tool = getattr(checks_config.tools, tool_name)
        if tool.run_tool:
            if tool_name == 'mediatrace':
                # mediatrace reads Matroska SimpleTags (custom MKV Tag metadata);
                # it only applies to MKV input. If it slipped through on a non-MKV
                # file, skip it gracefully rather than producing a meaningless XML.
                ext = getattr(checks_config, 'video_file_extension', 'mkv')
                if not is_mkv_extension(ext):
                    logger.warning(
                        f"Input extension '{ext}' is not MKV; the mediatrace custom-tag "
                        "check only applies to Matroska. Skipping mediatrace.\n"
                    )
                    return None
                logger.debug(f"Creating {tool_name.capitalize()} XML file to check custom MKV Tag metadata fields:")
            run_command(command, video_path, '>', output_path)

    return output_path


def _estimate_mkvalidator_seconds(video_path):
    '''
    Estimate mkvalidator runtime (seconds) from the input file size.

    Uses the super-linear power-law model (see the MKVALIDATOR_EST_* constants):
    estimated = COEFF * size_in_GiB ** EXP. Returns a floor of 1.0 s so the progress
    math never divides by ~0 on a tiny or unstattable file.
    '''
    try:
        size_gib = os.path.getsize(video_path) / _GIB
    except OSError:
        size_gib = 0
    return max(MKVALIDATOR_EST_COEFF * size_gib ** MKVALIDATOR_EST_EXP, 1.0)


def run_mkvalidator_command(video_path, output_path, signals=None):
    '''
    Run mkvalidator on an MKV file, capturing both stdout and stderr to the
    sidecar (mkvalidator emits its WRN/ERR diagnostics on stderr).

    mkvalidator prints only a run of dots (no parseable percentage), so progress is
    a time estimate: while the process runs we tick toward 99% based on elapsed time
    vs. a size-derived estimate, then snap to 100% on exit.
    '''
    env = os.environ.copy()
    env['PATH'] = '/opt/homebrew/bin:/usr/local/bin:' + env.get('PATH', '')

    full_command = f'mkvalidator "{video_path}"'
    logger.debug(f'Running command: {full_command} > "{output_path}" 2>&1\n')

    estimated_total = _estimate_mkvalidator_seconds(video_path)
    if signals:
        signals.mkvalidator_progress.emit(0)

    try:
        with open(output_path, 'w') as out:
            process = subprocess.Popen(full_command, shell=True, env=env,
                                       stdout=out, stderr=subprocess.STDOUT)
            start = time.monotonic()
            while process.poll() is None:
                if signals:
                    pct = int((time.monotonic() - start) / estimated_total * 100)
                    signals.mkvalidator_progress.emit(min(99, max(0, pct)))
                time.sleep(0.5)
        if signals:
            signals.mkvalidator_progress.emit(100)
    except (OSError, subprocess.SubprocessError) as e:
        logger.error(f"Failed to run mkvalidator: {e}")


def _get_file_extension(tool_name):
    """
    Get the appropriate file extension for each tool's output.
    
    Args:
        tool_name (str): Name of the tool
        
    Returns:
        str: File extension for the tool's output
    """
    extension_map = {
        'exiftool': 'json',
        'mediainfo': 'json',
        'mediatrace': 'xml',
        'ffprobe': 'txt',
        'mkvalidator': 'txt'
    }
    return extension_map.get(tool_name, 'txt')

