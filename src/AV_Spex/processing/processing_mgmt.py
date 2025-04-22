import os
import shutil
import subprocess
import time

from ..processing import run_tools
from ..utils import dir_setup
from ..utils.log_setup import logger
from ..utils.config_setup import ChecksConfig, SpexConfig
from ..utils.config_manager import ConfigManager
from ..utils.generate_report import generate_final_report
from ..checks.fixity_check import check_fixity, output_fixity
from ..checks.mediainfo_check import parse_mediainfo
from ..checks.mediatrace_check import parse_mediatrace, create_metadata_difference_report
from ..checks.exiftool_check import parse_exiftool
from ..checks.ffprobe_check import parse_ffprobe
from ..checks.embed_fixity import validate_embedded_md5, process_embedded_fixity
from ..checks.make_access import process_access_file
from ..checks.qct_parse import run_qctparse
from ..checks.mediaconch_check import find_mediaconch_policy, run_mediaconch_command, parse_mediaconch_output


class ProcessingManager:
    def __init__(self, signals=None, check_cancelled_fn=None):
        self.signals = signals
        self.check_cancelled = check_cancelled_fn or (lambda: False)
        # Force a reload of the config from disk
         # Store config manager as an instance attribute
        self.config_mgr = ConfigManager()
        self.config_mgr.refresh_configs()
        self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
        self.spex_config = self.config_mgr.get_config('spex', SpexConfig)

    def process_fixity(self, source_directory, video_path, video_id):
        """
        Orchestrates the entire fixity process, including embedded and file-level operations.

        Args:
            source_directory (str): Directory containing source files
            video_path (str): Path to the video file
            video_id (str): Unique identifier for the video
        """
        
        if self.check_cancelled():
            return None
        
        # Embed stream fixity if required  
        if self.checks_config.fixity.embed_stream_fixity == 'yes':
            if self.signals:
                self.signals.fixity_progress.emit("Embedding fixity...")
            if self.check_cancelled():
                return False
            process_embedded_fixity(video_path, check_cancelled=self.check_cancelled, signals=self.signals)
            if self.check_cancelled():
                return False
            # Mark checkbox
            if self.signals:
                self.signals.step_completed.emit("Embed Stream Fixity")

        # Validate stream hashes if required
        if self.checks_config.fixity.validate_stream_fixity == 'yes':
            if self.signals:
                self.signals.fixity_progress.emit("Validating embedded fixity...")
            if self.checks_config.fixity.embed_stream_fixity == 'yes':
                logger.critical("Embed stream fixity is turned on, which overrides validate_fixity. Skipping validate_fixity.\n")
            else:
                validate_embedded_md5(video_path, check_cancelled=self.check_cancelled, signals=self.signals)
            # Mark checkbox
            if self.signals:
                self.signals.step_completed.emit("Validate Stream Fixity")

        # Initialize md5_checksum variable
        md5_checksum = None

        # Create checksum for video file and output results
        if self.checks_config.fixity.output_fixity == 'yes':
            if self.signals:
                self.signals.fixity_progress.emit("Outputting fixity...")
            md5_checksum = output_fixity(source_directory, video_path, check_cancelled=self.check_cancelled, signals=self.signals)
            if self.signals:
                self.signals.step_completed.emit("Output Fixity")

        # Verify stored checksum and write results  
        if self.checks_config.fixity.check_fixity == 'yes':
            if self.signals:
                self.signals.fixity_progress.emit("Validating fixity...")
            check_fixity(source_directory, video_id, actual_checksum=md5_checksum, check_cancelled=self.check_cancelled, signals=self.signals)
            if self.signals:
                self.signals.step_completed.emit("Validate Fixity")

        if self.check_cancelled():
            return None


    def validate_video_with_mediaconch(self, video_path, destination_directory, video_id):
        """
        Coordinate the entire MediaConch validation process.
        
        Args:
            video_path (str): Path to the input video file
            destination_directory (str): Directory to store output files
            video_id (str): Unique identifier for the video
            config_path (object): Configuration path object
            
        Returns:
            dict: Validation results from MediaConch policy check
        """
        # Check if MediaConch should be run
        if self.checks_config.tools.mediaconch.run_mediaconch != 'yes':
            logger.info(f"MediaConch validation skipped\n")
            return {}
        
        if self.signals:
            self.signals.mediaconch_progress.emit("Locating MediaConch policy...")
        if self.check_cancelled():
            return None
        
        # Find the policy file
        policy_name = self.checks_config.tools.mediaconch.mediaconch_policy
        policy_path = self.config_mgr.get_policy_path(policy_name)
        if not policy_path:
            return {}

        # Prepare output path
        mediaconch_output_path = os.path.join(destination_directory, f'{video_id}_mediaconch_output.csv')

        if self.signals:
            self.signals.mediaconch_progress.emit("Running MediaConch...")
        if self.check_cancelled():
            return None

        # Run MediaConch command
        if not run_mediaconch_command(
            'mediaconch -p', 
            video_path, 
            '-oc', 
            mediaconch_output_path, 
            policy_path
        ):
            return {}
        
        if self.check_cancelled():
            return None

        # Parse and validate MediaConch output
        validation_results = parse_mediaconch_output(mediaconch_output_path)

        return validation_results
    

    def process_video_metadata(self, video_path, destination_directory, video_id):
        """
        Main function to process video metadata using multiple tools.
        
        Args:
            video_path (str): Path to the input video file
            destination_directory (str): Directory to store output files
            video_id (str): Unique identifier for the video
            
        Returns:
            dict: Dictionary of metadata differences from various tools
        """
        if self.check_cancelled():
            return None
        
        tools = ['exiftool', 'mediainfo', 'mediatrace', 'ffprobe']
        
        # Store differences for each tool
        metadata_differences = {}

        if self.signals:
            self.signals.metadata_progress.emit("Running metadata tools...")
        
        # Process each tool
        for tool in tools:
            if self.check_cancelled():
                return None
                
            # Run tool and get output path
            output_path = run_tools.run_tool_command(tool, video_path, destination_directory, video_id)
            
            # Check metadata and store differences
            differences = check_tool_metadata(tool, output_path)
            if differences:
                metadata_differences[tool] = differences
            
            # Emit step completed signal for this tool
            if self.signals:
                # Capitalize first letter for the step name to match the format in populate_steps_list
                tool_name = tool.capitalize() if tool != 'ffprobe' else 'FFprobe'
                self.signals.step_completed.emit(tool_name)
                
            if self.check_cancelled():
                return None
        
        return metadata_differences
    

    def process_video_outputs(self, video_path, source_directory, destination_directory, video_id, metadata_differences):
        """
        Coordinate the entire output processing workflow.
        
        Args:
            video_path (str): Path to the input video file
            source_directory (str): Source directory for the video
            destination_directory (str): Destination directory for output files
            video_id (str): Unique identifier for the video
            metadata_differences (dict): Differences found in metadata checks
            
        Returns:
            dict: Processing results and file paths
        """

        # Collect processing results
        processing_results = {
            'metadata_diff_report': None,
            'qctools_output': None,
            'access_file': None,
            'html_report': None
        }

        if self.check_cancelled():
            return None
       
        # Create report directory if report is enabled
        report_directory = None
        if self.checks_config.outputs.report == 'yes':
            report_directory = dir_setup.make_report_dir(source_directory, video_id)
            # Process metadata differences report
            processing_results['metadata_diff_report'] = create_metadata_difference_report(
                    metadata_differences, report_directory, video_id
                )
        else:
            processing_results['metadata_diff_report'] =  None
        
        if self.signals:
            self.signals.output_progress.emit("Running QCTools and qct-parse...")
        if self.check_cancelled():
            return None

        # Process QCTools output
        process_qctools_output(
            video_path, source_directory, destination_directory, video_id, report_directory=report_directory,
            check_cancelled=self.check_cancelled, signals=self.signals
        )

        if self.signals:
            self.signals.output_progress.emit("Creating access file...")
        if self.check_cancelled():
            return None

        # Generate access file
        processing_results['access_file'] = process_access_file(
            video_path, source_directory, video_id, 
            check_cancelled=self.check_cancelled,
            signals=self.signals
        )

        if self.signals:
            self.signals.output_progress.emit("Preparing report...")
        if self.check_cancelled():
            return None

        # Generate final HTML report
        processing_results['html_report'] = generate_final_report(
            video_id, source_directory, report_directory, destination_directory,
            check_cancelled=self.check_cancelled, signals=self.signals
        )

        return processing_results


def process_qctools_output(video_path, source_directory, destination_directory, video_id, report_directory=None, check_cancelled=None, signals=None):
    """
    Process QCTools output, including running QCTools and optional parsing.
    
    Args:
        video_path (str): Path to the input video file
        destination_directory (str): Directory to store output files
        video_id (str): Unique identifier for the video
        report_directory (str, optional): Directory to save reports
        
    Returns:
        dict: Processing results and paths
    """
    config_mgr = ConfigManager()
    checks_config = config_mgr.get_config('checks', ChecksConfig)
    
    results = {
        'qctools_output_path': None,
        'qctools_check_output': None
    }

    # Prepare QCTools output path
    qctools_ext = checks_config.outputs.qctools_ext
    qctools_output_path = os.path.join(destination_directory, f'{video_id}.{qctools_ext}')

    if check_cancelled():
            return None

    # Run QCTools command
    if checks_config.tools.qctools.run_tool == 'yes':
        run_qctools_command('qcli -i', video_path, '-o', qctools_output_path, check_cancelled=check_cancelled)
        logger.debug('')  # Add new line for cleaner terminal output
        results['qctools_output_path'] = qctools_output_path
        if signals:
            signals.step_completed.emit("QCTools")

    # Check QCTools output if configured
    if checks_config.tools.qct_parse.run_tool == 'yes':
        # Ensure report directory exists
        if not report_directory:
            report_directory = dir_setup.make_report_dir(source_directory, video_id)

        # Verify QCTools output file exists
        if not os.path.isfile(qctools_output_path):
            logger.critical(f"Unable to check qctools report. No file found at: {qctools_output_path}\n")
            return results

        # Run QCTools parsing
        run_qctparse(video_path, qctools_output_path, report_directory, check_cancelled=check_cancelled)
        if signals:
            signals.step_completed.emit("QCT Parse")

    return results

def run_qctools_command(command, input_path, output_type, output_path, check_cancelled=None):
    
    if check_cancelled():
        return None
    
    env = os.environ.copy()
    env['PATH'] = '/usr/local/bin:' + env.get('PATH', '')

    full_command = f"{command} \"{input_path}\" {output_type} {output_path}"
    logger.debug(f'Running command: {full_command}\n')
    
    process = subprocess.Popen(full_command, shell=True, env=env)
    
    while process.poll() is None:  # While process is running
        if check_cancelled():
            process.terminate()  # Send SIGTERM
            try:
                process.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
            except subprocess.TimeoutExpired:
                process.kill()  # Force kill if process doesn't terminate
            return None
        time.sleep(1)  # Check cancel status every 0.5 seconds
    
    #return process.returncode


def check_tool_metadata(tool_name, output_path):
    """
    Check metadata for a specific tool if configured.
    
    Args:
        tool_name (str): Name of the tool
        output_path (str): Path to the tool's output file
        
    Returns:
        dict or None: Differences found by parsing the tool's output, or None
    """
    config_mgr = ConfigManager()
    checks_config = config_mgr.get_config('checks', ChecksConfig)

    # Mapping of tool names to their parsing functions
    parse_functions = {
        'exiftool': parse_exiftool,
        'mediainfo': parse_mediainfo,
        'mediatrace': parse_mediatrace,
        'ffprobe': parse_ffprobe
    }

    # Check if tool metadata checking is enabled
    tool = getattr(checks_config.tools, tool_name)
    if output_path and tool.check_tool == 'yes':
        parse_function = parse_functions.get(tool_name)
        if parse_function:
            return parse_function(output_path)
    
    return None


def setup_mediaconch_policy(user_policy_path: str = None) -> str:
    """
    Set up MediaConch policy file, either using user-provided policy or default.
    
    Args:
        user_policy_path (str, optional): Path to user-provided policy file
        
    Returns:
        str: Name of the policy file that will be used
    """
    config_mgr = ConfigManager()
    
    if not user_policy_path:
        # Return current policy file name from config
        current_config = config_mgr.get_config('checks', ChecksConfig)
        return current_config.tools.mediaconch.mediaconch_policy
        
    try:
        # Verify user policy file exists
        if not os.path.exists(user_policy_path):
            logger.critical(f"User provided policy file not found: {user_policy_path}")
            return None
            
        # Get policy file name
        policy_filename = os.path.basename(user_policy_path)
        
        # Copy policy file to user policies directory
        user_policy_dest = os.path.join(config_mgr._user_policies_dir, policy_filename)
        
        # Copy policy file, overwriting if file exists
        shutil.copy2(user_policy_path, user_policy_dest, follow_symlinks=False)
        logger.info(f"Copied user policy file to user policies directory: {policy_filename}")
        
        # Get current config to preserve run_mediaconch value
        current_config = config_mgr.get_config('checks', ChecksConfig)
        run_mediaconch = current_config.tools.mediaconch.run_mediaconch
        
        # Update config to use new policy file while preserving run_mediaconch
        config_mgr.update_config('checks', {
            'tools': {
                'mediaconch': {
                    'mediaconch_policy': policy_filename,
                    'run_mediaconch': run_mediaconch
                }
            }
        })
        logger.info(f"Updated config to use new policy file: {policy_filename}")
        
        return policy_filename
        
    except Exception as e:
        logger.critical(f"Error setting up MediaConch policy: {str(e)}")
        return None