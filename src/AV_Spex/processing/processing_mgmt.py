import os
import shutil
import subprocess
import time
import re
from pathlib import Path

from AV_Spex.processing import run_tools
from AV_Spex.utils import dir_setup
from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ChecksConfig, SpexConfig
from AV_Spex.utils.config_manager import ConfigManager
from AV_Spex.utils.generate_report import generate_final_report
from AV_Spex.checks.fixity_check import check_fixity, output_fixity
from AV_Spex.checks.mediainfo_check import parse_mediainfo
from AV_Spex.checks.mediatrace_check import parse_mediatrace, create_metadata_difference_report
from AV_Spex.checks.exiftool_check import parse_exiftool
from AV_Spex.checks.ffprobe_check import parse_ffprobe
from AV_Spex.checks.embed_fixity import validate_embedded_md5, process_embedded_fixity
from AV_Spex.checks.make_access import process_access_file
from AV_Spex.checks.qct_parse import run_qctparse
from AV_Spex.checks.mediaconch_check import find_mediaconch_policy, run_mediaconch_command, parse_mediaconch_output


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
        if self.checks_config.fixity.embed_stream_fixity:
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
        if self.checks_config.fixity.validate_stream_fixity:
            if self.signals:
                self.signals.fixity_progress.emit("Validating embedded fixity...")
            if self.checks_config.fixity.embed_stream_fixity:
                logger.critical("Embed stream fixity is turned on, which overrides validate_fixity. Skipping validate_fixity.\n")
            else:
                validate_embedded_md5(video_path, check_cancelled=self.check_cancelled, signals=self.signals)
            # Mark checkbox
            if self.signals:
                self.signals.step_completed.emit("Validate Stream Fixity")

        # Initialize file_checksum variable
        file_checksum = None

        # Create checksum for video file and output results
        if self.checks_config.fixity.output_fixity:
            if self.signals:
                self.signals.fixity_progress.emit("Outputting fixity...")
            file_checksum = output_fixity(source_directory, video_path, check_cancelled=self.check_cancelled, signals=self.signals)
            if self.signals:
                self.signals.step_completed.emit("Output Fixity")

        # Verify stored checksum and write results  
        if self.checks_config.fixity.check_fixity:
            if self.signals:
                self.signals.fixity_progress.emit("Validating fixity...")
            check_fixity(source_directory, video_id, actual_checksum=file_checksum, check_cancelled=self.check_cancelled, signals=self.signals)
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
        if not self.checks_config.tools.mediaconch.run_mediaconch:
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
        if self.checks_config.outputs.report:
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
            video_path=video_path,
            check_cancelled=self.check_cancelled, signals=self.signals
        )

        return processing_results
    

def find_qctools_report(source_directory, video_id):
    """
    Search for existing qctools files in both _qc_metadata and _vrecord_metadata folders.
    
    Args:
        source_directory (str): Path to the source directory containing metadata folders
        video_id (str): Video identifier (e.g., "JPC_AV_01581")
        
    Returns:
        str or None: Path to the qctools report if found, None otherwise
    """
    source_path = Path(source_directory)
    
    # Define the folders to search in
    search_folders = [
        source_path / f"{video_id}_qc_metadata",
        source_path / f"{video_id}_vrecord_metadata"
    ]
    
    # Search patterns for qctools files
    qctools_patterns = [
        "*.qctools.xml.gz",
        "*.qctools.mkv"
    ]
    
    # Search in each folder
    for folder in search_folders:
        if folder.exists() and folder.is_dir():
            for pattern in qctools_patterns:
                matches = list(folder.glob(pattern))
                if matches:
                    return str(matches[0])  # Return the first match
    
    return None


def process_qctools_output(video_path, source_directory, destination_directory, video_id, report_directory=None, check_cancelled=None, signals=None):
    """
    Process QCTools output, including running QCTools and optional parsing.
    Now searches for existing QCTools reports in both _qc_metadata and _vrecord_metadata folders.
    
    Args:
        video_path (str): Path to the input video file
        source_directory (str): Source directory for the video
        destination_directory (str): Directory to store output files
        video_id (str): Unique identifier for the video
        report_directory (str, optional): Directory to save reports
        check_cancelled (callable, optional): Function to check if operation was cancelled
        signals (object, optional): Signal object for progress updates
        
    Returns:
        dict: Processing results and paths
    """
    config_mgr = ConfigManager()
    checks_config = config_mgr.get_config('checks', ChecksConfig)
    
    results = {
        'qctools_output_path': None,
        'qctools_check_output': None
    }

    if check_cancelled and check_cancelled():
        return None

    # Get configuration settings
    qct_run_tool = getattr(checks_config.tools.qctools, 'run_tool')
    qct_parse_run_tool = getattr(checks_config.tools.qct_parse, 'run_tool')
    
    # Always search for existing QCTools reports first
    existing_qctools_path = find_qctools_report(source_directory, video_id)
    
    if existing_qctools_path:
        logger.info(f"Found existing QCTools report: {existing_qctools_path}\n")
        results['qctools_output_path'] = existing_qctools_path
    
    # Handle QCTools generation (only if configured and no existing report)
    if qct_run_tool:
        if existing_qctools_path:
            # Mark step as completed since we found an existing report
            if signals:
                signals.step_completed.emit("QCTools")
        else:
            # No existing report found, create a new one in the destination directory
            qctools_ext = checks_config.outputs.qctools_ext
            qctools_output_path = os.path.join(destination_directory, f'{video_id}.{qctools_ext}')
            
            # Check if we already created one in the destination directory
            if os.path.exists(qctools_output_path):
                logger.warning("QCTools report already exists in destination directory, not overwriting...")
                results['qctools_output_path'] = qctools_output_path
            else:
                # Create new QCTools report
                logger.info(f"No existing QCTools report found. Creating new report: {qctools_output_path}")
                run_qctools_command('qcli -i', video_path, '-o', qctools_output_path, check_cancelled=check_cancelled, signals=signals)
                logger.debug('')  # Add new line for cleaner terminal output
                results['qctools_output_path'] = qctools_output_path
            
            if signals:
                signals.step_completed.emit("QCTools")

    # Handle QCTools parsing (independent of whether QCTools was run)
    if qct_parse_run_tool:
        # Ensure we have a QCTools report to parse
        if not results['qctools_output_path'] or not os.path.isfile(results['qctools_output_path']):
            logger.critical(f"Unable to check qctools report. No QCTools report file found in input directory.")
        else:
            # Ensure report directory exists
            if not report_directory:
                report_directory = dir_setup.make_report_dir(source_directory, video_id)

            # Run QCTools parsing
            logger.info(f"Running qct-parse on: {results['qctools_output_path']}")
            run_qctparse(video_path, results['qctools_output_path'], report_directory, check_cancelled=check_cancelled)
            if signals:
                signals.step_completed.emit("QCT Parse")

    return results

def run_qctools_command(command, input_path, output_type, output_path, check_cancelled=None, signals=None):
    
    if check_cancelled():
        return None
    
    env = os.environ.copy()
    env['PATH'] = '/usr/local/bin:' + env.get('PATH', '')
    full_command = f"{command} \"{input_path}\" {output_type} \"{output_path}\""
    logger.debug(f'Running command: {full_command}\n')
    
    # Use subprocess.Popen with stdout and stderr capture
    process = subprocess.Popen(
        full_command, 
        shell=True, 
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout to catch all output
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    try:
        while True:
            if check_cancelled():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return None
            
            # Read output line by line
            output = process.stdout.readline()
            
            if output == '' and process.poll() is not None:
                # Process has finished and no more output
                break
                
            if output:
                # Log the output for debugging
                #logger.debug(f"QCTools output: {output.strip()}")
                
                # Extract percentage from output
                # Common patterns: "50%", "Progress: 50%", "50.5%", etc.
                percentage = extract_percentage(output.strip(), signals=signals)
                
                if percentage is not None and signals:
                    # Emit the progress signal
                    safe_percent = min(100, max(0, int(percentage)))
                    #logger.debug(f"About to emit QCTools progress: {safe_percent}%")  # Add this debug line
                    signals.qctools_progress.emit(safe_percent)
    
    except Exception as e:
        logger.error(f"Error reading QCTools output: {str(e)}")
    
    # Wait for process to complete and get return code
    return_code = process.wait()
    
    # Emit 100% completion if signals available
    if signals:
        signals.qctools_progress.emit(100)
    
    return return_code

def extract_percentage(output_line, signals=None):
    """
    Extract percentage value from QCTools output line.
    Handles QCTools specific format: "dots + spaces + X of 100 %"
    """
    # QCTools specific pattern: any number of dots, then spaces, then "X of 100 %"
    pattern = r'\.+\s+(\d+)\s+of\s+100\s+%'

    match = re.search(pattern, output_line)
    if match:
        try:
            percentage = int(match.group(1))
            if signals:
                # logger.debug(f"QCTools emitting progress: {percentage}%")
                return percentage
            elif signals is None:
                print(f"\rQCTools progress: {percentage}%", end='', flush=True)
                return percentage
        except (ValueError, IndexError):
            pass
    else:
        # Try without dots (for early progress like 1%)
        pattern2 = r'(\d+)\s+of\s+100\s+%'
        match2 = re.search(pattern2, output_line)
        if match2:
            try:
                percentage = int(match2.group(1))
                if signals:
                    # logger.debug(f"QCTools emitting progress: {percentage}%")
                    return percentage
                elif signals is None:
                    print(f"\rQCTools progress: {percentage}%", end='', flush=True)
                    return percentage
            except (ValueError, IndexError):
                pass


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
    if output_path and tool.check_tool:
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