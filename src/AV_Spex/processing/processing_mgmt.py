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
from AV_Spex.checks.border_detector import detect_video_borders, detect_simple_borders
from AV_Spex.checks.ffmpeg_signalstats_analyzer import analyze_video_signalstats
from AV_Spex.checks.active_area_brng_analyzer import analyze_active_area_brng


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
        
        frame_analysis_results = None
        frame_config = getattr(self.checks_config.outputs, 'frame_analysis', None)
        
        if frame_config and frame_config.enabled == 'yes':
            if self.signals:
                self.signals.output_progress.emit("Performing frame analysis...")
            
            frame_analysis_results = self.process_frame_analysis(
                video_path, source_directory, destination_directory, video_id
            )
            
            processing_results['frame_analysis'] = frame_analysis_results

        # Generate final HTML report
        processing_results['html_report'] = generate_final_report(
            video_id, source_directory, report_directory, destination_directory,
            video_path=video_path,
            check_cancelled=self.check_cancelled, 
            signals=self.signals
        )
        
        return processing_results
    
    def process_frame_analysis(self, video_path, source_directory, destination_directory, video_id):
        """
        Process comprehensive frame analysis including border detection,
        BRNG violations, and optionally signalstats (only with sophisticated borders).
        
        Args:
            video_path (str): Path to the input video file
            source_directory (str): Source directory for the video
            destination_directory (str): Destination directory for output files
            video_id (str): Unique identifier for the video
            
        Returns:
            dict: Analysis results from all three components
        """
        
        if self.check_cancelled():
            return None
        
        analysis_results = {
            'border_results': None,
            'brng_results': None,
            'signalstats_results': None,
            'border_retry_performed': False,
            'border_detection_method': None,
            'signalstats_skipped_reason': None,
            'color_bars_detected': False,
            'color_bars_end_time': None
        }
        
        # Access the frame analysis config
        frame_config = self.checks_config.outputs.frame_analysis
        
        if frame_config.enabled != 'yes':
            logger.info("Frame analysis not enabled in config")
            return analysis_results
        
        # Use the config values
        use_sophisticated = frame_config.border_detection_mode == "sophisticated"
        border_pixels = frame_config.simple_border_pixels
        skip_color_bars = frame_config.brng_skip_color_bars == "yes"
        
        # Check for color bars if enabled
        color_bars_end_seconds = None
        if skip_color_bars:
            report_directory = Path(source_directory) / f"{video_id}_report_csvs"
            if report_directory.exists():
                colorbars_csv = report_directory / "qct-parse_colorbars_durations.csv"
                if colorbars_csv.exists():
                    start_seconds, end_seconds = parse_colorbars_duration_csv(str(colorbars_csv))
                    if end_seconds:
                        color_bars_end_seconds = end_seconds
                        analysis_results['color_bars_detected'] = True
                        analysis_results['color_bars_end_time'] = end_seconds
                        logger.info(f"Color bars detected by qct-parse, ending at {end_seconds:.1f}s")
        
        # Step 1: Border Detection
        if use_sophisticated:
            if self.signals:
                self.signals.output_progress.emit("Detecting video borders (sophisticated analysis)...")
            
            logger.info("Using sophisticated border detection with frame quality analysis")
            analysis_results['border_detection_method'] = 'sophisticated'
            
            border_results = detect_video_borders(
                video_path,
                destination_directory,
                target_viz_time=frame_config.sophisticated_viz_time,
                search_window=frame_config.sophisticated_search_window,
                threshold=frame_config.sophisticated_threshold,
                edge_sample_width=frame_config.sophisticated_edge_sample_width,
                sample_frames=frame_config.sophisticated_sample_frames,
                padding=frame_config.sophisticated_padding
            )
        else:
            if self.signals:
                self.signals.output_progress.emit(f"Applying simple {border_pixels}px border detection...")
            
            logger.info(f"Using simple border detection with {border_pixels}px borders")
            analysis_results['border_detection_method'] = 'simple'
            
            border_results = detect_simple_borders(
                video_path,
                border_size=border_pixels,
                output_dir=destination_directory
            )
        
        if self.check_cancelled():
            return None
        
        analysis_results['border_results'] = border_results
        
        # Get the path to the border data file
        border_data_path = Path(destination_directory) / f"{video_id}_border_data.json"
        
        # Step 2: BRNG Analysis (runs for both simple and sophisticated)
        if self.signals:
            self.signals.output_progress.emit("Analyzing BRNG violations...")
        
        brng_results = None
        if border_results and border_results.get('active_area'):
            brng_results = analyze_active_area_brng(
                video_path=video_path,
                border_data_path=border_data_path,
                output_dir=destination_directory,
                duration_limit=getattr(frame_config, 'brng_duration_limit', 300)
            )
        else:
            # Analyze full frame if no borders detected
            brng_results = analyze_active_area_brng(
                video_path=video_path,
                border_data_path=None,
                output_dir=destination_directory,
                duration_limit=getattr(frame_config, 'brng_duration_limit', 300)
            )
        
        if self.check_cancelled():
            return None
        
        analysis_results['brng_results'] = brng_results
        
        # Step 3: Check if border adjustment is needed (only for sophisticated mode)
        if use_sophisticated and brng_results and frame_config.auto_retry_borders == 'yes':
            aggregate = brng_results.get('aggregate_patterns', {})
            requires_adjustment = aggregate.get('requires_border_adjustment', False)
            
            if requires_adjustment:
                if self.signals:
                    self.signals.output_progress.emit("Re-detecting borders with adjusted parameters...")
                
                logger.warning("BRNG analysis detected boundary artifacts - adjusting border detection")
                
                # Re-run sophisticated border detection with adjusted parameters
                adjusted_threshold = getattr(frame_config, 'border_threshold', 10) + 5
                adjusted_edge_width = getattr(frame_config, 'border_edge_sample_width', 100) + 50
                adjusted_samples = min(getattr(frame_config, 'border_sample_frames', 30) + 20, 100)
                adjusted_padding = getattr(frame_config, 'border_padding', 5) + 5
                
                border_results = detect_video_borders(
                    video_path,
                    destination_directory,
                    target_viz_time=getattr(frame_config, 'border_viz_time', 150),
                    search_window=getattr(frame_config, 'border_search_window', 120),
                    threshold=adjusted_threshold,
                    edge_sample_width=adjusted_edge_width,
                    sample_frames=adjusted_samples,
                    padding=adjusted_padding
                )
                
                analysis_results['border_retry_performed'] = True
                analysis_results['border_results'] = border_results
                
                # Re-run BRNG analysis with new borders
                if self.signals:
                    self.signals.output_progress.emit("Re-analyzing BRNG with adjusted borders...")
                
                if border_results and border_results.get('active_area'):
                    brng_results = analyze_active_area_brng(
                        video_path=video_path,
                        border_data_path=border_data_path,
                        output_dir=destination_directory,
                        duration_limit=getattr(frame_config, 'brng_duration_limit', 300)
                    )
                    analysis_results['brng_results'] = brng_results
        
        if self.check_cancelled():
            return None
        
        # Step 4: FFprobe Signalstats Analysis (ONLY if sophisticated border detection was used)
        if use_sophisticated:
            if self.signals:
                self.signals.output_progress.emit("Running FFprobe signalstats analysis...")
            
            logger.info("Running signalstats analysis (sophisticated borders detected)")
            
            signalstats_results = analyze_video_signalstats(
                video_path=video_path,
                border_data_path=border_data_path if border_results else None,
                output_dir=destination_directory,
                start_time=getattr(frame_config, 'signalstats_start_time', 120),
                duration=getattr(frame_config, 'signalstats_duration', 60)
            )
            
            analysis_results['signalstats_results'] = signalstats_results
        else:
            # Skip signalstats for simple border detection
            logger.info("Skipping signalstats analysis (requires sophisticated border detection)")
            analysis_results['signalstats_skipped_reason'] = 'simple_borders'
            
            if self.signals:
                self.signals.output_progress.emit("Skipping signalstats (simple borders mode)...")
        
        # Log summary
        self._log_broadcast_analysis_summary(analysis_results, video_id)
        
        if self.signals:
            self.signals.step_completed.emit("Broadcast Analysis")
        
        return analysis_results
    
    def _log_broadcast_analysis_summary(self, analysis_results, video_id):
        """Log a summary of the broadcast analysis results"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FRAME ANALYSIS SUMMARY - {video_id}")
        logger.info(f"{'='*60}")
        
        # Border detection summary
        detection_method = analysis_results.get('border_detection_method', 'unknown')
        logger.info(f"Border detection method: {detection_method}")
        
        if analysis_results['border_results'] and analysis_results['border_results'].get('active_area'):
            x, y, w, h = analysis_results['border_results']['active_area']
            logger.info(f"Active area: {w}x{h} at ({x},{y})")
            
            if detection_method == 'simple':
                border_size = analysis_results['border_results'].get('border_size_used', 25)
                logger.info(f"  Using fixed {border_size}px borders")
            elif analysis_results['border_retry_performed']:
                logger.info("  ✓ Borders adjusted after BRNG analysis")
        else:
            logger.info("No borders detected - full frame active")
        
        # BRNG analysis summary
        if analysis_results['brng_results']:
            report = analysis_results['brng_results'].get('actionable_report', {})
            logger.info(f"BRNG Assessment: {report.get('overall_assessment', 'Complete')}")
            logger.info(f"Priority: {report.get('action_priority', 'none').upper()}")
        
        # Signalstats summary
        if analysis_results['signalstats_results']:
            logger.info(f"Signalstats: {analysis_results['signalstats_results'].get('diagnosis', 'Complete')}")
        elif analysis_results.get('signalstats_skipped_reason') == 'simple_borders':
            logger.info("Signalstats: Skipped (requires sophisticated border detection)")
        
        logger.info(f"{'='*60}\n")
    

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
        'qctools_check_output': None,
        'color_bars_end_time': None
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
    if qct_run_tool == 'yes':
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
    if qct_parse_run_tool == 'yes':
        # Ensure we have a QCTools report to parse
        if not results['qctools_output_path'] or not os.path.isfile(results['qctools_output_path']):
            logger.critical(f"Unable to check qctools report. No file found at: {results['qctools_output_path']}")
        else:
            # Ensure report directory exists
            if not report_directory:
                report_directory = dir_setup.make_report_dir(source_directory, video_id)

            # Run QCTools parsing
            logger.info(f"Running qct-parse on: {results['qctools_output_path']}")
            run_qctparse(video_path, results['qctools_output_path'], report_directory, check_cancelled=check_cancelled)
            if signals:
                signals.step_completed.emit("QCT Parse")
             # After qct-parse completes, check for color bars results
            colorbars_csv = Path(report_directory) / "qct-parse_colorbars_durations.csv"
            if colorbars_csv.exists():
                start_seconds, end_seconds = parse_colorbars_duration_csv(str(colorbars_csv))
                if end_seconds:
                    results['color_bars_end_time'] = end_seconds
                    logger.info(f"Color bars detected, ending at {end_seconds:.1f}s")

    return results

def run_qctools_command(command, input_path, output_type, output_path, check_cancelled=None, signals=None):
    
    if check_cancelled():
        return None
    
    env = os.environ.copy()
    env['PATH'] = '/usr/local/bin:' + env.get('PATH', '')
    full_command = f"{command} \"{input_path}\" {output_type} {output_path}"
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
    
def parse_colorbars_duration_csv(csv_path):
    """
    Parse the qct-parse color bars duration CSV to extract start and end times.
    
    Args:
        csv_path (str): Path to qct-parse_colorbars_durations.csv
        
    Returns:
        tuple: (start_time_seconds, end_time_seconds) or (None, None) if no bars found
    """
    import csv
    
    if not os.path.exists(csv_path):
        return None, None
    
    try:
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            
            if len(rows) >= 2 and "color bars found" in rows[0][0]:
                # Color bars were found - parse the timestamps
                # Format is HH:MM:SS.ssss
                start_str = rows[1][0]
                end_str = rows[1][1] if len(rows[1]) > 1 else None
                
                # Convert timestamp string to seconds
                def timestamp_to_seconds(ts_str):
                    parts = ts_str.split(':')
                    if len(parts) == 3:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = float(parts[2])
                        return hours * 3600 + minutes * 60 + seconds
                    return 0
                
                start_seconds = timestamp_to_seconds(start_str)
                end_seconds = timestamp_to_seconds(end_str) if end_str else start_seconds
                
                return start_seconds, end_seconds
                
    except Exception as e:
        logger.warning(f"Could not parse color bars duration CSV: {e}")
        
    return None, None