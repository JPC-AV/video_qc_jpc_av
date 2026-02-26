#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DryRunAnalyzer - Analyzes what would happen during processing without executing tools.

Reports configuration settings and whether preconditions are met for each processing step.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum

from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ChecksConfig, SpexConfig, VALID_QCTOOLS_EXTENSIONS
from AV_Spex.utils.config_manager import ConfigManager

# Import MessageType for color-coded console output
try:
    from AV_Spex.gui.gui_processing_window_console import MessageType
except ImportError:
    # Fallback if GUI module not available (e.g., CLI mode)
    MessageType = None


class StepStatus(Enum):
    """Status of a processing step in dry run analysis."""
    WILL_RUN = "will_run"                    # Enabled and preconditions met
    SKIPPED_DISABLED = "disabled"            # Disabled in config
    SKIPPED_PRECONDITION = "precondition_not_met"  # Enabled but precondition failed
    SKIPPED_ALREADY_EXISTS = "already_exists"  # Output already exists


@dataclass
class StepAnalysis:
    """Analysis result for a single processing step."""
    step_name: str
    status: StepStatus
    enabled_in_config: bool
    precondition_met: bool
    reason: str
    details: Optional[str] = None


class DryRunAnalyzer:
    """
    Analyzes processing configuration and directory state to report
    what would happen during actual processing.
    """
    
    def __init__(self, signals=None):
        self.signals = signals
        self._cancelled = False
        
        self.config_mgr = ConfigManager()
        self.config_mgr.refresh_configs()
        self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
        self.spex_config = self.config_mgr.get_config('spex', SpexConfig)
    
    def cancel(self):
        self._cancelled = True
    
    def check_cancelled(self):
        return self._cancelled
    
    def analyze_directories(self, source_directories: List[str]) -> Dict[str, List[StepAnalysis]]:
        """
        Analyze multiple directories and return what would happen during processing.
        
        Returns:
            Dict mapping directory paths to their step analyses
        """
        results = {}
        total_dirs = len(source_directories)
        
        self._log_header("DRY RUN ANALYSIS")
        self._log_config_summary()
        
        for idx, source_directory in enumerate(source_directories, 1):
            if self.check_cancelled():
                return results
            
            source_directory = os.path.normpath(source_directory)
            self._log_header(f"Directory {idx}/{total_dirs}: {os.path.basename(source_directory)}")
            
            analyses = self.analyze_single_directory(source_directory)
            results[source_directory] = analyses
            
            self._log_analysis_summary(analyses)
        
        self._log_header("DRY RUN COMPLETE")
        return results
    
    def analyze_single_directory(self, source_directory: str) -> List[StepAnalysis]:
        """Analyze a single directory and return step analyses."""
        analyses = []
        
        # Initialize directory to get video info
        init_result = self._initialize_for_analysis(source_directory)
        if init_result is None:
            logger.error(f"Failed to initialize directory: {source_directory}")
            return [StepAnalysis(
                step_name="Directory Initialization",
                status=StepStatus.SKIPPED_PRECONDITION,
                enabled_in_config=True,
                precondition_met=False,
                reason="Could not find valid video file in directory"
            )]
        
        video_path, video_id, destination_directory, access_file_found = init_result
        logger.info(f"Video file: {os.path.basename(video_path)}")
        logger.info(f"Video ID: {video_id}")
        logger.info(f"Destination: {destination_directory}\n")
        
        # Analyze each processing step
        analyses.extend(self._analyze_fixity_steps(source_directory, video_path, video_id))
        analyses.extend(self._analyze_mediaconch(video_path, destination_directory, video_id))
        analyses.extend(self._analyze_metadata_tools(video_path, destination_directory, video_id))
        analyses.extend(self._analyze_output_steps(
            video_path, source_directory, destination_directory, video_id, access_file_found
        ))
        analyses.extend(self._analyze_frame_analysis(
            video_path, source_directory, destination_directory, video_id
        ))
        
        return analyses
    
    # -------------------------------------------------------------------------
    # Init Dir
    # -------------------------------------------------------------------------
    
    def _initialize_for_analysis(self, source_directory: str):
        """
        Simplified directory initialization for dry run analysis.
        Does not create directories or move files—just identifies the video
        and verifies naming consistency.
        
        Returns:
            tuple: (video_path, video_id, destination_directory, access_file_found)
            None: if no valid video file found
        """
        # Find MKV files (excluding qctools reports)
        found_mkvs = [
            f for f in os.listdir(source_directory)
            if f.lower().endswith('.mkv') and 'qctools' not in f.lower()
        ]
        
        if not found_mkvs:
            logger.error(f"No MKV video file found in: {source_directory}")
            return None
        
        if len(found_mkvs) > 1:
            logger.error(f"Multiple MKV files found in {source_directory}: {found_mkvs}")
            return None
        
        video_path = os.path.join(source_directory, found_mkvs[0])
        video_id = os.path.splitext(found_mkvs[0])[0]
        
        # Verify directory name matches video_id
        directory_name = os.path.basename(source_directory)
        if not directory_name.startswith(video_id):
            logger.warning(
                f'Directory name "{directory_name}" does not match video file "{video_id}"'
            )
        
        # Determine destination directory path (don't create it)
        destination_directory = os.path.join(source_directory, f'{video_id}_qc_metadata')
        
        # Check for existing access file
        access_file_found = next(
            (f for f in os.listdir(source_directory) if f.lower().endswith('.mp4')),
            None
        )
        
        return video_path, video_id, destination_directory, access_file_found
    
    # -------------------------------------------------------------------------
    # Fixity Analysis
    # -------------------------------------------------------------------------
    
    def _analyze_fixity_steps(self, source_directory: str, video_path: str, 
                              video_id: str) -> List[StepAnalysis]:
        """Analyze all fixity-related steps."""
        analyses = []
        fixity_config = self.checks_config.fixity
        
        # Check if file is MKV (required for stream fixity)
        is_mkv = video_path.lower().endswith('.mkv')
        
        # Embed Stream Fixity
        analyses.append(self._analyze_step(
            step_name="Embed Stream Fixity",
            enabled=fixity_config.embed_stream_fixity,
            precondition_met=is_mkv,
            precondition_reason="File must be MKV format" if not is_mkv else None
        ))
        
        # Validate Stream Fixity
        # Note: If embed is enabled, validate is skipped
        if fixity_config.embed_stream_fixity:
            validate_reason = "Skipped when Embed Stream Fixity is enabled"
            validate_met = False
        else:
            validate_met = is_mkv
            validate_reason = None
            if not is_mkv:
                validate_reason = "File must be MKV format"
        
        analyses.append(self._analyze_step(
            step_name="Validate Stream Fixity",
            enabled=fixity_config.validate_stream_fixity,
            precondition_met=validate_met,
            precondition_reason=validate_reason
        ))
        
        # Output Fixity
        analyses.append(self._analyze_step(
            step_name="Output Fixity",
            enabled=fixity_config.output_fixity,
            precondition_met=True,  # Always possible
            precondition_reason=None
        ))
        
        # Check/Validate Fixity
        fixity_file = self._find_fixity_file(source_directory, video_id)
        has_fixity_file = fixity_file is not None
        analyses.append(self._analyze_step(
            step_name="Validate Fixity (against stored checksum)",
            enabled=fixity_config.check_fixity,
            precondition_met=has_fixity_file,
            precondition_reason=(
                f"Found: {os.path.basename(fixity_file)}" if has_fixity_file 
                else "No checksum file found (.md5, .sha256, or _fixity.txt)"
            )
        ))
        
        return analyses
    
    def _find_fixity_file(self, source_directory: str, video_id: str) -> Optional[str]:
        """Check if fixity sidecar file exists."""
        source_path = Path(source_directory)
        
        # Match the suffixes recognized by fixity_check.py
        # Check source directory for checksum/fixity files
        fixity_patterns = [
            f"{video_id}*_checksums.md5",
            f"{video_id}*_fixity.md5",
            f"{video_id}*_checksums.sha256",
            f"{video_id}*_fixity.sha256",
            f"{video_id}*_fixity.txt",
            # Fallback to any matching extension
            f"{video_id}*.md5",
            f"{video_id}*.sha256",
        ]
        for pattern in fixity_patterns:
            matches = list(source_path.glob(pattern))
            if matches:
                return str(matches[0])
        
        # Check for fixity_check.txt in metadata folder
        metadata_dir = source_path / f"{video_id}_qc_metadata"
        if metadata_dir.exists():
            fixity_files = list(metadata_dir.glob("*_fixity_check.txt"))
            if fixity_files:
                return str(fixity_files[0])
        
        return None
        
    # -------------------------------------------------------------------------
    # MediaConch Analysis
    # -------------------------------------------------------------------------
    
    def _analyze_mediaconch(self, video_path: str, destination_directory: str,
                            video_id: str) -> List[StepAnalysis]:
        """Analyze MediaConch validation step."""
        mc_config = self.checks_config.tools.mediaconch
        
        policy_name = mc_config.mediaconch_policy
        policy_path = self.config_mgr.get_policy_path(policy_name) if policy_name else None
        
        precondition_met = policy_path is not None and os.path.exists(policy_path)
        reason = None
        if not policy_name:
            reason = "No policy file configured"
        elif not precondition_met:
            reason = f"Policy file not found: {policy_name}"
        else:
            reason = f"Using policy: {policy_name}"
        
        return [self._analyze_step(
            step_name="MediaConch Validation",
            enabled=mc_config.run_mediaconch,
            precondition_met=precondition_met,
            precondition_reason=reason
        )]
    
    # -------------------------------------------------------------------------
    # Metadata Tools Analysis
    # -------------------------------------------------------------------------
    
    def _analyze_metadata_tools(self, video_path: str, destination_directory: str,
                                 video_id: str) -> List[StepAnalysis]:
        """Analyze metadata extraction and checking tools."""
        analyses = []
        tools_config = self.checks_config.tools
        
        tools = [
            ('mediainfo', 'MediaInfo'),
            ('mediatrace', 'MediaTrace'),
            ('exiftool', 'ExifTool'),
            ('ffprobe', 'FFprobe')
        ]
        
        for tool_attr, display_name in tools:
            tool_config = getattr(tools_config, tool_attr)
            
            # Run tool analysis
            analyses.append(self._analyze_step(
                step_name=f"{display_name} (run)",
                enabled=tool_config.run_tool,
                precondition_met=True, 
                precondition_reason=None
            ))
            
            # Check tool analysis - needs output to exist OR run_tool enabled
            existing_output = self._find_tool_output(destination_directory, video_id, tool_attr)
            can_check = tool_config.run_tool or existing_output is not None
            
            reason = None
            if not can_check:
                reason = f"No existing {display_name} output and run_tool is disabled"
            elif tool_config.run_tool:
                reason = "Will check output after running tool"
            elif existing_output:
                reason = f"Will use existing output: {os.path.basename(existing_output)}"
            
            analyses.append(self._analyze_step(
                step_name=f"{display_name} (check against expected values)",
                enabled=tool_config.check_tool,
                precondition_met=can_check,
                precondition_reason=reason
            ))
        
        return analyses
    
    def _find_tool_output(self, destination_directory: str, video_id: str, 
                      tool_name: str) -> Optional[str]:
        """Find existing tool output file in the destination directory."""
        extensions = {
            'mediainfo': '_mediainfo_output.json',
            'mediatrace': '_mediatrace_output.xml',
            'exiftool': '_exiftool_output.json',
            'ffprobe': '_ffprobe_output.txt'
        }
        
        ext = extensions.get(tool_name)
        if not ext:
            return None
        
        # destination_directory is already the _qc_metadata directory
        if os.path.isdir(destination_directory):
            output_path = os.path.join(destination_directory, f"{video_id}{ext}")
            if os.path.exists(output_path):
                return output_path
        
        return None
    # -------------------------------------------------------------------------
    # Output Steps Analysis
    # -------------------------------------------------------------------------
    
    def _analyze_output_steps(self, video_path: str, source_directory: str,
                               destination_directory: str, video_id: str,
                               access_file_found: bool) -> List[StepAnalysis]:
        """Analyze output generation steps."""
        analyses = []
        outputs_config = self.checks_config.outputs
        qctools_config = self.checks_config.tools.qctools
        qct_parse_config = self.checks_config.tools.qct_parse
        
        # QCTools
        existing_qctools = self._find_qctools_report(source_directory, video_id)
        qctools_reason = None
        qctools_status_override = None
        
        if existing_qctools and qctools_config.run_tool:
            qctools_reason = f"Existing report found: {os.path.basename(existing_qctools)} (will use existing)"
            qctools_status_override = StepStatus.SKIPPED_ALREADY_EXISTS
        elif existing_qctools:
            qctools_reason = f"Existing report found: {os.path.basename(existing_qctools)}"
        else:
            qctools_reason = "Will create new report"
            
        analyses.append(self._analyze_step(
            step_name="QCTools (generate report)",
            enabled=qctools_config.run_tool,
            precondition_met=True,
            precondition_reason=qctools_reason,
            status_override=qctools_status_override
        ))
        
        # QCT Parse
        can_parse = existing_qctools is not None or qctools_config.run_tool
        analyses.append(self._analyze_step(
            step_name="QCT Parse (analyze QCTools report)",
            enabled=qct_parse_config.run_tool,
            precondition_met=can_parse,
            precondition_reason="No QCTools report available and QCTools run is disabled" if not can_parse else None
        ))
        
        # Access File
        access_status = None
        access_reason = None
        if access_file_found:
            access_status = StepStatus.SKIPPED_ALREADY_EXISTS
            access_reason = "Access file already exists in directory"
        
        analyses.append(self._analyze_step(
            step_name="Access File (create proxy)",
            enabled=outputs_config.access_file,
            precondition_met=not access_file_found,
            precondition_reason=access_reason,
            status_override=access_status
        ))
        
        # HTML Report
        analyses.append(self._analyze_step(
            step_name="HTML Report",
            enabled=outputs_config.report,
            precondition_met=True,
            precondition_reason=None
        ))
        
        return analyses
    
    def _find_qctools_report(self, source_directory: str, video_id: str) -> Optional[str]:
        """Find existing QCTools report."""
        source_path = Path(source_directory)
        
        search_folders = [
            source_path / f"{video_id}_qc_metadata",
            source_path / f"{video_id}_vrecord_metadata"
        ]
        
        patterns = ["*.qctools.xml.gz", "*.qctools.mkv"]
        
        for folder in search_folders:
            if folder.exists():
                for pattern in patterns:
                    matches = list(folder.glob(pattern))
                    if matches:
                        return str(matches[0])
        return None
    
    # -------------------------------------------------------------------------
    # Frame Analysis
    # -------------------------------------------------------------------------
    
    def _analyze_frame_analysis(self, video_path: str, source_directory: str,
                                 destination_directory: str, 
                                 video_id: str) -> List[StepAnalysis]:
        """Analyze frame analysis steps (border detection, BRNG analysis, signalstats)."""
        analyses = []
        frame_config = self.checks_config.outputs.frame_analysis
        qctools_config = self.checks_config.tools.qctools
        qct_parse_config = self.checks_config.tools.qct_parse
        
        # Check for existing QCTools report (needed for BRNG and signalstats)
        existing_qctools = self._find_qctools_report(source_directory, video_id)
        has_qctools = existing_qctools is not None or qctools_config.run_tool
        
        # Border Detection
        analyses.append(self._analyze_step(
            step_name="Frame Analysis: Border Detection",
            enabled=frame_config.enable_border_detection,
            precondition_met=True,
            precondition_reason=f"Mode: {frame_config.border_detection_mode}"
        ))
        
        # BRNG Analysis
        brng_reason = None
        if not has_qctools:
            brng_reason = "No QCTools report available and QCTools run is disabled (needed for QCTools-based BRNG analysis)"
        else:
            details = []
            details.append(f"Duration limit: {frame_config.brng_duration_limit}s")
            if frame_config.brng_skip_color_bars:
                details.append("Will skip color bars region")
            brng_reason = ", ".join(details)
        
        analyses.append(self._analyze_step(
            step_name="Frame Analysis: BRNG Violation Analysis",
            enabled=frame_config.enable_brng_analysis,
            precondition_met=True,  # Can run with or without QCTools (uses FFmpeg fallback)
            precondition_reason=brng_reason
        ))
        
        # Signalstats
        signalstats_reason = None
        signalstats_met = True
        
        if not frame_config.enable_border_detection:
            signalstats_reason = "Border detection is disabled (signalstats requires border data for region-based analysis)"
            signalstats_met = False
        else:
            signalstats_reason = f"Periods: {frame_config.analysis_period_count}, duration: {frame_config.analysis_period_duration}s each"
        
        analyses.append(self._analyze_step(
            step_name="Frame Analysis: Signalstats",
            enabled=frame_config.enable_signalstats,
            precondition_met=signalstats_met,
            precondition_reason=signalstats_reason
        ))
        
        return analyses
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _analyze_step(self, step_name: str, enabled: bool, precondition_met: bool,
                      precondition_reason: Optional[str] = None,
                      status_override: Optional[StepStatus] = None) -> StepAnalysis:
        """Create a StepAnalysis with appropriate status."""
        
        if status_override:
            status = status_override
        elif not enabled:
            status = StepStatus.SKIPPED_DISABLED
        elif not precondition_met:
            status = StepStatus.SKIPPED_PRECONDITION
        else:
            status = StepStatus.WILL_RUN
        
        # Build reason string
        if status == StepStatus.SKIPPED_DISABLED:
            reason = "Disabled in configuration"
        elif status == StepStatus.SKIPPED_PRECONDITION:
            reason = precondition_reason or "Precondition not met"
        elif status == StepStatus.SKIPPED_ALREADY_EXISTS:
            reason = precondition_reason or "Output already exists"
        else:
            reason = precondition_reason or "Ready to run"
        
        return StepAnalysis(
            step_name=step_name,
            status=status,
            enabled_in_config=enabled,
            precondition_met=precondition_met,
            reason=reason
        )
    
    def _log_header(self, text: str):
        """Log a section header."""
        separator = "=" * 60
        logger.warning(f"\n{separator}")
        logger.warning(f"  {text}")
        logger.warning(f"{separator}\n")
    
    def _log_config_summary(self):
        """Log a summary of current configuration."""
        logger.info("Current Configuration Summary:")
        
        # Fixity
        fx = self.checks_config.fixity
        logger.info(f"  Fixity:")
        logger.info(f"    - Embed Stream Fixity: {fx.embed_stream_fixity}")
        logger.info(f"    - Validate Stream Fixity: {fx.validate_stream_fixity}")
        logger.info(f"    - Output Fixity: {fx.output_fixity}")
        logger.info(f"    - Check Fixity: {fx.check_fixity}")
        
        # Tools
        logger.info(f"  Tools:")
        for tool in ['mediainfo', 'mediatrace', 'exiftool', 'ffprobe']:
            t = getattr(self.checks_config.tools, tool)
            logger.info(f"    - {tool}: run={t.run_tool}, check={t.check_tool}")
        
        mc = self.checks_config.tools.mediaconch
        logger.info(f"    - mediaconch: run={mc.run_mediaconch}, policy={mc.mediaconch_policy}")
        
        qc = self.checks_config.tools.qctools
        logger.info(f"    - qctools: run={qc.run_tool}")
        
        qp = self.checks_config.tools.qct_parse
        logger.info(f"    - qct_parse: run={qp.run_tool}")
        
        # Outputs
        out = self.checks_config.outputs
        logger.info(f"  Outputs:")
        logger.info(f"    - Access File: {out.access_file}")
        logger.info(f"    - Report: {out.report}")
        
        # Frame Analysis
        fa = self.checks_config.outputs.frame_analysis
        logger.info(f"  Frame Analysis:")
        logger.info(f"    - Border Detection: {fa.enable_border_detection} (mode: {fa.border_detection_mode})")
        logger.info(f"    - BRNG Analysis: {fa.enable_brng_analysis}")
        logger.info(f"    - Signalstats: {fa.enable_signalstats}")
        logger.info("")
    
    def _log_analysis_summary(self, analyses: List[StepAnalysis]):
        """Log analysis results in a readable format with color-coded output."""
        
        will_run = [a for a in analyses if a.status == StepStatus.WILL_RUN]
        skipped_disabled = [a for a in analyses if a.status == StepStatus.SKIPPED_DISABLED]
        skipped_precondition = [a for a in analyses if a.status == StepStatus.SKIPPED_PRECONDITION]
        skipped_exists = [a for a in analyses if a.status == StepStatus.SKIPPED_ALREADY_EXISTS]
        
        # Helper to get extra dict for color-coded logging
        def get_msg_extra(msg_type):
            if MessageType is not None:
                return {'msg_type': msg_type}
            return {}
        
        if will_run:
            logger.info("Steps that WILL RUN:", extra=get_msg_extra(MessageType.SUCCESS if MessageType else None))
            for analysis in will_run:
                logger.info(f"  ✓ {analysis.step_name}", extra=get_msg_extra(MessageType.SUCCESS if MessageType else None))
                if analysis.reason and analysis.reason != "Ready to run":
                    logger.debug(f"      {analysis.reason}")
        
        if skipped_disabled:
            logger.info("\nSteps DISABLED in configuration:", extra=get_msg_extra(MessageType.NORMAL if MessageType else None))
            for analysis in skipped_disabled:
                logger.info(f"  ○ {analysis.step_name}", extra=get_msg_extra(MessageType.NORMAL if MessageType else None))
                logger.info(f"      Reason: {analysis.reason}", extra=get_msg_extra(MessageType.NORMAL if MessageType else None))
        
        if skipped_precondition:
            logger.info("\nSteps SKIPPED (precondition not met):", extra=get_msg_extra(MessageType.WARNING if MessageType else None))
            for analysis in skipped_precondition:
                logger.info(f"  ✗ {analysis.step_name}", extra=get_msg_extra(MessageType.WARNING if MessageType else None))
                logger.info(f"      Reason: {analysis.reason}", extra=get_msg_extra(MessageType.WARNING if MessageType else None))
        
        if skipped_exists:
            logger.info("\nSteps SKIPPED (output already exists):", extra=get_msg_extra(MessageType.INFO if MessageType else None))
            for analysis in skipped_exists:
                logger.info(f"  ● {analysis.step_name}", extra=get_msg_extra(MessageType.INFO if MessageType else None))
                logger.info(f"      Reason: {analysis.reason}", extra=get_msg_extra(MessageType.INFO if MessageType else None))
        
        logger.info("")