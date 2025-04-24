#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["PATH"] = "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
import time
from art import art, text2art
from dataclasses import asdict

from ..processing.processing_mgmt import ProcessingManager
from ..utils import dir_setup
from ..utils.log_setup import logger
from ..utils.deps_setup import required_commands, check_external_dependency, check_py_version
from ..utils.config_setup import ChecksConfig, SpexConfig
from ..utils.config_manager import ConfigManager

config_mgr = ConfigManager()

def display_processing_banner(video_id=None):
    """
    Display ASCII art banners before and after processing.
    
    Args:
        video_id (str, optional): Video identifier for additional banner
    """
    tape_icon = art('cassette1')
    banner = f'\n{tape_icon}    {tape_icon}    {tape_icon}    {tape_icon}    {tape_icon}\n'
    print(banner)

    if video_id:
        ascii_video_id = text2art(video_id, font='tarty2')
        logger.warning(f'Processing complete:{ascii_video_id}\n')


def print_av_spex_logo():
    avspex_icon = text2art("A-V Spex", font='5lineoblique')
    print(f'{avspex_icon}\n')


def print_nmaahc_logo():
    nmaahc_icon = text2art("nmaahc",font='tarty1')
    print(f'{nmaahc_icon}\n')


def log_overall_time(overall_start_time, overall_end_time):
    logger.warning(f'All files processed!\n')
    overall_total_time = overall_end_time - overall_start_time
    formatted_overall_time = time.strftime("%H:%M:%S", time.gmtime(overall_total_time))
    logger.info(f"Overall processing time for all directories: {formatted_overall_time}\n")

    return formatted_overall_time


class AVSpexProcessor:
    def __init__(self, signals=None):
        # signals are connected in setup_signal_connections() function in gui_main_window
        # passed to AVSpexProcessor from ProcessingWorker
        self.signals = signals
        self._cancelled = False
        self._cancel_emitted = False 
        
        # Force a reload of the config from disk
        self.config_mgr = ConfigManager()
        self.config_mgr.refresh_configs()
        self.checks_config = config_mgr.get_config('checks', ChecksConfig)
        self.spex_config = config_mgr.get_config('spex', SpexConfig)


    def cancel(self):
        self._cancelled = True

    def check_cancelled(self):
        """Check if processing was cancelled and emit signal if needed"""
        if self._cancelled and self.signals and not self._cancel_emitted:
            self.signals.cancelled.emit()
            self._cancel_emitted = True
        return self._cancelled

    def initialize(self):
        """Check all prerequisites before processing"""
        if self.check_cancelled():
            return False

        check_py_version()
        
        if self.check_cancelled():
            return False
        
        for command in required_commands:
            if self.check_cancelled():
                return False
                
            if not check_external_dependency(command):
                error_msg = f"Error: {command} not found. Please install it."
                if self.signals:
                    self.signals.error.emit(error_msg)
                raise RuntimeError(error_msg)
        
        if self.signals:
            self.signals.step_completed.emit("Dependencies Check")
        
        return True

    def process_directories(self, source_directories):
        if self.check_cancelled():
            return False

        overall_start_time = time.time()
        total_dirs = len(source_directories)

        for idx, source_directory in enumerate(source_directories, 1):
            if self.check_cancelled():
                return False

            if self.signals:
                self.signals.file_started.emit(source_directory, idx, total_dirs)
            
            source_directory = os.path.normpath(source_directory)
            self.process_single_directory(source_directory)

        overall_end_time = time.time()
        formatted_time =  log_overall_time(overall_start_time, overall_end_time)

        if self.signals:
            # Signal that all processing is complete
            self.signals.step_completed.emit("All Processing")
            
        return formatted_time

    def process_single_directory(self, source_directory):
        if self.check_cancelled():
            return False

        init_dir_result = dir_setup.initialize_directory(source_directory)
        if init_dir_result is None:
            if self.signals:
                self.signals.error.emit(f"Failed to initialize directory: {source_directory}")
            return False

        video_path, video_id, destination_directory, access_file_found = init_dir_result
        processing_mgmt = ProcessingManager(signals=self.signals, check_cancelled_fn=self.check_cancelled)

        if self.check_cancelled():
            return False

        # Check if fixity is enabled in config
        fixity_enabled = False
        fixity_config = self.checks_config.fixity

        # Check each relevant attribute directly
        if (fixity_config.check_fixity == "yes" or 
            fixity_config.validate_stream_fixity == "yes" or 
            fixity_config.embed_stream_fixity == "yes" or 
            fixity_config.output_fixity == "yes"):
            fixity_enabled = True
            
        if fixity_enabled:
            if self.signals:
                self.signals.tool_started.emit("Fixity...")
            processing_mgmt.process_fixity(source_directory, video_path, video_id)
            if self.signals:
                self.signals.tool_completed.emit("Fixity processing complete")
                

        if self.check_cancelled():
            return False

        # Check if mediaconch is enabled
        mediaconch_enabled = self.checks_config.tools.mediaconch.run_mediaconch == "yes"
        if mediaconch_enabled:
            if self.signals:
                self.signals.tool_started.emit("MediaConch")
                
            mediaconch_results = processing_mgmt.validate_video_with_mediaconch(
                video_path, destination_directory, video_id
            )
            
            if self.signals:
                self.signals.tool_completed.emit("MediaConch validation complete")
                self.signals.step_completed.emit("MediaConch Validation")

        if self.check_cancelled():
            return False

         # Process metadata tools (mediainfo, ffprobe, exiftool, etc.)
        metadata_tools_enabled = False
        tools_config = self.checks_config.tools

        # Check if any metadata tools are enabled
        if (hasattr(tools_config.mediainfo, 'check_tool') and tools_config.mediainfo.check_tool == "yes" or
            hasattr(tools_config.mediatrace, 'check_tool') and tools_config.mediatrace.check_tool == "yes" or
            hasattr(tools_config.exiftool, 'check_tool') and tools_config.exiftool.check_tool == "yes" or
            hasattr(tools_config.ffprobe, 'check_tool') and tools_config.ffprobe.check_tool == "yes"):
            metadata_tools_enabled = True
                    
        # Initialize metadata_differences
        # Needed for process_video_outputs, if not created in process_video_metadata
        metadata_differences = None

        if metadata_tools_enabled:
            if self.signals:
                self.signals.tool_started.emit("Metadata Tools")
            
            metadata_differences = processing_mgmt.process_video_metadata(
                video_path, destination_directory, video_id
            )
            
            if self.signals:
                self.signals.tool_completed.emit("Metadata tools complete")
                # Emit signals for each completed metadata tool
                if tools_config.mediainfo.check_tool == "yes":
                    self.signals.step_completed.emit("Mediainfo")
                if tools_config.mediatrace.check_tool == "yes":
                    self.signals.step_completed.emit("Mediatrace")
                if tools_config.exiftool.check_tool == "yes":
                    self.signals.step_completed.emit("Exiftool")
                if tools_config.ffprobe.check_tool == "yes":
                    self.signals.step_completed.emit("FFprobe")

        if self.check_cancelled():
            return False

        # Process output tools (QCTools, report generation, etc.)
        outputs_enabled = (
            self.checks_config.outputs.access_file == "yes" or
            self.checks_config.outputs.report == "yes" or
            self.checks_config.tools.qctools.run_tool == "yes" or
            self.checks_config.tools.qct_parse.run_tool == "yes"
        )
        
        if outputs_enabled:
            if self.signals:
                self.signals.tool_started.emit("Output Processing")
            
            processing_results = processing_mgmt.process_video_outputs(
                video_path, source_directory, destination_directory,
                video_id, metadata_differences
            )
            
            if self.signals:
                self.signals.tool_completed.emit("Outputs complete")

        if self.check_cancelled():
            return False
        
        if self.signals:
            self.signals.tool_completed.emit("All processing for this directory complete")
        if self.signals:
            self.signals.step_completed.emit("All Processing")
            time.sleep(0.1) # pause for a ms to let the list update before the QMessage box pops up
        
        logger.debug('Please note that any warnings on metadata are just used to help any issues with your file. If they are not relevant at this point in your workflow, just ignore this. Thanks!\n')
        
        display_processing_banner(video_id)
        return True