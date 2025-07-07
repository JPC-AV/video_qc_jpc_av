#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["PATH"] = "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
import time
from art import art, text2art
from dataclasses import asdict

from AV_Spex.processing.processing_mgmt import ProcessingManager
from AV_Spex.utils import dir_setup
from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ChecksConfig, SpexConfig
from AV_Spex.utils.config_manager import ConfigManager

from PyQt6.QtWidgets import QApplication

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

        # ADD THESE LINES:
        self._paused = False
        self._pause_requested = False
        self._current_step = 'fixity'
        self._completed_steps = set()
        self._processing_context = None

        if self.signals:
            self.signals.pause_requested.connect(self.request_pause)
            self.signals.resume_requested.connect(self.request_resume)

        self.config_mgr = ConfigManager()
        # logger.debug("==== PROCESSOR INITIALIZATION DEBUGGING ====")
        
        # Get the config before refresh
        pre_refresh_spex = self.config_mgr.get_config('spex', SpexConfig, use_last_used=True)
        # logger.debug(f"Pre-refresh spex config has {len(pre_refresh_spex.filename_values.fn_sections)} sections")
        
        self.config_mgr.refresh_configs()
        
        # Get configs after refresh
        self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
        self.spex_config = self.config_mgr.get_config('spex', SpexConfig)
        
        # Log details about the refreshed config
        # logger.debug(f"Post-refresh spex config has {len(self.spex_config.filename_values.fn_sections)} sections")
        # for idx, (key, section) in enumerate(sorted(self.spex_config.filename_values.fn_sections.items()), 1):
        #     logger.debug(f"  Section {idx}: {key} = {section.value} ({section.section_type})")

    def cancel(self):
        self._cancelled = True

    def check_cancelled(self):
        """Check for cancellation OR pause - reuse existing mechanism"""
        # Handle actual cancellation
        if self._cancelled and self.signals and not self._cancel_emitted:
            self.signals.cancelled.emit()
            self._cancel_emitted = True
            return True
        
        # Handle pause request
        if self._pause_requested and not self._paused:
            self._paused = True
            self._pause_requested = False
            if self.signals:
                self.signals.paused.emit()
                self.signals.status_update.emit("Processing paused")
            print(f"DEBUG: Paused during step {self._current_step}")
        
        # Return True for BOTH cancel AND pause
        return_value = self._cancelled or self._paused
        if return_value:
            print(f"DEBUG: check_cancelled returning True - cancelled: {self._cancelled}, paused: {self._paused}")
        return return_value

    
    def request_pause(self):
        """Request pause - will pause at next check_cancelled call"""
        print("DEBUG: Pause requested!")
        self._pause_requested = True
        # Add debug about current state
        print(f"DEBUG: _pause_requested = {self._pause_requested}")
        print(f"DEBUG: _paused = {self._paused}")
        print(f"DEBUG: Current step: {self._current_step}")

    def request_resume(self):
        """Resume processing"""
        print("DEBUG: Resume requested!")
        self._paused = False
        self._pause_requested = False
        if self.signals:
            self.signals.resumed.emit()

    def process_directories(self, source_directories):
        """Process directories with step-level resume"""
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
            
            # Keep processing this directory until completed or cancelled
            while True:
                result = self.process_single_directory(source_directory)
                
                if result == "paused":
                    print(f"DEBUG: Directory processing paused at step {self._current_step}")
                    # Wait for resume
                    while self._paused and not self._cancelled:
                        if self.signals:
                            QApplication.processEvents()
                        time.sleep(0.1)
                    # Continue from where we left off
                    continue
                elif result == False:
                    return False
                else:
                    break  # Directory completed

        overall_end_time = time.time()
        formatted_time = log_overall_time(overall_start_time, overall_end_time)

        if self.signals:
            self.signals.step_completed.emit("All Processing")
            
        return formatted_time

    def process_single_directory(self, source_directory):
        """Process directory with step-level pause/resume using existing check_cancelled"""
        
        # Initialize directory context (only once)
        if not hasattr(self, '_processing_context') or self._processing_context is None:
            init_dir_result = dir_setup.initialize_directory(source_directory)
            if init_dir_result is None:
                if self.signals:
                    self.signals.error.emit(f"Failed to initialize directory: {source_directory}")
                return False

            video_path, video_id, destination_directory, access_file_found = init_dir_result
            
            self._processing_context = {
                'source_directory': source_directory,
                'video_path': video_path,
                'video_id': video_id,
                'destination_directory': destination_directory,
                'access_file_found': access_file_found
            }
            
            # Initialize step tracking
            self._completed_steps = getattr(self, '_completed_steps', set())
            self._current_step = getattr(self, '_current_step', 'fixity')

        processing_mgmt = ProcessingManager(signals=self.signals, check_cancelled_fn=self.check_cancelled)

        # STEP 1: Fixity
        if 'fixity' not in self._completed_steps:
            self._current_step = 'fixity'
            if self._run_fixity_step(processing_mgmt):
                self._completed_steps.add('fixity')
            elif self.check_cancelled():
                if self._paused:
                    print("DEBUG: Fixity step paused")
                    return "paused"  # Will restart this step on resume
                else:
                    return False  # Actual cancellation

        # STEP 2: MediaConch  
        if 'mediaconch' not in self._completed_steps:
            self._current_step = 'mediaconch'
            if self._run_mediaconch_step(processing_mgmt):
                self._completed_steps.add('mediaconch')
            elif self.check_cancelled():
                if self._paused:
                    print("DEBUG: MediaConch step paused")
                    return "paused"
                else:
                    return False

        # STEP 3: Metadata
        if 'metadata' not in self._completed_steps:
            self._current_step = 'metadata'
            if self._run_metadata_step(processing_mgmt):
                self._completed_steps.add('metadata')
            elif self.check_cancelled():
                if self._paused:
                    print("DEBUG: Metadata step paused")
                    return "paused"
                else:
                    return False

        # STEP 4: Outputs
        if 'outputs' not in self._completed_steps:
            self._current_step = 'outputs'
            if self._run_outputs_step(processing_mgmt):
                self._completed_steps.add('outputs')
            elif self.check_cancelled():
                if self._paused:
                    print("DEBUG: Outputs step paused")
                    return "paused"
                else:
                    return False

        # All steps completed successfully
        self._complete_processing()
        return True

    def _run_fixity_step(self, processing_mgmt):
        """Run fixity step - existing logic, just returns success/failure"""
        print("DEBUG: _run_fixity_step starting")
        
        fixity_enabled = (
            self.checks_config.fixity.check_fixity == "yes" or 
            self.checks_config.fixity.validate_stream_fixity == "yes" or 
            self.checks_config.fixity.embed_stream_fixity == "yes" or 
            self.checks_config.fixity.output_fixity == "yes"
        )
        
        if not fixity_enabled:
            print("DEBUG: Fixity not enabled, returning True")
            return True

        if self.signals:
            self.signals.tool_started.emit("Fixity...")
        
        print("DEBUG: About to call processing_mgmt.process_fixity")
        
        try:
            ctx = self._processing_context
            processing_mgmt.process_fixity(ctx['source_directory'], ctx['video_path'], ctx['video_id'])
            
            # CHECK FOR PAUSE AFTER THE OPERATION
            if self.check_cancelled():
                if self._paused:
                    print("DEBUG: Fixity step was paused, not completed")
                    return False  # Return False so the step isn't marked complete
                else:
                    print("DEBUG: Fixity step was cancelled")
                    return False
            
            print("DEBUG: process_fixity completed normally")
            
            if self.signals:
                self.signals.tool_completed.emit("Fixity processing complete")
            return True
            
        except Exception as e:
            print(f"DEBUG: Fixity step error: {e}")
            return False

    def _run_mediaconch_step(self, processing_mgmt):
        """Run MediaConch step"""
        if self.checks_config.tools.mediaconch.run_mediaconch != "yes":
            return True

        if self.signals:
            self.signals.tool_started.emit("MediaConch")
        
        try:
            ctx = self._processing_context
            processing_mgmt.validate_video_with_mediaconch(ctx['video_path'], ctx['destination_directory'], ctx['video_id'])
            
            # CHECK FOR PAUSE AFTER THE OPERATION
            if self.check_cancelled():
                if self._paused:
                    print("DEBUG: MediaConch step was paused, not completed")
                    return False  # Return False so the step isn't marked complete
                else:
                    print("DEBUG: MediaConch step was cancelled")
                    return False
            
            if self.signals:
                self.signals.tool_completed.emit("MediaConch validation complete")
                self.signals.step_completed.emit("MediaConch Validation")
            return True
            
        except Exception as e:
            print(f"DEBUG: MediaConch step error: {e}")
            return False

    def _run_metadata_step(self, processing_mgmt):
        """Run metadata step"""
        # Check if any metadata tools are enabled
        tools_to_check = ['mediainfo', 'mediatrace', 'exiftool', 'ffprobe']
        metadata_tools_enabled = False

        for tool_name in tools_to_check:
            tool = getattr(tools_config, tool_name, None)
            if tool and (getattr(tool, 'check_tool', 'no') == 'yes' or 
                        getattr(tool, 'run_tool', 'no') == 'yes'):
                    metadata_tools_enabled = True

        if self.signals:
            self.signals.tool_started.emit("Metadata Tools")
        
        try:
            ctx = self._processing_context
            metadata_differences = processing_mgmt.process_video_metadata(ctx['video_path'], ctx['destination_directory'], ctx['video_id'])
            
            # CHECK FOR PAUSE AFTER THE OPERATION
            if self.check_cancelled():
                if self._paused:
                    print("DEBUG: Metadata step was paused, not completed")
                    return False  # Return False so the step isn't marked complete
                else:
                    print("DEBUG: Metadata step was cancelled")
                    return False
            
            # Store for outputs step
            ctx['metadata_differences'] = metadata_differences
            
            if self.signals:
                self.signals.tool_completed.emit("Metadata tools complete")
                # Emit signals for each completed metadata tool
                tools_to_signal = [
                    ('mediainfo', 'Mediainfo'),
                    ('mediatrace', 'Mediatrace'), 
                    ('exiftool', 'Exiftool'),
                    ('ffprobe', 'FFprobe')
                ]
                
                for tool_name, display_name in tools_to_signal:
                    tool = getattr(tools_config, tool_name)
                    if tool.check_tool == "yes" or tool.run_tool == "yes":
                        self.signals.step_completed.emit(display_name)

            if self.signals:
                self.signals.clear_status.emit()

            if self.check_cancelled():
                return True
            
        except Exception as e:
            print(f"DEBUG: Metadata step error: {e}")
            return False

    def _run_outputs_step(self, processing_mgmt):
        """Run outputs step"""
        outputs_enabled = (
            self.checks_config.outputs.access_file == "yes" or
            self.checks_config.outputs.report == "yes" or
            self.checks_config.tools.qctools.run_tool == "yes" or
            self.checks_config.tools.qct_parse.run_tool == "yes"
        )
        
        if not outputs_enabled:
            return True

        if self.signals:
            self.signals.tool_started.emit("Output Processing\n")
        
        try:
            ctx = self._processing_context
            metadata_differences = ctx.get('metadata_differences')
            processing_mgmt.process_video_outputs(
                ctx['video_path'], ctx['source_directory'], ctx['destination_directory'],
                ctx['video_id'], metadata_differences
            )
            
            # CHECK FOR PAUSE AFTER THE OPERATION
            if self.check_cancelled():
                if self._paused:
                    print("DEBUG: Outputs step was paused, not completed")
                    return False  # Return False so the step isn't marked complete
                else:
                    print("DEBUG: Outputs step was cancelled")
                    return False
            
            # CHECK FOR PAUSE AFTER THE OPERATION
            if self.check_cancelled():
                if self._paused:
                    print("DEBUG: Outputs step was paused, not completed")
                    return False  # Return False so the step isn't marked complete
                else:
                    print("DEBUG: Outputs step was cancelled")
                    return False
            
            if self.signals:
                self.signals.tool_completed.emit("Outputs complete\n")

        if self.check_cancelled():
            return False

    def _complete_processing(self):
        """Complete processing and reset state"""
        ctx = self._processing_context
        
        if self.signals:
            self.signals.tool_completed.emit("All processing for this directory complete\n")
        if self.signals:
            self.signals.step_completed.emit("All Processing")
            time.sleep(0.1)
        
        logger.debug('Please note that any warnings on metadata are just used to help any issues with your file. If they are not relevant at this point in your workflow, just ignore this. Thanks!\n')

        if self.signals:
            self.signals.clear_status.emit()
        
        display_processing_banner(video_id)
        return True