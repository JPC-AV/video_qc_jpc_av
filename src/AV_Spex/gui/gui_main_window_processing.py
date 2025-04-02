from PyQt6.QtWidgets import QApplication, QMessageBox
from ..gui.gui_processing_gui import ProcessingWindow

from ..utils.log_setup import logger
from ..processing.worker_thread import ProcessingWorker

class MainWindowProcessing:
    """Processing-related event handlers for the main window"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def call_process_directories(self):
        """Initialize and start the worker thread"""
        try:
            # Create the processing window if it doesn't exist
            if not hasattr(self.parent, 'processing_window') or self.parent.processing_window is None:
                # Create and initialize the processing window
                self.initialize_processing_window()
            
            # Create and configure the worker
            self.parent.worker = ProcessingWorker(self.parent.source_directories, self.parent.signals)
            
            # Connect worker-specific signals
            self.parent.worker.started_processing.connect(self.on_processing_started)
            self.parent.worker.finished.connect(self.on_worker_finished)
            self.parent.worker.error.connect(self.on_error)
            self.parent.worker.processing_time.connect(self.on_processing_time)
            
            # Start the worker thread
            self.parent.worker.start()
            
        except Exception as e:
            self.parent.signals.error.emit(str(e))

    def initialize_processing_window(self):
        """Create and configure the processing window and connect signals"""
        self.parent.processing_window = ProcessingWindow(self.parent)
            
        # Connect signals to the processing window 
        self.parent.signals.status_update.connect(self.parent.processing_window.update_status)
        self.parent.signals.error.connect(self.parent.processing_window.update_status)
        self.parent.signals.progress.connect(self.update_progress)
        self.parent.signals.file_started.connect(self.parent.processing_window.update_file_status)

        # reset steps list when a new file starts
        self.parent.signals.file_started.connect(self.parent.processing_window.reset_steps_list)

        # Progress bar signal connections
        self.parent.signals.stream_hash_progress.connect(self.parent.processing_window.update_detail_progress)
        self.parent.signals.md5_progress.connect(self.parent.processing_window.update_detail_progress)
        self.parent.signals.access_file_progress.connect(self.parent.processing_window.update_detail_progress)
            
        # Connect the step_completed signal
        self.parent.signals.step_completed.connect(self.parent.processing_window.mark_step_complete)
            
        # Connect the cancel button
        self.parent.processing_window.cancel_button.clicked.connect(self.cancel_processing)

        # Connect open processing button
        if hasattr(self.parent, 'open_processing_button'):
            self.parent.open_processing_button.setText("Show Processing Window")
        
        # Show the window
        self.parent.processing_window.show()
        self.parent.processing_window.raise_()
    
    def update_progress(self, current, total):
        """Update progress bar in the processing window."""
        if hasattr(self.parent, 'processing_window') and self.parent.processing_window:
            self.parent.processing_window.progress_bar.setMaximum(total)
            self.parent.processing_window.progress_bar.setValue(current)

    def on_worker_finished(self):
        """Handle worker thread completion."""
        # Check if this was a cancellation
        was_cancelled = hasattr(self.parent.worker, 'user_cancelled') and self.parent.worker.user_cancelled

        # Hide the processing indicator
        self.parent.processing_indicator.setVisible(False)
        self.parent.main_status_label.setVisible(False)
        
        # Update UI to indicate processing is complete
        if hasattr(self.parent, 'processing_window') and self.parent.processing_window:
            if not was_cancelled:
                self.parent.processing_window.update_status("Processing completed successfully!")
                self.parent.processing_window.progress_bar.setMaximum(100)
                self.parent.processing_window.progress_bar.setValue(100)
            
            # Change the cancel button to a close button
            self.parent.processing_window.cancel_button.setText("Close")
            self.parent.processing_window.cancel_button.setEnabled(True)
            
            # Disconnect previous handler if any (use try/except in case it's not connected)
            try:
                self.parent.processing_window.cancel_button.clicked.disconnect()
            except TypeError:
                # This catches the case where no connections exist
                pass
                
            self.parent.processing_window.cancel_button.clicked.connect(self.parent.processing_window.close)
        
        # Re-enable the Check Spex button
        if hasattr(self.parent, 'check_spex_button'):
            self.parent.check_spex_button.setEnabled(True)
        
        # Disable the Open Processing Window button when not processing
        if hasattr(self.parent, 'open_processing_button'):
            self.parent.open_processing_button.setEnabled(False)

        # Disable the Cancel Processing button in the main window
        if hasattr(self.parent, 'cancel_processing_button'):
            self.parent.cancel_processing_button.setEnabled(False)
        
        # Clean up the worker (but don't close the window)
        self.parent.worker = None

    def on_processing_started(self, message=None):
        """Handle processing start"""
        # Reset the status label
        if hasattr(self.parent, 'main_status_label'):
            self.parent.main_status_label.setText("Starting processing...")
        
        # Start showing the processing indicator
        if hasattr(self.parent, 'processing_indicator'):
            self.parent.processing_indicator.setVisible(True)

        if hasattr(self.parent, 'main_status_label'):
            self.parent.main_status_label.setVisible(True)
        
        # Enable the processing window button
        if hasattr(self.parent, 'open_processing_button'):
            self.parent.open_processing_button.setEnabled(True)
        
        # Create processing window if it doesn't exist and is requested
        if not hasattr(self.parent, 'processing_window') or self.parent.processing_window is None:
            # Create and initialize the processing window
            self.initialize_processing_window()
        else:
            # Reset the cancel button if it exists but was changed to "Close"
            if self.parent.processing_window.cancel_button.text() == "Close":
                # Change text back to "Cancel"
                self.parent.processing_window.cancel_button.setText("Cancel")
                
                # Disconnect any existing connections
                try:
                    self.parent.processing_window.cancel_button.clicked.disconnect()
                except TypeError:
                    pass  # No connections exist
                
                # Reconnect to cancel_processing
                self.parent.processing_window.cancel_button.clicked.connect(self.cancel_processing)

        # Add a divider in the console for the new processing run
        if self.parent.processing_window and hasattr(self.parent.processing_window, 'details_text'):
            self.parent.processing_window.details_text.add_processing_divider()
        
        # Update status if a message was provided
        if message and hasattr(self.parent, 'processing_window') and self.parent.processing_window:
            self.parent.processing_window.update_status(message)

        # Enable Cancel Processing button
        if hasattr(self.parent, 'cancel_processing_button'):
            self.parent.cancel_processing_button.setEnabled(True)
        
        # Disable Check Spex button
        if hasattr(self.parent, 'check_spex_button'):
            self.parent.check_spex_button.setEnabled(False)

        # Apply disabled style to Check Spex button
        if hasattr(self.parent, 'check_spex_button'):
            self.parent.check_spex_button.setStyleSheet("""
                QPushButton {
                    font-weight: bold;
                    padding: 8px 16px;
                    font-size: 14px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:disabled {
                    background-color: #A5D6A7; 
                    color: #E8F5E9;             
                    opacity: 0.8;               
                }
            """)
        
        QApplication.processEvents()
        
    def on_processing_completed(self, message):
        """Handle processing complete"""
        # Reset the status label
        if hasattr(self.parent, 'main_status_label'):
            self.parent.main_status_label.setText("Processing completed")
        
        # Hide the progress indicator
        if hasattr(self.parent, 'processing_indicator'):
            self.parent.processing_indicator.setVisible(False)

        if hasattr(self.parent, 'main_status_label'):
            self.parent.main_status_label.setVisible(False)
        
        if self.parent.processing_window:
            self.parent.processing_window.close()
            self.parent.processing_window = None  # Explicitly set to None
        
        # Re-enable both buttons
        if hasattr(self.parent, 'check_spex_button'):
            self.parent.check_spex_button.setEnabled(True)
        if hasattr(self.parent, 'open_processing_button'):
            self.parent.open_processing_button.setEnabled(False)
        
        QMessageBox.information(self.parent, "Complete", message)
    
    def on_processing_time(self, processing_time):
        """Handle processing time message from worker"""
        # Only show processing time if the worker wasn't cancelled
        if not hasattr(self.parent.worker, 'user_cancelled') or not self.parent.worker.user_cancelled:
            if self.parent.processing_window:
                self.parent.processing_window.update_status(f"Total processing time: {processing_time}")
                
            QMessageBox.information(self.parent, "Complete", f"Processing completed in {processing_time}!")

    def on_error(self, error_message):
        """Handle errors"""
        # Log the error
        logger.error(f"Processing error: {error_message}")
        
        # Reset the status label
        if hasattr(self.parent, 'main_status_label'):
            self.parent.main_status_label.setText("Error occurred")
        
        # Hide the processing indicator
        if hasattr(self.parent, 'processing_indicator'):
            self.parent.processing_indicator.setVisible(False)
        
        # Disable the Open Processing Window button
        if hasattr(self.parent, 'open_processing_button'):
            self.parent.open_processing_button.setEnabled(False)
        
        if hasattr(self.parent, 'processing_window') and self.parent.processing_window:
            self.parent.processing_window.update_status(f"ERROR: {error_message}")
            # Don't close the window automatically, let the user close it
        
        # Re-enable the Check Spex button
        if hasattr(self.parent, 'check_spex_button'):
            self.parent.check_spex_button.setEnabled(True)

        # Show error message box to the user
        QMessageBox.critical(self.parent, "Error", error_message)
        
        # Clean up worker if it exists
        if self.parent.worker:
            self.parent.worker.quit()
            self.parent.worker.wait()
            self.parent.worker.deleteLater()
            self.parent.worker = None

    def cancel_processing(self):
        """Cancel ongoing processing"""
        if hasattr(self.parent, 'worker') and self.parent.worker and self.parent.worker.isRunning():
            # Update the processing window
            if self.parent.processing_window:
                self.parent.processing_window.update_status("Cancelling processing...")
                
                # Update UI to indicate cancellation state
                self.parent.processing_window.progress_bar.setMaximum(100)
                self.parent.processing_window.progress_bar.setValue(0)
                
                # Disable the cancel button to prevent multiple clicks
                self.parent.processing_window.cancel_button.setEnabled(False)
            
            # Call the worker's cancel method
            self.parent.worker.cancel()
            
            # Hide the processing indicator
            self.parent.processing_indicator.setVisible(False)
            self.parent.main_status_label.setVisible(False)
            
            # Disable the Cancel button button
            self.parent.cancel_processing_button.setEnabled(False)
            
            # Re-enable the Check Spex button
            self.parent.check_spex_button.setEnabled(True)

    def on_processing_cancelled(self):
        """Handle processing cancellation"""
        # Reset the status label
        if hasattr(self.parent, 'main_status_label'):
            self.parent.main_status_label.setText("Processing cancelled")
        
        # Hide the processing indicator
        if hasattr(self.parent, 'processing_indicator'):
            self.parent.processing_indicator.setVisible(False)

        # Reset the status label
        if hasattr(self.parent, 'main_status_label'):
            self.parent.main_status_label.setVisible(False)
        
        # Disable the Open Processing Window button
        if hasattr(self.parent, 'open_processing_button'):
            self.parent.open_processing_button.setEnabled(False)

        # Disable the Cancel button button
        if hasattr(self.parent, 'cancel_processing_button'):
            self.parent.cancel_processing_button.setEnabled(False)
        
        # Re-enable the Check Spex button
        if hasattr(self.parent, 'check_spex_button'):
            self.parent.check_spex_button.setEnabled(True)
        
        # Notify user
        QMessageBox.information(self.parent, "Cancelled", "Processing was cancelled.")

    def on_tool_started(self, tool_name):
        """Handle tool processing start"""
        if self.parent.processing_window:
            self.parent.processing_window.update_status(f"Starting {tool_name}")
        
    def on_tool_completed(self, message):
        """Handle tool processing completion"""
        if self.parent.processing_window:
            self.parent.processing_window.update_status(message)
            # Let UI update
            QApplication.processEvents()

    def on_fixity_progress(self, message):
        """Handle fixity progress updates"""
        if self.parent.processing_window:
            self.parent.processing_window.update_detailed_status(message)

    def on_mediaconch_progress(self, message):
        """Handle mediaconch progress updates"""
        if self.parent.processing_window:
            self.parent.processing_window.update_detailed_status(message)

    def on_metadata_progress(self, message):
        """Handle metadata progress updates"""
        if self.parent.processing_window:
            self.parent.processing_window.update_detailed_status(message)

    def on_output_progress(self, message):
        """Handle output progress updates"""
        if self.parent.processing_window:
            self.parent.processing_window.update_detailed_status(message)