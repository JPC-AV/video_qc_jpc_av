from PyQt6.QtWidgets import QApplication
import os

class MainWindowSignals:
    """Signal connections and handlers for the main window"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def setup_signal_connections(self):
        """Setup all signal connections"""
        # Processing window signals
        self.parent.signals.started.connect(self.parent.processing.on_processing_started)
        self.parent.signals.completed.connect(self.parent.processing.on_processing_completed)
        self.parent.signals.error.connect(self.parent.processing.on_error)
        self.parent.signals.cancelled.connect(self.parent.processing.on_processing_cancelled)

        # Connect file_started signal to update main status label
        self.parent.signals.file_started.connect(self.update_main_status_label)
        
        # Tool-specific signals
        self.parent.signals.tool_started.connect(self.parent.processing.on_tool_started)
        self.parent.signals.tool_completed.connect(self.parent.processing.on_tool_completed)
        self.parent.signals.fixity_progress.connect(self.parent.processing.on_fixity_progress)
        self.parent.signals.mediaconch_progress.connect(self.parent.processing.on_mediaconch_progress)
        self.parent.signals.metadata_progress.connect(self.parent.processing.on_metadata_progress)
        self.parent.signals.output_progress.connect(self.parent.processing.on_output_progress)
    
    def on_processing_window_hidden(self):
        """Handle processing window hidden event."""
        # Update the open processing button text/functionality
        if hasattr(self.parent, 'open_processing_button'):
            self.parent.open_processing_button.setText("Show Processing Window")
            self.parent.open_processing_button.setEnabled(True)
    
    def on_processing_window_closed(self):
        """Handle processing window closed event."""
        # Re-enable both buttons
        self.parent.check_spex_button.setEnabled(True)
        
        self.parent.open_processing_button.setEnabled(True)
        
        # Reset processing window reference
        self.parent.processing_window = None
    
    def on_open_processing_clicked(self):
        """Show the processing window if it exists, or create it if it doesn't."""
        if hasattr(self.parent, 'processing_window') and self.parent.processing_window:
            # If the window exists but is hidden, show it
            self.parent.processing_window.show()
            self.parent.processing_window.raise_()
            self.parent.processing_window.activateWindow()
        else:
            # Create processing window if it doesn't exist
            self.parent.processing.initialize_processing_window()
        
        # Update button text while window is visible
        if hasattr(self.parent, 'open_processing_button'):
            self.parent.open_processing_button.setText("Show Processing Window")
    
    def on_quit_clicked(self):
        """Handle the 'Quit' button click."""
        self.parent.selected_directories = None  # Clear any selections
        self.parent.check_spex_clicked = False  # Ensure the flag is reset
        self.parent.config_mgr.save_last_used_config('checks')
        self.parent.config_mgr.save_last_used_config('spex')
        self.parent.close()  # Close the GUI
    
    def update_main_status_label(self, filename, current_index=None, total_files=None):
        """Update the status label in the main window."""
        if not hasattr(self.parent, 'main_status_label'):
            return
            
        if current_index is not None and total_files is not None:
            # Get just the basename of the file
            base_filename = os.path.basename(filename)
            self.parent.main_status_label.setText(f"Processing ({current_index}/{total_files}): {base_filename}")
        else:
            self.parent.main_status_label.setText(f"Processing: {filename}")
        
        # Make sure the UI updates
        QApplication.processEvents()