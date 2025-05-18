from PyQt6.QtWidgets import QApplication
import os
from AV_Spex.utils.config_manager import ConfigManager

config_mgr = ConfigManager()

class MainWindowSignals:
    """Signal connections and handlers for the main window"""
    
    def __init__(self, main_window):
        self.main_window = main_window
    
    def setup_signal_connections(self):
        """Setup all signal connections"""
        # Processing window signals
        self.main_window.signals.started.connect(self.main_window.processing.on_processing_started)
        self.main_window.signals.completed.connect(self.main_window.processing.on_processing_completed)
        self.main_window.signals.error.connect(self.main_window.processing.on_error)
        self.main_window.signals.cancelled.connect(self.main_window.processing.on_processing_cancelled)

        # Connect file_started signal to update main status label
        self.main_window.signals.file_started.connect(self.update_main_status_label)
        
        # Tool-specific signals
        self.main_window.signals.tool_started.connect(self.main_window.processing.on_tool_started)
        self.main_window.signals.tool_completed.connect(self.main_window.processing.on_tool_completed)
        self.main_window.signals.fixity_progress.connect(self.main_window.processing.on_fixity_progress)
        self.main_window.signals.mediaconch_progress.connect(self.main_window.processing.on_mediaconch_progress)
        self.main_window.signals.metadata_progress.connect(self.main_window.processing.on_metadata_progress)
        self.main_window.signals.output_progress.connect(self.main_window.processing.on_output_progress)
    
    def on_processing_window_hidden(self):
        """Handle processing window hidden event."""
        # Update the open processing button text/functionality
        if hasattr(self.main_window, 'open_processing_button'):
            self.main_window.open_processing_button.setText("Show Processing Window")
            self.main_window.open_processing_button.setEnabled(True)
    
    def on_processing_window_closed(self):
        """Handle processing window closed event."""
        # Re-enable both buttons
        self.main_window.check_spex_button.setEnabled(True)
        
        self.main_window.open_processing_button.setEnabled(True)
        
        # Reset processing window reference
        self.main_window.processing_window = None
    
    def on_open_processing_clicked(self):
        """Show the processing window if it exists, or create it if it doesn't."""
        if hasattr(self.main_window, 'processing_window') and self.main_window.processing_window:
            # If the window exists but is hidden, show it
            self.main_window.processing_window.show()
            self.main_window.processing_window.raise_()
            self.main_window.processing_window.activateWindow()
        else:
            # Create processing window if it doesn't exist
            self.main_window.processing.initialize_processing_window()
        
        # Update button text while window is visible
        if hasattr(self.main_window, 'open_processing_button'):
            self.main_window.open_processing_button.setText("Show Processing Window")
    
    def on_quit_clicked(self):
        """Handle the 'Quit' button click."""
        self.main_window.selected_directories = None  # Clear any selections
        self.main_window.check_spex_clicked = False  # Ensure the flag is reset
         # Only save configs that are actually in the ConfigManager._configs dictionary
        if 'checks' in config_mgr._configs:
            config_mgr.save_config('checks', is_last_used=True)
        
        if 'spex' in config_mgr._configs:
            config_mgr.save_config('spex', is_last_used=True)

    
    def update_main_status_label(self, filename, current_index=None, total_files=None):
        """Update the status label in the main window."""
        if not hasattr(self.main_window, 'main_status_label'):
            return
            
        if current_index is not None and total_files is not None:
            # Get just the basename of the file
            base_filename = os.path.basename(filename)
            self.main_window.main_status_label.setText(f"Processing ({current_index}/{total_files}): {base_filename}")
        else:
            self.main_window.main_status_label.setText(f"Processing: {filename}")
        
        # Make sure the UI updates
        QApplication.processEvents()