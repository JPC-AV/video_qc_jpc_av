from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QScrollArea, QFileDialog, QMenuBar, QListWidget, QPushButton, QFrame, 
    QComboBox, QTabWidget, QTextEdit, QMessageBox, QDialog, QProgressBar, 
    QSizePolicy, QListView
)
from PyQt6.QtCore import Qt, QSettings, QDir, QTimer, QSize
from PyQt6.QtGui import QPixmap, QPalette

import os
import sys
from dataclasses import asdict

from ..gui.gui_processing_gui import ProcessingWindow, DirectoryListWidget
from ..gui.gui_checks_window import ChecksWindow
from ..gui.gui_theme_manager import ThemeManager, ThemeableMixin

from ..utils.config_setup import SpexConfig, ChecksConfig
from ..utils.config_manager import ConfigManager
from ..utils.config_io import ConfigIO
from ..utils import config_edit

from ..utils.log_setup import logger

from ..processing.worker_thread import ProcessingWorker
from ..processing.avspex_processor import AVSpexProcessor
from ..gui.gui_signals import ProcessingSignals

from AV_Spex import __version__
version_string = __version__

class MainWindow(QMainWindow, ThemeableMixin):
    """Main application window with tabs for configuration and settings."""
    
    def __init__(self):
        super().__init__()
        self.signals = ProcessingSignals()
        self.worker = None
        self.processing_window = None

        # Initialize collections for theme-aware components
        self.import_tab_group_boxes = [] 
        self.spex_tab_group_boxes = []
        self.checks_tab_group_boxes = []

        # Connect all signals
        self.setup_signal_connections()
        
        # Setup UI
        self.setup_ui()
        
        # Setup theme handling
        self.setup_theme_handling()
    
    def on_theme_changed(self, palette):
        """Handle theme changes across the application."""
        # Apply palette to main window
        self.setPalette(palette)
        
        # Get the theme manager
        theme_manager = ThemeManager.instance()
        
        # Update the tabs
        if hasattr(self, 'tabs'):
            self.tabs.setStyleSheet(theme_manager.get_tab_style())
        
        # Update all groupboxes in both tabs
        for group_box in self.checks_tab_group_boxes + self.spex_tab_group_boxes + self.import_tab_group_boxes:
            theme_manager.style_groupbox(group_box)
            # Style buttons inside the group box
            theme_manager.style_buttons(group_box)
        
        # Special styling for green button
        if hasattr(self, 'check_spex_button'):
            self.check_spex_button.setStyleSheet("""
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
        
        # Special styling for open processing window button
        if hasattr(self, 'open_processing_button'):
            self.open_processing_button.setStyleSheet("""
                QPushButton {
                    font-weight: bold;
                    padding: 8px 16px;
                    font-size: 14px;
                    background-color: white;
                    color: #4CAF50;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #d2ffed;
                }
                QPushButton:disabled {
                    background-color: #E8F5E9; 
                    color: #A5D6A7;             
                    opacity: 0.8;               
                }
            """)

        # Update child windows
        for child_name in ['config_widget', 'processing_window']:
            child = getattr(self, child_name, None)
            if child and hasattr(child, 'on_theme_changed'):
                child.on_theme_changed(palette)

        # Special styling for open processing window button
        if hasattr(self, 'processing_indicator'):
            self.processing_indicator.setStyleSheet("""
                QProgressBar {
                    background-color: palette(Base);
                    text-align: center;
                    padding: 1px;
                }
                QProgressBar::chunk {
                    background-color: palette(Highlight);
                }
            """)
        
        # Refresh the logo
        self._refresh_logo()
        
        # Force repaint
        self.update()

    def _refresh_logo(self):
        """Refresh the logo when theme changes"""
        # First, find and remove the existing logo layout
        for i in range(self.main_layout.count()):
            item = self.main_layout.itemAt(i)
            # Check if this layout item contains our logo (you might need to adapt this check)
            if item and item.layout() and item.layout().count() > 0:
                widget = item.layout().itemAt(0).widget()
                if isinstance(widget, QLabel) and widget.pixmap() is not None:
                    # Remove the existing logo layout
                    self._remove_layout_item(self.main_layout, i)
                    break
        
        # Now load the new theme-appropriate logo
        self._load_logo()

    def _remove_layout_item(self, layout, index):
        """Helper method to remove an item from a layout"""
        if index >= 0 and index < layout.count():
            item = layout.takeAt(index)
            if item:
                # If the item has a layout, we need to clear it first
                if item.layout():
                    while item.layout().count():
                        child = item.layout().takeAt(0)
                        if child.widget():
                            child.widget().deleteLater()
                # If the item has a widget, delete it
                if item.widget():
                    item.widget().deleteLater()
                # Delete the item itself
                del item
            

    def closeEvent(self, event):
        # Clean up theme connections
        self.cleanup_theme_handling()
        
        # Clean up child windows
        for child_name in ['config_widget', 'processing_window']:
            child = getattr(self, child_name, None)
            if child and hasattr(child, 'cleanup_theme_handling'):
                child.cleanup_theme_handling()
        
        # Stop worker if running
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        
        # Call quit handling method
        self.on_quit_clicked()
        super().closeEvent(event)

    def setup_tabs(self):
        """Set up tab styling"""
        theme_manager = ThemeManager.instance()
        
        # Create new tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(theme_manager.get_tab_style())

        self.main_layout.addWidget(self.tabs)

        self.setup_import_tab()
        self.setup_checks_tab()
        self.setup_spex_tab()

    def setup_signal_connections(self):
        """Setup all signal connections"""
        # Processing window signals
        self.signals.started.connect(self.on_processing_started)
        self.signals.completed.connect(self.on_processing_completed)
        self.signals.error.connect(self.on_error)
        self.signals.cancelled.connect(self.on_processing_cancelled)

        # Connect file_started signal to update main status label
        self.signals.file_started.connect(self.update_main_status_label)
        
        # Tool-specific signals
        self.signals.tool_started.connect(self.on_tool_started)
        self.signals.tool_completed.connect(self.on_tool_completed)
        self.signals.fixity_progress.connect(self.on_fixity_progress)
        self.signals.mediaconch_progress.connect(self.on_mediaconch_progress)
        self.signals.metadata_progress.connect(self.on_metadata_progress)
        self.signals.output_progress.connect(self.on_output_progress)

    def call_process_directories(self):
        """Initialize and start the worker thread"""
        try:
            # Create the processing window if it doesn't exist
            if not hasattr(self, 'processing_window') or self.processing_window is None:
                # Create and initialize the processing window
                self.initialize_processing_window()
            
            # Create and configure the worker
            self.worker = ProcessingWorker(self.source_directories, self.signals)
            
            # Connect worker-specific signals
            self.worker.started_processing.connect(self.on_processing_started)
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.error.connect(self.on_error)
            self.worker.processing_time.connect(self.on_processing_time)
            
            # Start the worker thread
            self.worker.start()
            
        except Exception as e:
            self.signals.error.emit(str(e))

    def initialize_processing_window(self):
        """Create and configure the processing window and connect signals"""
        self.processing_window = ProcessingWindow(self)
            
        # Connect signals to the processing window 
        self.signals.status_update.connect(self.processing_window.update_status)
        self.signals.error.connect(self.processing_window.update_status)
        self.signals.progress.connect(self.update_progress)
        self.signals.file_started.connect(self.processing_window.update_file_status)

        # reset steps list when a new file starts
        self.signals.file_started.connect(self.processing_window.reset_steps_list)

        # Progress bar signal connections
        self.signals.stream_hash_progress.connect(self.processing_window.update_detail_progress)
        self.signals.md5_progress.connect(self.processing_window.update_detail_progress)
        self.signals.access_file_progress.connect(self.processing_window.update_detail_progress)
            
        # Connect the step_completed signal
        self.signals.step_completed.connect(self.processing_window.mark_step_complete)
            
        # Connect the cancel button
        self.processing_window.cancel_button.clicked.connect(self.cancel_processing)

        # Connect open processing button
        if hasattr(self, 'open_processing_button'):
            self.open_processing_button.setText("Show Processing Window")
        
        # Show the window
        self.processing_window.show()
        self.processing_window.raise_()

    
    def on_processing_window_hidden(self):
        """Handle processing window hidden event."""
        # Update the open processing button text/functionality
        if hasattr(self, 'open_processing_button'):
            self.open_processing_button.setText("Show Processing Window")
            self.open_processing_button.setEnabled(True)

    def update_progress(self, current, total):
        """Update progress bar in the processing window."""
        if hasattr(self, 'processing_window') and self.processing_window:
            self.processing_window.progress_bar.setMaximum(total)
            self.processing_window.progress_bar.setValue(current)

    def on_worker_finished(self):
        """Handle worker thread completion."""
        # Check if this was a cancellation
        was_cancelled = hasattr(self.worker, 'user_cancelled') and self.worker.user_cancelled

        # Hide the processing indicator
        self.processing_indicator.setVisible(False)
        self.main_status_label.setVisible(False)
        
        # Update UI to indicate processing is complete
        if hasattr(self, 'processing_window') and self.processing_window:
            if not was_cancelled:
                self.processing_window.update_status("Processing completed successfully!")
                self.processing_window.progress_bar.setMaximum(100)
                self.processing_window.progress_bar.setValue(100)
            
            # Change the cancel button to a close button
            self.processing_window.cancel_button.setText("Close")
            self.processing_window.cancel_button.setEnabled(True)
            
            # Disconnect previous handler if any (use try/except in case it's not connected)
            try:
                self.processing_window.cancel_button.clicked.disconnect()
            except TypeError:
                # This catches the case where no connections exist
                pass
                
            self.processing_window.cancel_button.clicked.connect(self.processing_window.close)
        
        # Re-enable the Check Spex button
        if hasattr(self, 'check_spex_button'):
            self.check_spex_button.setEnabled(True)
        
        # Disable the Open Processing Window button when not processing
        if hasattr(self, 'open_processing_button'):
            self.open_processing_button.setEnabled(False)

        # Disable the Cancel Processing button in the main window
        if hasattr(self, 'cancel_processing_button'):
            self.cancel_processing_button.setEnabled(False)
        
        # Clean up the worker (but don't close the window)
        self.worker = None

    def on_processing_started(self, message=None):
        """Handle processing start"""
        # Reset the status label
        if hasattr(self, 'main_status_label'):
            self.main_status_label.setText("Starting processing...")
        
        # Start showing the processing indicator
        if hasattr(self, 'processing_indicator'):
            self.processing_indicator.setVisible(True)

        if hasattr(self, 'main_status_label'):
            self.main_status_label.setVisible(True)
        
        # Enable the processing window button
        if hasattr(self, 'open_processing_button'):
            self.open_processing_button.setEnabled(True)
        
        # Create processing window if it doesn't exist and is requested
        if not hasattr(self, 'processing_window') or self.processing_window is None:
            # Create and initialize the processing window
            self.initialize_processing_window()
        else:
            # Reset the cancel button if it exists but was changed to "Close"
            if self.processing_window.cancel_button.text() == "Close":
                # Change text back to "Cancel"
                self.processing_window.cancel_button.setText("Cancel")
                
                # Disconnect any existing connections
                try:
                    self.processing_window.cancel_button.clicked.disconnect()
                except TypeError:
                    pass  # No connections exist
                
                # Reconnect to cancel_processing
                self.processing_window.cancel_button.clicked.connect(self.cancel_processing)

        # Add a divider in the console for the new processing run
        if self.processing_window and hasattr(self.processing_window, 'details_text'):
            self.processing_window.details_text.add_processing_divider()
        
        # Update status if a message was provided
        if message and hasattr(self, 'processing_window') and self.processing_window:
            self.processing_window.update_status(message)

        # Enable Cancel Processing button
        if hasattr(self, 'cancel_processing_button'):
            self.cancel_processing_button.setEnabled(True)
        
        # Disable Check Spex button
        if hasattr(self, 'check_spex_button'):
            self.check_spex_button.setEnabled(False)

        # Apply disabled style to Check Spex button
        if hasattr(self, 'check_spex_button'):
            self.check_spex_button.setStyleSheet("""
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
        if hasattr(self, 'main_status_label'):
            self.main_status_label.setText("Processing completed")
        
        # Hide the progress indicator
        if hasattr(self, 'processing_indicator'):
            self.processing_indicator.setVisible(False)

        if hasattr(self, 'main_status_label'):
            self.main_status_label.setVisible(False)
        
        if self.processing_window:
            self.processing_window.close()
            self.processing_window = None  # Explicitly set to None
        
        # Re-enable both buttons
        if hasattr(self, 'check_spex_button'):
            self.check_spex_button.setEnabled(True)
        if hasattr(self, 'open_processing_button'):
            self.open_processing_button.setEnabled(False)
        
        QMessageBox.information(self, "Complete", message)
    
    def on_processing_time(self, processing_time):
        """Handle processing time message from worker"""
        # Only show processing time if the worker wasn't cancelled
        if not hasattr(self.worker, 'user_cancelled') or not self.worker.user_cancelled:
            if self.processing_window:
                self.processing_window.update_status(f"Total processing time: {processing_time}")
                
            QMessageBox.information(self, "Complete", f"Processing completed in {processing_time}!")

    def on_error(self, error_message):
        """Handle errors"""
        # Log the error
        logger.error(f"Processing error: {error_message}")
        
        # Reset the status label
        if hasattr(self, 'main_status_label'):
            self.main_status_label.setText("Error occurred")
        
        # Hide the processing indicator
        if hasattr(self, 'processing_indicator'):
            self.processing_indicator.setVisible(False)
        
        # Disable the Open Processing Window button
        if hasattr(self, 'open_processing_button'):
            self.open_processing_button.setEnabled(False)
        
        if hasattr(self, 'processing_window') and self.processing_window:
            self.processing_window.update_status(f"ERROR: {error_message}")
            # Don't close the window automatically, let the user close it
        
        # Re-enable the Check Spex button
        if hasattr(self, 'check_spex_button'):
            self.check_spex_button.setEnabled(True)

        # Show error message box to the user
        QMessageBox.critical(self, "Error", error_message)
        
        # Clean up worker if it exists
        if self.worker:
            self.worker.quit()
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None

    def cancel_processing(self):
        """Cancel ongoing processing"""
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            # Update the processing window
            if self.processing_window:
                self.processing_window.update_status("Cancelling processing...")
                
                # Update UI to indicate cancellation state
                self.processing_window.progress_bar.setMaximum(100)
                self.processing_window.progress_bar.setValue(0)
                
                # Disable the cancel button to prevent multiple clicks
                self.processing_window.cancel_button.setEnabled(False)
            
            # Call the worker's cancel method
            self.worker.cancel()
            
            # Hide the processing indicator
            self.processing_indicator.setVisible(False)
            self.main_status_label.setVisible(False)
            
            # Disable the Cancel button button
            self.cancel_processing_button.setEnabled(False)
            
            # Re-enable the Check Spex button
            self.check_spex_button.setEnabled(True)

    def on_processing_cancelled(self):
        """Handle processing cancellation"""
        # Reset the status label
        if hasattr(self, 'main_status_label'):
            self.main_status_label.setText("Processing cancelled")
        
        # Hide the processing indicator
        if hasattr(self, 'processing_indicator'):
            self.processing_indicator.setVisible(False)

        # Reset the status label
        if hasattr(self, 'main_status_label'):
            self.main_status_label.setVisible(False)
        
        # Disable the Open Processing Window button
        if hasattr(self, 'open_processing_button'):
            self.open_processing_button.setEnabled(False)

        # Disable the Cancel button button
        if hasattr(self, 'cancel_processing_button'):
            self.cancel_processing_button.setEnabled(False)
        
        # Re-enable the Check Spex button
        if hasattr(self, 'check_spex_button'):
            self.check_spex_button.setEnabled(True)
        
        # Notify user
        QMessageBox.information(self, "Cancelled", "Processing was cancelled.")

    def on_tool_started(self, tool_name):
        """Handle tool processing start"""
        if self.processing_window:
            self.processing_window.update_status(f"Starting {tool_name}")
        
    def on_tool_completed(self, message):
        """Handle tool processing completion"""
        if self.processing_window:
            self.processing_window.update_status(message)
            # Let UI update
            QApplication.processEvents()

    def on_fixity_progress(self, message):
        """Handle fixity progress updates"""
        if self.processing_window:
            self.processing_window.update_detailed_status(message)

    def on_mediaconch_progress(self, message):
        """Handle mediaconch progress updates"""
        if self.processing_window:
            self.processing_window.update_detailed_status(message)

    def on_metadata_progress(self, message):
        """Handle metadata progress updates"""
        if self.processing_window:
            self.processing_window.update_detailed_status(message)

    def on_output_progress(self, message):
        """Handle output progress updates"""
        if self.processing_window:
            self.processing_window.update_detailed_status(message)
        
    def setup_ui(self):
        """Set up the main UI components"""
        self.config_mgr = ConfigManager()
        self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
        self.spex_config = self.config_mgr.get_config('spex', SpexConfig)

        self.setMinimumSize(700, 800)

        ## self.windowFlags() retrieves the current window flags
        ## Qt.WindowType.WindowMaximizeButtonHint enables the maximize button in the window's title bar.
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        
        # Initialize settings
        self.settings = QSettings('NMAAHC', 'AVSpex')
        self.selected_directories = []
        
        self.check_spex_clicked = False
        self.setWindowTitle("AV Spex")
        
        # Set up menu bar
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        # App menu 
        self.app_menu = self.menu_bar.addMenu("AV Spex")
        self.about_action = self.app_menu.addAction("About AV Spex")
        self.about_action.triggered.connect(self.show_about_dialog)
        
        # Add a separator
        self.app_menu.addSeparator()
        
        # Add Quit action to the app menu
        self.quit_action = self.app_menu.addAction("Quit")
        self.quit_action.triggered.connect(self.on_quit_clicked)
        
        # File menu (comes after the app menu)
        self.file_menu = self.menu_bar.addMenu("File")
        self.import_action = self.file_menu.addAction("Import Directory")
        self.import_action.triggered.connect(self.import_directories)

        self.setup_main_layout()
        
        self.logo_setup()

        self.setup_tabs()

    def logo_setup(self):
        """Set up the logo display"""
        if getattr(sys, 'frozen', False):
            QTimer.singleShot(0, self._delayed_logo_setup)
        else:
            self._load_logo()

    def _delayed_logo_setup(self):
        """Delayed logo setup for frozen applications"""
        self._load_logo()

    def _load_logo(self):
        """Load and display the logo based on current theme"""
        # Get ThemeManager instance
        theme_manager = ThemeManager.instance()
        
        # Define light and dark logo paths
        light_logo_path = self.config_mgr.get_logo_path('Branding_avspex_noJPC_030725.png')
        dark_logo_path = self.config_mgr.get_logo_path('Branding_avspex_noJPC_inverted_032325.png')
        
        # Get appropriate logo for current theme
        logo_path = theme_manager.get_theme_appropriate_logo(light_logo_path, dark_logo_path)
        
        # Create and add image layout
        image_layout = self.add_image_to_top(logo_path)
        self.main_layout.insertLayout(0, image_layout)  # Insert at index 0 (top)

    def setup_import_tab(self):
        """Set up the Import tab for directory selection"""
        # Get the theme manager instance
        theme_manager = ThemeManager.instance()
        
        # Initialize the group boxes collection for this tab
        self.import_tab_group_boxes = []
        
        # Create the tab
        import_tab = QWidget()
        import_layout = QVBoxLayout(import_tab)
        self.tabs.addTab(import_tab, "Import")
        
        # Main scroll area
        main_scroll_area = QScrollArea(self)
        main_scroll_area.setWidgetResizable(True)
        main_widget = QWidget(self)
        main_scroll_area.setWidget(main_widget)
        
        # Vertical layout for the content
        vertical_layout = QVBoxLayout(main_widget)
        
        # Import directory section
        self.import_group = QGroupBox("Import Directories")
        theme_manager.style_groupbox(self.import_group, "top center")
        self.import_tab_group_boxes.append(self.import_group)
        
        import_layout_section = QVBoxLayout()
        
        # Import directory button
        import_directories_button = QPushButton("Import Directory...")
        import_directories_button.clicked.connect(self.import_directories)
        
        # Directory section
        directory_label = QLabel("Selected Directories:")
        directory_label.setStyleSheet("font-weight: bold;")
        self.directory_list = DirectoryListWidget(self)
        self.directory_list.setStyleSheet("""
            QListWidget {
                border: 1px solid gray;
                border-radius: 3px;
            }
        """)
        
        # Delete button
        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self.delete_selected_directory)
        
        # Add widgets to layout
        import_layout_section.addWidget(import_directories_button)
        import_layout_section.addWidget(directory_label)
        import_layout_section.addWidget(self.directory_list)
        import_layout_section.addWidget(delete_button)
        
        self.import_group.setLayout(import_layout_section)
        vertical_layout.addWidget(self.import_group)
        
        # Style all buttons in the section
        theme_manager.style_buttons(self.import_group)

        # Config Import section
        self.config_import_group = QGroupBox("Config Import")
        theme_manager.style_groupbox(self.config_import_group, "top center")
        self.import_tab_group_boxes.append(self.config_import_group)
        
        config_import_layout = QVBoxLayout()
        
        # Description label
        config_desc_label = QLabel("Import, export, or reset Checks/Spex configuration:")
        config_desc_label.setStyleSheet("font-weight: bold;")
        config_import_layout.addWidget(config_desc_label)
        
        # Add some spacing
        config_import_layout.addSpacing(10)
        
        # Create buttons layout
        buttons_layout = QHBoxLayout()
        
        # Import Config button
        import_config_button = QPushButton("Import Config")
        import_config_button.clicked.connect(self.import_config)
        buttons_layout.addWidget(import_config_button)
        
        # Export Config layout
        export_button_layout = QHBoxLayout()

        # Create the dropdown for export options
        self.export_combo = QComboBox()

        # Add the default placeholder option first
        self.export_combo.addItem("Export Config Type...")  
        self.export_combo.addItem("Checks Config")
        self.export_combo.addItem("Spex Config")
        self.export_combo.addItem("All Config")

        # Connect the combobox signal to your function
        self.export_combo.currentIndexChanged.connect(self.export_selected_config)

        # Add widgets to layout
        export_button_layout.addWidget(self.export_combo)
        buttons_layout.addLayout(export_button_layout)

        # Set the first item as the current item (the placeholder)
        self.export_combo.setCurrentIndex(0)
        
        # Reset to Default Config button
        reset_config_button = QPushButton("Reset to Default")
        reset_config_button.clicked.connect(self.reset_config)
        buttons_layout.addWidget(reset_config_button)
        
        config_import_layout.addLayout(buttons_layout)
        
        self.config_import_group.setLayout(config_import_layout)
        vertical_layout.addWidget(self.config_import_group)
        
        # Style all buttons in the config section
        theme_manager.style_buttons(self.config_import_group)
        
        # Add scroll area to main layout
        import_layout.addWidget(main_scroll_area)
        
        # Bottom section with processing controls
        # Similar to what you have in checks_tab but just the processing-related buttons
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 10, 0, 10)  # Add some vertical padding
        
        # Open Processing Window button
        self.open_processing_button = QPushButton("Show Processing Window")
        self.open_processing_button.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                padding: 8px 16px;
                font-size: 14px;
                background-color: white;
                color: #4CAF50;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d2ffed;
            }
            QPushButton:disabled {
                background-color: #E8F5E9; 
                color: #A5D6A7;             
                opacity: 0.8;               
            }
        """)
        self.open_processing_button.clicked.connect(self.on_open_processing_clicked)
        # Initially disable the button since no processing is running
        self.open_processing_button.setEnabled(False)
        bottom_row.addWidget(self.open_processing_button)
        
        # Cancel button
        self.cancel_processing_button = QPushButton("Cancel Processing")
        self.cancel_processing_button.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                padding: 8px 16px;
                font-size: 14px;
                background-color: #ff9999;
                color: #4d2b12;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #ff8080;
            }
            QPushButton:disabled {
                background-color: #f5e9e3; 
                color: #cd9e7f;             
                opacity: 0.8;               
            }
        """)
        self.cancel_processing_button.clicked.connect(self.cancel_processing)
        self.cancel_processing_button.setEnabled(False)
        bottom_row.addWidget(self.cancel_processing_button)
        
        # create layout for current processing
        self.now_processing_layout = QVBoxLayout()
        
        # Add a status label that shows current file being processed
        self.main_status_label = QLabel("Not processing")
        self.main_status_label.setWordWrap(True)
        self.main_status_label.setMaximumWidth(300)  # Limit width to prevent stretching
        self.main_status_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)  # Minimize height
        self.main_status_label.setVisible(False)  # initially hidden 
        self.now_processing_layout.addWidget(self.main_status_label)
        
        # Add a small indeterminate progress bar
        self.processing_indicator = QProgressBar(self)
        self.processing_indicator.setMaximumWidth(100)  # Make it small
        self.processing_indicator.setMaximumHeight(10)  # Make it shorter
        self.processing_indicator.setRange(0, 0)
        self.processing_indicator.setTextVisible(False)  # No percentage text
        self.processing_indicator.setStyleSheet("""
            QProgressBar {
                background-color: palette(Base);
                text-align: center;
                padding: 1px;
            }
        """)
        self.processing_indicator.setVisible(False)  # Initially hidden
        self.now_processing_layout.addWidget(self.processing_indicator)
        
        # Add the processing button layout to the bottom row
        # Use a stretch factor of 0 to keep it from expanding
        bottom_row.addLayout(self.now_processing_layout, 0)
        
        # Add a stretch to push the Check Spex button to the right
        bottom_row.addStretch(1)
        
        # Check Spex button
        self.check_spex_button = QPushButton("Check Spex!")
        self.check_spex_button.setStyleSheet("""
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
        self.check_spex_button.clicked.connect(self.on_check_spex_clicked)
        bottom_row.addWidget(self.check_spex_button, 0)
        
        import_layout.addLayout(bottom_row)

    def export_selected_config(self):
        selected_option = self.export_combo.currentText()
        # Skip export if the placeholder option is selected
        if selected_option == "Export Config Type...":
            return
        elif selected_option == "Checks Config":
            self.export_config('checks')
        elif selected_option == "Spex Config":
            self.export_config('spex')
        elif selected_option == "All Config":
            self.export_config('all')

    def import_config(self):
        """Import configuration from a file."""
        file_dialog = QFileDialog(self, "Import Configuration")
        file_dialog.setNameFilter("Config Files (*.json);;All Files (*)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            try:
                # Use the ConfigIO class to import config
                config_io = ConfigIO(self.config_mgr)
                config_io.import_configs(file_path)
                
                # Reload UI components to reflect new settings
                self.config_widget.load_config_values()

                # Ensure recent config ref
                self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
                self.spex_config = self.config_mgr.get_config('spex', SpexConfig)

                # Spex dropdowns
                # file name dropdown
                if self.spex_config.filename_values.Collection == "JPC":
                    self.filename_profile_dropdown.setCurrentText("JPC file names")
                elif self.spex_config.filename_values.Collection == "2012_79":
                    self.filename_profile_dropdown.setCurrentText("Bowser file names")
                
                # Signalflow profile dropdown
                # Set initial state based on config
                encoder_settings = self.spex_config.mediatrace_values.ENCODER_SETTINGS
                if isinstance(encoder_settings, dict):
                    source_vtr = encoder_settings.get('Source_VTR', [])
                else:
                    source_vtr = encoder_settings.Source_VTR
                if any("SVO5800" in vtr for vtr in source_vtr):
                    self.signalflow_profile_dropdown.setCurrentText("JPC_AV_SVHS Signal Flow")
                elif any("Sony BVH3100" in vtr for vtr in source_vtr):
                    self.signalflow_profile_dropdown.setCurrentText("BVH3100 Signal Flow")
                
                QMessageBox.information(self, "Success", f"Configuration imported successfully from {file_path}")
            except Exception as e:
                logger.error(f"Error importing config: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error importing configuration: {str(e)}")

    def export_config(self, config_type):
        """Export configuration to a file."""
        file_dialog = QFileDialog(self, "Export Configuration")
        file_dialog.setNameFilter("JSON Files (*.json);;All Files (*)")
        file_dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setDefaultSuffix("json")
        
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            try:
                # Use the ConfigIO class to export config
                config_io = ConfigIO(self.config_mgr)
                config_io.save_configs(file_path, config_type)
                
                QMessageBox.information(self, "Success", f"Configuration exported successfully to {file_path}")
            except Exception as e:
                logger.error(f"Error exporting config: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error exporting configuration: {str(e)}")

    def reset_config(self):
        """Reset configuration to default values."""
        # Ask for confirmation
        result = QMessageBox.question(
            self,
            "Confirm Reset",
            "Are you sure you want to reset all configuration to default values? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if result == QMessageBox.StandardButton.Yes:
            try:
                # Reset config by removing user config files
                user_config_dir = self.config_mgr._user_config_dir
                try:
                    # Remove the user config files
                    os.remove(os.path.join(user_config_dir, "last_used_checks_config.json"))
                    os.remove(os.path.join(user_config_dir, "last_used_spex_config.json"))
                    
                    # Reinitialize configs from default
                    self.config_mgr.configs['checks'] = ChecksConfig()
                    self.config_mgr.configs['spex'] = SpexConfig()
                    
                    # Reload UI components to reflect new settings
                    if hasattr(self, 'config_widget') and self.config_widget:
                        self.config_widget.load_config_values()
                    
                    QMessageBox.information(self, "Success", "Configuration has been reset to default values")
                except FileNotFoundError:
                    # It's okay if the files don't exist
                    QMessageBox.information(self, "Information", "Already using default configuration")
            except Exception as e:
                logger.error(f"Error resetting config: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error resetting configuration: {str(e)}")

    def setup_checks_tab(self):
        """Set up or update the Checks tab with theme-aware styling"""
        # Get the theme manager instance
        theme_manager = ThemeManager.instance()
        
        # If we're here, we're creating the tab from scratch or recreating it
        # Initialize or reset the group boxes collection
        self.checks_tab_group_boxes = []
    
        # Create the tab
        checks_tab = QWidget()
        checks_layout = QVBoxLayout(checks_tab)
        self.tabs.addTab(checks_tab, "Checks")

        # Scroll Area for Vertical Scrolling in "Checks" Tab
        main_scroll_area = QScrollArea(self)
        main_scroll_area.setWidgetResizable(True)
        main_widget = QWidget(self)
        main_scroll_area.setWidget(main_widget)

        # Vertical layout for the main content in "Checks"
        vertical_layout = QVBoxLayout(main_widget)


        # 1. Checks Profile section
        self.profile_group = QGroupBox("Checks Profiles")
        theme_manager.style_groupbox(self.profile_group, "top center")
        self.checks_tab_group_boxes.append(self.profile_group)
        
        profile_layout = QVBoxLayout()
        
        command_profile_label = QLabel("Select a Checks profile:")
        command_profile_label.setStyleSheet("font-weight: bold;")
        command_profile_desc = QLabel("Choose from a preset Checks profile to apply a set of Checks to run on your Spex")
        
        self.command_profile_dropdown = QComboBox()
        self.command_profile_dropdown.addItem("Step 1")
        self.command_profile_dropdown.addItem("Step 2")
        self.command_profile_dropdown.addItem("All Off")
        
        # Set initial dropdown state
        if self.checks_config.tools.exiftool.run_tool == "yes":
            self.command_profile_dropdown.setCurrentText("Step 1")
        elif self.checks_config.tools.exiftool.run_tool == "no":
            self.command_profile_dropdown.setCurrentText("Step 2")

        self.command_profile_dropdown.currentIndexChanged.connect(self.on_profile_selected)

        # Add widgets to layout
        profile_layout.addWidget(command_profile_label)
        profile_layout.addWidget(command_profile_desc)
        profile_layout.addWidget(self.command_profile_dropdown)
        
        self.profile_group.setLayout(profile_layout)
        vertical_layout.addWidget(self.profile_group)

        # 3. Config section
        self.config_group = QGroupBox("Checks Options")
        theme_manager.style_groupbox(self.config_group, "top center")
        self.checks_tab_group_boxes.append(self.config_group)
        
        config_layout = QVBoxLayout()
        
        config_scroll_area = QScrollArea()
        config_scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
            }
        """)
        self.config_widget = ChecksWindow(config_mgr=self.config_mgr)
        config_scroll_area.setWidgetResizable(True)
        config_scroll_area.setWidget(self.config_widget)

        # Set a minimum width for the config widget to ensure legibility
        config_scroll_area.setMinimumWidth(450)

        config_layout.addWidget(config_scroll_area)
        self.config_group.setLayout(config_layout)
        vertical_layout.addWidget(self.config_group)

        # Add scroll area to main layout
        checks_layout.addWidget(main_scroll_area)

    def update_main_status_label(self, filename, current_index=None, total_files=None):
        """Update the status label in the main window."""
        if not hasattr(self, 'main_status_label'):
            return
            
        if current_index is not None and total_files is not None:
            # Get just the basename of the file
            base_filename = os.path.basename(filename)
            self.main_status_label.setText(f"Processing ({current_index}/{total_files}): {base_filename}")
        else:
            self.main_status_label.setText(f"Processing: {filename}")
        
        # Make sure the UI updates
        QApplication.processEvents()

    def on_open_processing_clicked(self):
        """Show the processing window if it exists, or create it if it doesn't."""
        if hasattr(self, 'processing_window') and self.processing_window:
            # If the window exists but is hidden, show it
            self.processing_window.show()
            self.processing_window.raise_()
            self.processing_window.activateWindow()
        else:
            # Create processing window if it doesn't exist
            self.initialize_processing_window()
        
        # Update button text while window is visible
        if hasattr(self, 'open_processing_button'):
            self.open_processing_button.setText("Show Processing Window")

    # Add this method to handle processing window closed event
    def on_processing_window_closed(self):
        """Handle processing window closed event."""
        # Re-enable both buttons
        self.check_spex_button.setEnabled(True)
        
        self.open_processing_button.setEnabled(True)
        
        # Reset processing window reference
        self.processing_window = None

    def setup_main_layout(self):
        """Set up the main window layout structure"""
        # Create and set central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create main vertical layout
        self.main_layout = QVBoxLayout(self.central_widget)

        # Set layout margins and spacing
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

    def setup_spex_tab(self):
        """Set up or update the Spex tab with theme-aware styling"""
        # Get the theme manager instance
        theme_manager = ThemeManager.instance()
        
        # If we're here, we're creating the tab from scratch or recreating it
        # Initialize or reset the group boxes collection
        self.spex_tab_group_boxes = []
        
        # Create the tab if it doesn't exist
        if not hasattr(self, 'spex_tab') or not self.spex_tab:
            spex_tab = QWidget()
            spex_layout = QVBoxLayout(spex_tab)
            self.tabs.addTab(spex_tab, "Spex")
            self.spex_tab = spex_tab
        else:
            spex_tab = self.spex_tab
            spex_layout = spex_tab.layout()
            # Clear existing layout if needed
            while spex_layout.count():
                item = spex_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        
        # Create scroll area for vertical scrolling
        main_scroll_area = QScrollArea(self)
        main_scroll_area.setWidgetResizable(True)
        main_widget = QWidget(self)
        main_scroll_area.setWidget(main_widget)
        vertical_layout = QVBoxLayout(main_widget)
        
        # 1. Filename section
        self.filename_group = QGroupBox("Filename Values")
        theme_manager.style_groupbox(self.filename_group, "top center")
        self.spex_tab_group_boxes.append(self.filename_group)
        
        filename_layout = QVBoxLayout()
        
        # Profile dropdown
        profile_label = QLabel("Expected filename profiles:")
        profile_label.setStyleSheet("font-weight: bold;")
        self.filename_profile_dropdown = QComboBox()
        self.filename_profile_dropdown.addItem("Bowser file names")
        self.filename_profile_dropdown.addItem("JPC file names")
        
        # Set initial state based on config
        if self.spex_config.filename_values.Collection == "JPC":
            self.filename_profile_dropdown.setCurrentText("JPC file names")
        elif self.spex_config.filename_values.Collection == "2012_79":
            self.filename_profile_dropdown.setCurrentText("Bowser file names")
        
        self.filename_profile_dropdown.currentIndexChanged.connect(self.on_filename_profile_changed)
        
        # Open section button
        open_button = QPushButton("Open Section")
        open_button.clicked.connect(
            lambda: self.open_new_window('Filename Values', 'filename_values')
        )
        
        # Add widgets to layout
        filename_layout.addWidget(profile_label)
        filename_layout.addWidget(self.filename_profile_dropdown)
        filename_layout.addWidget(open_button)
        self.filename_group.setLayout(filename_layout)
        vertical_layout.addWidget(self.filename_group)
        
        # Style the button using theme manager
        theme_manager.style_buttons(self.filename_group)
        
        # 2. MediaInfo section
        self.mediainfo_group = QGroupBox("MediaInfo Values")
        theme_manager.style_groupbox(self.mediainfo_group, "top center")
        self.spex_tab_group_boxes.append(self.mediainfo_group)
        
        mediainfo_layout = QVBoxLayout()
        
        mediainfo_button = QPushButton("Open Section")
        mediainfo_button.clicked.connect(
            lambda: self.open_new_window('MediaInfo Values', 'mediainfo_values')
        )
        
        mediainfo_layout.addWidget(mediainfo_button)
        self.mediainfo_group.setLayout(mediainfo_layout)
        vertical_layout.addWidget(self.mediainfo_group)
        
        # Style the button
        theme_manager.style_buttons(self.mediainfo_group)
        
        # 3. Exiftool section
        self.exiftool_group = QGroupBox("Exiftool Values")
        theme_manager.style_groupbox(self.exiftool_group, "top center")
        self.spex_tab_group_boxes.append(self.exiftool_group)
        
        exiftool_layout = QVBoxLayout()
        
        exiftool_button = QPushButton("Open Section")
        exiftool_button.clicked.connect(
            lambda: self.open_new_window('Exiftool Values', 'exiftool_values')
        )
        
        exiftool_layout.addWidget(exiftool_button)
        self.exiftool_group.setLayout(exiftool_layout)
        vertical_layout.addWidget(self.exiftool_group)
        
        # Style the button
        theme_manager.style_buttons(self.exiftool_group)
        
        # 4. FFprobe section
        self.ffprobe_group = QGroupBox("FFprobe Values")
        theme_manager.style_groupbox(self.ffprobe_group, "top center")
        self.spex_tab_group_boxes.append(self.ffprobe_group)
        
        ffprobe_layout = QVBoxLayout()
        
        ffprobe_button = QPushButton("Open Section")
        ffprobe_button.clicked.connect(
            lambda: self.open_new_window('FFprobe Values', 'ffmpeg_values')
        )
        
        ffprobe_layout.addWidget(ffprobe_button)
        self.ffprobe_group.setLayout(ffprobe_layout)
        vertical_layout.addWidget(self.ffprobe_group)
        
        # Style the button
        theme_manager.style_buttons(self.ffprobe_group)
        
        # 5. Mediatrace section
        self.mediatrace_group = QGroupBox("Mediatrace Values")
        theme_manager.style_groupbox(self.mediatrace_group, "top center")
        self.spex_tab_group_boxes.append(self.mediatrace_group)
        
        mediatrace_layout = QVBoxLayout()
        
        # Signalflow profile dropdown
        signalflow_label = QLabel("Expected Signalflow profiles:")
        signalflow_label.setStyleSheet("font-weight: bold;")
        self.signalflow_profile_dropdown = QComboBox()
        self.signalflow_profile_dropdown.addItem("JPC_AV_SVHS Signal Flow")
        self.signalflow_profile_dropdown.addItem("BVH3100 Signal Flow")
        
        # Set initial state based on config
        encoder_settings = self.spex_config.mediatrace_values.ENCODER_SETTINGS
        if isinstance(encoder_settings, dict):
            source_vtr = encoder_settings.get('Source_VTR', [])
        else:
            source_vtr = encoder_settings.Source_VTR
            
        if any("SVO5800" in vtr for vtr in source_vtr):
            self.signalflow_profile_dropdown.setCurrentText("JPC_AV_SVHS Signal Flow")
        elif any("Sony BVH3100" in vtr for vtr in source_vtr):
            self.signalflow_profile_dropdown.setCurrentText("BVH3100 Signal Flow")
            
        self.signalflow_profile_dropdown.currentIndexChanged.connect(self.on_signalflow_profile_changed)
        
        mediatrace_button = QPushButton("Open Section")
        mediatrace_button.clicked.connect(
            lambda: self.open_new_window('Mediatrace Values', 'mediatrace_values')
        )
        
        mediatrace_layout.addWidget(signalflow_label)
        mediatrace_layout.addWidget(self.signalflow_profile_dropdown)
        mediatrace_layout.addWidget(mediatrace_button)
        self.mediatrace_group.setLayout(mediatrace_layout)
        vertical_layout.addWidget(self.mediatrace_group)
        
        # Style the button
        theme_manager.style_buttons(self.mediatrace_group)
        
        # 6. QCT section
        self.qct_group = QGroupBox("qct-parse Values")
        theme_manager.style_groupbox(self.qct_group, "top center")
        self.spex_tab_group_boxes.append(self.qct_group)
        
        qct_layout = QVBoxLayout()
        
        qct_button = QPushButton("Open Section")
        qct_button.clicked.connect(
            lambda: self.open_new_window('Expected qct-parse options', 'qct_parse_values')
        )
        
        qct_layout.addWidget(qct_button)
        self.qct_group.setLayout(qct_layout)
        vertical_layout.addWidget(self.qct_group)
        
        # Style the button
        theme_manager.style_buttons(self.qct_group)
        
        # Add scroll area to main layout
        spex_layout.addWidget(main_scroll_area)
    
    def add_image_to_top(self, logo_path):
        """Add image to the top of the main layout."""
        image_layout = QHBoxLayout()
        
        label = QLabel()
        label.setMinimumHeight(100)
        
        if logo_path and os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                # Scale pixmap to window width while keeping aspect ratio
                scaled_pixmap = pixmap.scaledToWidth(self.width(), Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
            else:
                print(f"Failed to load image at path: {logo_path}")
        else:
            print(f"Invalid logo path: {logo_path}")
        
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(label)
        return image_layout

    def import_directories(self):
        """Import directories for processing."""
        # Get the last directory from settings
        last_directory = self.settings.value('last_directory', '')
        
        # Use native file dialog
        file_dialog = QFileDialog(self, "Select Directories")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        
        # Set the starting directory to the parent of the last used directory
        if last_directory:
            dir_info = QDir(last_directory)
            if dir_info.cdUp():  # Move up to parent directory
                parent_dir = dir_info.absolutePath()
                file_dialog.setDirectory(parent_dir)
        
        # Try to enable multiple directory selection with the native dialog
        file_dialog.setOption(QFileDialog.Option.ReadOnly, False)

        if file_dialog.exec():
            directories = file_dialog.selectedFiles()  # Get selected directories
            
            # Save the last used directory
            if directories:
                self.settings.setValue('last_directory', directories[0])
                self.settings.sync()  # Ensure settings are saved
            
            for directory in directories:
                if directory not in self.selected_directories:
                    self.selected_directories.append(directory)
                    self.directory_list.addItem(directory)

    def update_selected_directories(self):
        """Update source_directories from the QListWidget."""
        self.source_directories = [self.directory_list.item(i).text() for i in range(self.directory_list.count())]

    def get_source_directories(self):
        """Return the selected directories if Check Spex was clicked."""
        return self.selected_directories if self.check_spex_clicked else None
    
    def delete_selected_directory(self):
        """Delete the selected directory from the list widget and the selected_directories list."""
        # Get the selected items
        selected_items = self.directory_list.selectedItems()
        
        if not selected_items:
            return  # No item selected, do nothing
        
        # Remove each selected item from both the QListWidget and selected_directories list
        for item in selected_items:
            # Remove from the selected_directories list
            directory = item.text()
            if directory in self.selected_directories:
                self.selected_directories.remove(directory)
            
            # Remove from the QListWidget
            self.directory_list.takeItem(self.directory_list.row(item))

    def on_check_spex_clicked(self):
        """Handle the Check Spex button click."""
        self.update_selected_directories()
        self.check_spex_clicked = True  # Mark that the button was clicked
        self.config_mgr.save_last_used_config('checks')
        self.config_mgr.save_last_used_config('spex')
        # Make sure the processing window is visible before starting the process
        if hasattr(self, 'processing_window') and self.processing_window:
            # If it exists but might be hidden, show it
            self.processing_window.show()
            self.processing_window.raise_()
            self.processing_window.activateWindow()
        
        # Call worker thread
        self.call_process_directories()

    def on_profile_selected(self, index):
        """Handle profile selection from dropdown."""
        selected_profile = self.command_profile_dropdown.currentText()
        if selected_profile == "Step 1":
            profile = config_edit.profile_step1
        elif selected_profile == "Step 2":
            profile = config_edit.profile_step2
        elif selected_profile == "All Off":
            profile = config_edit.profile_allOff
        try:
            # Call the backend function to apply the selected profile
            config_edit.apply_profile(profile)
            logger.debug(f"Profile '{selected_profile}' applied successfully.")
            self.config_mgr.save_last_used_config('checks')
        except ValueError as e:
            logger.critical(f"Error: {e}")

        self.config_widget.load_config_values()

    def on_filename_profile_changed(self, index):
        """Handle filename profile selection change."""
        selected_option = self.filename_profile_dropdown.itemText(index)
        updates = {}
        
        if selected_option == "JPC file names":
            updates = {
                "filename_values": {
                    "Collection": "JPC",
                    "MediaType": "AV",
                    "ObjectID": r"\d{5}",
                    "DigitalGeneration": None,
                    "FileExtension": "mkv"
                }
            }
        elif selected_option == "Bowser file names":
            updates = {
                "filename_values": {
                    "Collection": "2012_79",
                    "MediaType": "2",
                    "ObjectID": r"\d{3}_\d{1}[a-zA-Z]",
                    "DigitalGeneration": "PM",
                    "FileExtension": "mkv"
                }
            }
        
        self.config_mgr.update_config('spex', updates)
        self.config_mgr.save_last_used_config('spex')

    def on_signalflow_profile_changed(self, index):
        """Handle signal flow profile selection change."""
        selected_option = self.signalflow_profile_dropdown.itemText(index)
        logger.debug(f"Selected signal flow profile: {selected_option}")

        if selected_option == "JPC_AV_SVHS Signal Flow":
            sn_config_changes = config_edit.JPC_AV_SVHS
        elif selected_option == "BVH3100 Signal Flow":
            sn_config_changes = config_edit.BVH3100
        else:
            logger.error("Signal flow identifier not recognized, config not updated")
            return

        if sn_config_changes:
            config_edit.apply_signalflow_profile(sn_config_changes)

    def prepare_config_data(self, title, config_data):
        """
        Prepare configuration data for display in a new window.
        Handles conversion from dataclasses to dictionaries based on data type.
        
        Args:
            title (str): The title/type of configuration being displayed
            config_data: The configuration data object (dataclass or dictionary)
            
        Returns:
            dict: A dictionary representation of the configuration data
        """
        
        # If the input is already a dictionary, return it as is
        if isinstance(config_data, dict):
            return config_data
        
        # Handle special cases based on title
        if title == 'MediaInfo Values':
            return {
                'expected_general': config_data['expected_general'],
                'expected_video': config_data['expected_video'], 
                'expected_audio': config_data['expected_audio']
            }
        elif title == 'FFprobe Values':
            return {
                'video_stream': config_data['video_stream'],
                'audio_stream': config_data['audio_stream'],
                'format': config_data['format']
            }
        
        # For standard dataclasses, convert to dict
        from dataclasses import asdict
        return asdict(config_data)


    def open_new_window(self, title, config_attribute_name):
        """Open a new window to display configuration details."""
        self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
        self.spex_config = self.config_mgr.get_config('spex', SpexConfig)

        # Get the fresh config data using the attribute name
        config_data = getattr(self.spex_config, config_attribute_name)
        
        # Prepare the data using the helper function
        nested_dict = self.prepare_config_data(title, config_data)
        
        # Convert the dictionary to a string representation
        content_text = self.dict_to_string(nested_dict)

        # Create and configure the window
        self.new_window = QWidget()
        self.new_window.setWindowTitle(title)
        self.new_window.setLayout(QVBoxLayout())
        
        scroll_area = QScrollArea(self.new_window)
        scroll_area.setWidgetResizable(True)
        
        content_widget = QTextEdit()
        content_widget.setPlainText(content_text)
        content_widget.setReadOnly(True)
        content_widget.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        
        # Remove the hardcoded background color and use system palette instead
        content_widget.setStyleSheet("padding: 5px;")
        
        # Explicitly set the text color to follow system palette
        palette = content_widget.palette()
        palette.setColor(QPalette.ColorRole.Base, palette.color(QPalette.ColorRole.Window))
        palette.setColor(QPalette.ColorRole.Text, palette.color(QPalette.ColorRole.WindowText))
        content_widget.setPalette(palette)
        
        scroll_area.setWidget(content_widget)
        
        self.new_window.layout().addWidget(scroll_area)
        self.new_window.resize(600, 400)
        self.new_window.show()

    def dict_to_string(self, content_dict, indent_level=0):
        """Convert a dictionary to a string representation for display.
        
        Handles nested dictionaries and lists with proper formatting and indentation.
        """
        content_lines = []
        indent = "  " * indent_level  # Two spaces per indent level

        for key, value in content_dict.items():
            if isinstance(value, dict):  # If the value is a nested dictionary
                content_lines.append(f"{indent}{key}:")
                # Recursively process the nested dictionary
                content_lines.append(self.dict_to_string(value, indent_level + 1))
            elif isinstance(value, list):  # If the value is a list
                content_lines.append(f"{indent}{key}:")
                # Add each list item on a new line with additional indentation
                for item in value:
                    content_lines.append(f"{indent}  - {item}")
            else:  # For all other types (e.g., strings, numbers)
                content_lines.append(f"{indent}{key}: {value}")

        return "\n".join(content_lines)
    
    def on_quit_clicked(self):
        """Handle the 'Quit' button click."""
        self.selected_directories = None  # Clear any selections
        self.check_spex_clicked = False  # Ensure the flag is reset
        self.config_mgr.save_last_used_config('checks')
        self.config_mgr.save_last_used_config('spex')
        self.close()  # Close the GUI

    def show_about_dialog(self):
        """Show the About dialog with version information and logo."""
        # Create a dialog
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("About AV Spex")
        about_dialog.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(about_dialog)
        
        # Add logo
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Get the logo path
        logo_path = self.config_mgr.get_logo_path('av_spex_the_logo.png')
        
        if logo_path and os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                # Scale pixmap to a reasonable size for the dialog
                scaled_pixmap = pixmap.scaled(QSize(300, 150), 
                                            Qt.AspectRatioMode.KeepAspectRatio, 
                                            Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(scaled_pixmap)
        
        layout.addWidget(logo_label)
        
        # Add version information
        version_label = QLabel(f"Version: {version_string}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
        layout.addWidget(version_label)
        
        # Add additional information if needed
        info_label = QLabel("AV Spex - Audio/Video Specification Checker")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        copyright_label = QLabel("GNU General Public License v3.0")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copyright_label)
        
        # Show the dialog
        about_dialog.exec()