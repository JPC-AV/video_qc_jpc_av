from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QListWidget, QListWidgetItem, QPushButton, QAbstractItemView, QTextEdit, 
    QProgressBar, QSplitter
)
from PyQt6.QtCore import Qt, QEvent, QSize
from PyQt6.QtGui import QPalette, QFont

import os
from AV_Spex.gui.gui_theme_manager import ThemeManager, ThemeableMixin
from AV_Spex.gui.gui_processing_window_console import ConsoleTextEdit, MessageType

from AV_Spex.utils.config_manager import ConfigManager
from AV_Spex.utils.config_setup import ChecksConfig
from AV_Spex.utils.log_setup import connect_logger_to_ui

config_mgr = ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)

class ProcessingWindow(QMainWindow, ThemeableMixin):
    """Window to display processing status and progress."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Status")
        self.resize(700, 500)  # Set initial size
        self.setMinimumSize(500, 300)  # Set minimum size
        self.setWindowFlags(Qt.WindowType.Window)
        
        # Central widget and main_layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)  # Add some padding
        
        # Status label with larger font
        self.file_status_label = QLabel("No file processing yet...")
        file_font = self.file_status_label.font()
        file_font.setPointSize(10)
        self.file_status_label.setFont(file_font)
        self.file_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.file_status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimum(0)
        main_layout.addWidget(self.progress_bar)

        # Create a splitter for steps list and details text
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)  # stretch factor of 1

        # Steps list widget - shows steps that will be executed
        self.steps_list = QListWidget()
        self.steps_list.setMinimumHeight(150)
        self.steps_list.setAlternatingRowColors(True)
        self.steps_list.setMinimumWidth(150)  # Ensure minimum width
        splitter.addWidget(self.steps_list)

        # Details text - use custom ConsoleTextEdit instead of QTextEdit
        self.details_text = ConsoleTextEdit()
        splitter.addWidget(self.details_text)

        # Set initial splitter sizes
        splitter.setSizes([200, 500])  # Allocate more space to the details text

        # Detailed status
        self.detailed_status = QLabel("")
        self.detailed_status.setWordWrap(True)
        main_layout.addWidget(self.detailed_status)

        # Detail progress bar
        self.setup_details_progress_bar(main_layout)

        # Add cancel button
        self.cancel_button = QPushButton("Cancel")
        main_layout.addWidget(self.cancel_button)

        # Load the configuration and populate steps
        checks_config = config_mgr.get_config('checks', ChecksConfig)
        self.populate_steps_list()
        
        # Setup theme handling (only once)
        self.setup_theme_handling()

        # Apply initial progress bar styles
        self.apply_progress_bar_style()
        
        # Connect theme changes to progress bar styling
        self.theme_manager = ThemeManager.instance()
        self.theme_manager.themeChanged.connect(self.apply_progress_bar_style)

        # Initial welcome message
        self.details_text.append_message("Processing window initialized", MessageType.INFO)
        self.details_text.append_message("Ready to process files", MessageType.SUCCESS)

        self.logger = connect_logger_to_ui(self)

    def sizeHint(self):
        """Override size hint to provide default window size"""
        return QSize(700, 500)

    def setup_details_progress_bar(self, layout):
        """Set up the modern overlay progress bar."""
        # Create progress bar
        self.detail_progress_bar = QProgressBar()
        self.detail_progress_bar.setTextVisible(False)  # Hide default text
        self.detail_progress_bar.setMinimum(0)
        self.detail_progress_bar.setMaximum(100)
        
        # Create overlay label
        self.overlay_container = QWidget(self.detail_progress_bar)
        overlay_layout = QHBoxLayout(self.overlay_container)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        
        self.overlay_label = QLabel("0%")
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addWidget(self.overlay_label)
        
        # Set overlay to cover the progress bar
        self.overlay_container.setGeometry(self.detail_progress_bar.rect())
        self.detail_progress_bar.installEventFilter(self)
        
        # Add to layout
        layout.addWidget(self.detail_progress_bar)

    def apply_progress_bar_style(self, palette=None):
        """Apply modern overlay style to progress bar using current palette."""
        if palette is None:
            palette = self.palette()
        
        # Get colors from palette
        base_color = palette.color(QPalette.ColorRole.Base).name()
        highlight_color = palette.color(QPalette.ColorRole.Highlight).name()
        text_color = palette.color(QPalette.ColorRole.HighlightedText).name()
        
        # Style the progress bar
        self.detail_progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {base_color};
                text-align: center;
                height: 22px;
            }}
            
            QProgressBar::chunk {{
                background-color: {highlight_color};
                border-radius: 4px;
            }}
        """)
        
        # Style the overlay text
        self.overlay_label.setStyleSheet(f"""
            color: {text_color};
            font-weight: bold;
        """)

    def eventFilter(self, obj, event):
        """Ensure overlay label stays positioned correctly."""
        if obj == self.detail_progress_bar and event.type() == QEvent.Type.Resize:
            self.overlay_container.setGeometry(self.detail_progress_bar.rect())
        return super().eventFilter(obj, event)


    def populate_steps_list(self):
        """Populate the steps list with enabled checks from config."""
        try:
            # Get checks config
            checks_config = config_mgr.get_config('checks', ChecksConfig)
            if not checks_config:
                self.update_status("Warning: Could not load checks configuration")
                return

            # Fixity Steps - now using boolean checks
            if checks_config.fixity.validate_stream_fixity:
                self._add_step_item("Validate Stream Fixity")
            if checks_config.fixity.check_fixity:
                self._add_step_item("Validate Fixity")
            if checks_config.fixity.embed_stream_fixity:
                self._add_step_item("Embed Stream Fixity")
            if checks_config.fixity.output_fixity:
                self._add_step_item("Output Fixity")
            
            # MediaConch - now using boolean check
            if checks_config.tools.mediaconch.run_mediaconch:
                self._add_step_item("MediaConch Validation")
            
            # Metadata tools - note consistent naming - now using boolean checks
            if checks_config.tools.exiftool.run_tool or checks_config.tools.exiftool.check_tool:
                self._add_step_item("Exiftool")
            if checks_config.tools.ffprobe.run_tool or checks_config.tools.ffprobe.check_tool:
                self._add_step_item("FFprobe")
            if checks_config.tools.mediainfo.run_tool or checks_config.tools.mediainfo.check_tool:
                self._add_step_item("Mediainfo")
            if checks_config.tools.mediatrace.run_tool or checks_config.tools.mediatrace.check_tool:
                self._add_step_item("Mediatrace")
            
            # Output tools - now using boolean checks
            if checks_config.tools.qctools.run_tool:
                self._add_step_item("QCTools")
            if checks_config.tools.qct_parse.run_tool:
                self._add_step_item("QCT Parse")
            
            # Output files - now using boolean checks
            if checks_config.outputs.access_file:
                self._add_step_item("Generate Access File")
            if checks_config.outputs.report:
                self._add_step_item("Generate Report")
            
            # Final steps
            self._add_step_item("All Processing")
            
        except Exception as e:
            self.update_status(f"Error loading steps: {str(e)}")
    
    def _add_step_item(self, step_name):
        """Add a step item to the list."""
        item = QListWidgetItem(f"⬜ {step_name}")
        self.steps_list.addItem(item)
    
    def mark_step_complete(self, step_name):
        """Mark a step as complete in the list."""
        # Find and update the item
        found = False
        for i in range(self.steps_list.count()):
            item = self.steps_list.item(i)
            item_text = item.text()[2:]  # Remove the checkbox prefix
            
            # Check for exact match first
            if item_text == step_name:
                item.setText(f"✅ {step_name}")
                item.setFont(QFont("Arial", weight=QFont.Weight.Bold))
                found = True
                break
            # If no exact match, try case-insensitive matching
            elif item_text.lower() == step_name.lower():
                item.setText(f"✅ {item_text}")  # Keep original capitalization
                item.setFont(QFont("Arial", weight=QFont.Weight.Bold))
                found = True
                break
        
        if not found:
            self.details_text.append(f"Warning: No matching step found for '{step_name}'")

    def reset_steps_list(self):
        """Reset the steps list when processing a new file, but preserve dependency check status."""
        for i in range(self.steps_list.count()):
            item = self.steps_list.item(i)
            item_text = item.text()
        
        # Clear the list widget
        self.steps_list.clear()
        
        # Repopulate with fresh steps
        self.populate_steps_list()


    def update_detailed_status(self, message):
        """Update the detailed status message."""
        self.detailed_status.setText(message)
        QApplication.processEvents()

    def update_detail_progress(self, percentage):
        """Update the detail progress bar with the current percentage."""
        # If this is the first update (percentage very small) or a reset signal (percentage = 0),
        # we're likely starting a new process step
        if percentage <= 1:
            # Reset the progress bar
            self.detail_progress_bar.setMaximum(100)
            self.detail_progress_bar.setValue(0)
        
        # Now update with the current progress
        self.detail_progress_bar.setValue(percentage)
        
        # Update percentage label
        self.overlay_label.setText(f"{percentage}%")

    def update_status(self, message, msg_type=None):
        """
        Update the main status message and append to details text.
        Detects message type based on content and formats accordingly.
        """
        if msg_type is None:
            # Determine message type based on content
            msg_type = MessageType.NORMAL
            lowercase_msg = message.lower()
            
            # ERROR detection
            if "error" in lowercase_msg or "failed" in lowercase_msg:
                msg_type = MessageType.ERROR
            
            # WARNING detection
            elif "warning" in lowercase_msg:
                msg_type = MessageType.WARNING
            
            # COMMAND detection
            elif lowercase_msg.startswith(("finding", "checking", "executing", "running")):
                msg_type = MessageType.COMMAND
            
            # SUCCESS detection
            elif any(success_term in lowercase_msg for success_term in [
                "success", "complete", "finished", "done", "identified successfully"
            ]):
                msg_type = MessageType.SUCCESS
            
            # INFO detection
            elif any(info_term in lowercase_msg for info_term in [
                "found", "version", "dependencies", "starting", "processing"
            ]):
                msg_type = MessageType.INFO
        
        # Append the message to the console with styling
        self.details_text.append_message(message, msg_type)

    def update_file_status(self, filename, current_index=None, total_files=None):
        """Update the file status label when processing a new file."""
        if current_index is not None and total_files is not None:
            self.file_status_label.setText(f"Processing ({current_index} / {total_files}): {os.path.basename(filename)}")
        else:
            self.file_status_label.setText(f"Processing: {filename}")
        
        # Update the progress bar
        self.progress_bar.setMaximum(total_files)  # Set maximum to total files
        self.progress_bar.setValue(current_index - 1)  # Set value to index - 1

        # Reset the detail progress bar for the new file
        self.reset_progress_bars()

        # Scroll to bottom
        scrollbar = self.details_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def showEvent(self, event):
        super().showEvent(event)
        self.raise_()  # Bring window to front
        self.activateWindow()  # Activate the window

    def closeEvent(self, event):
        # Check if this is a forced close from the application quit
        if QApplication.instance().closingDown():
            # Allow actual close when application is quitting
            if hasattr(self, 'theme_manager'):
                try:
                    self.theme_manager.themeChanged.disconnect(self.apply_progress_bar_style)
                except:
                    pass  # Already disconnected
            super().closeEvent(event)
            return
            
        # Prevent the default close behavior
        event.ignore()
        
        # Hide the window instead
        self.hide()
        
        # Notify the parent window that the processing window was hidden
        parent = self.parent()
        if parent and hasattr(parent, 'on_processing_window_hidden'):
            parent.on_processing_window_hidden()

    def on_theme_changed(self, palette):
        """Handle theme changes"""
        # Apply palette to all components
        self.setPalette(palette)
        self.file_status_label.setPalette(palette)
        
        # Style the cancel button
        theme_manager = ThemeManager.instance()
        theme_manager.style_buttons(self)
        theme_manager.style_console_text(self.details_text)
        
        # Force repaint
        self.update()

    def reset_progress_bars(self):
        """Reset all progress bars when starting a new file."""
        # Reset the detail progress bar
        self.detail_progress_bar.setValue(0)
        self.detail_progress_bar.setMaximum(100)
        self.overlay_label.setText("0%")
        
        # Optionally reset the main progress bar's text
        self.progress_bar.setFormat("%p%")


class DirectoryListWidget(QListWidget):
    """Custom list widget with drag and drop support for directories."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Critical settings for drag and drop
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self.main_window = parent

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            
            for url in urls:
                path = url.toLocalFile()
                
                if os.path.isdir(path):
                    # Check for duplicates before adding
                    if path not in [self.item(i).text() for i in range(self.count())]:
                        self.addItem(path)
                        
                        # Update selected_directories if main_window is available
                        if hasattr(self.main_window, 'selected_directories'):
                            if path not in self.main_window.selected_directories:
                                self.main_window.selected_directories.append(path)
            
            event.acceptProposedAction()