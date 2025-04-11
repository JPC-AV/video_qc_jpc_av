from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QLineEdit, QLabel, 
    QScrollArea, QFileDialog, QMenuBar, QListWidget, QPushButton, QFrame, QComboBox, QTabWidget,
    QTextEdit, QAbstractItemView, QInputDialog, QMessageBox, QProgressBar, QDialog
)
from PyQt6.QtCore import Qt, QSettings, QDir, QTimer
from PyQt6.QtGui import QPixmap, QPalette

import os
import sys
from dataclasses import asdict

from ..utils.setup_config import SpexConfig, ChecksConfig, FilenameConfig, FilenameProfile, FilenameSection
from ..utils.config_manager import ConfigManager
from ..utils.log_setup import logger
from ..utils import edit_config

from ..processing.processing_mgmt import setup_mediaconch_policy
from ..processing.worker_thread import ProcessingWorker

from ..processing.avspex_processor import AVSpexProcessor
from ..utils.signals import ProcessingSignals


class CustomFilenameDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pattern = None
        self.setWindowTitle("Custom Filename Pattern")
        self.setModal(True)

        # Set minimum size for the dialog
        self.setMinimumSize(500, 600)  # Width: 500px, Height: 600px
        
        # Initialize layout
        layout = QVBoxLayout()
        layout.setSpacing(10)  # Reduce overall vertical spacing
        
        # Add description
        description = QLabel("Define your filename pattern using 1-8 sections separated by underscores.")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Scrollable area for sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.sections_layout = QVBoxLayout(scroll_widget)
        self.sections_layout.setSpacing(5)  # Reduce spacing between sections
        self.sections_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        scroll.setWidget(scroll_widget)

        # Set a reasonable fixed height for the scroll area
        scroll.setMinimumHeight(300)  # Ensure scroll area is tall enough
        
        # Initial section
        self.sections = []
        self.add_section()
        
        # Buttons for managing sections
        section_buttons_layout = QHBoxLayout()
        add_button = QPushButton("Add Section")
        add_button.clicked.connect(self.add_section)
        remove_button = QPushButton("Remove Last Section")
        remove_button.clicked.connect(self.remove_section)
        section_buttons_layout.addWidget(add_button)
        section_buttons_layout.addWidget(remove_button)
        
        # File Extension input
        extension_layout = QHBoxLayout()
        extension_layout.addWidget(QLabel("File Extension:"))
        self.extension_input = QLineEdit()
        self.extension_input.setText("mkv")
        extension_layout.addWidget(self.extension_input)
        
        # Preview section
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Preview:"))
        self.preview_label = QLabel()
        preview_layout.addWidget(self.preview_label)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save Pattern")
        save_button.clicked.connect(self.on_save_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        # Add all layouts to main layout
        layout.addWidget(scroll)
        layout.addLayout(section_buttons_layout)
        layout.addLayout(extension_layout)
        layout.addLayout(preview_layout)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.update_preview()
        
    def add_section(self):
        """Add a new filename section widget"""
        if len(self.sections) >= 8:
            QMessageBox.warning(self, "Warning", "Maximum 8 sections allowed")
            return
            
        section_widget = QWidget()
        section_layout = QHBoxLayout(section_widget)
        section_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins around section
        section_layout.setSpacing(5)  # Reduce spacing between elements
        
        # Section number label
        section_num = len(self.sections) + 1
        section_label = QLabel(f"Section {section_num}:")
        section_layout.addWidget(section_label)
        
        # Section type combo box
        type_combo = QComboBox()
        type_combo.addItems(["Literal", "Wildcard", "Regex"])
        section_layout.addWidget(type_combo)
        
        # Value input
        value_input = QLineEdit()
        section_layout.addWidget(value_input)
        
        # Help button with tooltip
        help_button = QPushButton("?")
        help_button.setFixedSize(20, 20)
        help_text = {
            0: "Literal: Exact text match (e.g., 'JPC')",
            1: "Wildcard: Use # for digits, @ for letters, * for either\n" +
               "Examples:\n" +
               "#### = exactly 4 digits\n" +
               "@@ = exactly 2 letters\n" +
               "*** = 3 characters (letters or numbers)",
            2: "Regex: Regular expression pattern (e.g., '\\d{3}')"
        }
        help_button.clicked.connect(lambda: QMessageBox.information(self, "Help", help_text[type_combo.currentIndex()]))
        section_layout.addWidget(help_button)
        
        # Store section controls
        section = {
            'widget': section_widget,
            'type_combo': type_combo,
            'value_input': value_input
        }
        self.sections.append(section)
        
        # Connect signals for preview updates
        type_combo.currentIndexChanged.connect(self.update_preview)
        value_input.textChanged.connect(self.update_preview)
        
        self.sections_layout.addWidget(section_widget)
        self.update_preview()
        
    def remove_section(self):
        """Remove the last filename section"""
        if self.sections:
            section = self.sections.pop()
            section['widget'].deleteLater()
            self.update_preview()
        if len(self.sections) < 1:
            self.add_section()  # Ensure at least one section exists
            
    def update_preview(self):
        """Update the filename preview"""
        parts = []
        for section in self.sections:
            value = section['value_input'].text()
            if value:
                parts.append(value)
                
        if parts:
            preview = "_".join(parts) + "." + self.extension_input.text()
            self.preview_label.setText(preview)
            
    def get_filename_pattern(self):
        """Get the filename pattern as a FilenameProfile dataclass"""
        if not self.sections:
            QMessageBox.warning(self, "Validation Error", "At least one section is required.")
            return None
            
        if not all(section['value_input'].text() for section in self.sections):
            QMessageBox.warning(self, "Validation Error", "All sections must have a value.")
            return None
            
        if not self.extension_input.text():
            QMessageBox.warning(self, "Validation Error", "File extension is required.")
            return None
            
        fn_sections = {}
        for i, section in enumerate(self.sections, 1):
            section_type = section['type_combo'].currentText().lower()
            value = section['value_input'].text()
            
            # Create a FilenameSection instance for each section
            fn_sections[f"section{i}"] = FilenameSection(
                value=value,
                section_type=section_type
            )
            
        # Create and return a FilenameProfile instance
        return FilenameProfile(
            fn_sections=fn_sections,
            FileExtension=self.extension_input.text()
        )

    def on_save_clicked(self):
        """Handle save button click"""
        pattern = self.get_filename_pattern()
        if pattern:
            try:
                edit_config.apply_filename_profile(pattern)
                self.pattern = pattern
                self.accept()  # This will trigger QDialog.accepted
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to apply filename pattern: {str(e)}")

    def get_pattern(self):
        """Return the stored pattern"""
        return self.pattern
            
    def load_existing_pattern(self, pattern):
        """Load an existing filename pattern into the dialog"""
        if not pattern or 'fn_sections' not in pattern:
            return
            
        # Clear existing sections
        while self.sections:
            self.remove_section()
            
        # Load sections
        for section_key, section_data in pattern['fn_sections'].items():
            self.add_section()
            section = self.sections[-1]
            
            # Set section type
            type_index = {
                'literal': 0,
                'wildcard': 1,
                'regex': 2
            }.get(section_data['section_type'].lower(), 0)
            section['type_combo'].setCurrentIndex(type_index)
            
            # Set value
            section['value_input'].setText(section_data['value'])
                
        # Load extension
        if 'FileExtension' in pattern:
            self.extension_input.setText(pattern['FileExtension'])
            
        self.update_preview()


class ProcessingWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Status")
        self.resize(500, 200)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Status label with larger font
        self.status_label = QLabel("Initializing...")
        font = self.status_label.font()
        font.setPointSize(12)
        self.status_label.setFont(font)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # This makes it into a "bouncing ball" progress bar
        layout.addWidget(self.progress_bar)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(100)
        layout.addWidget(self.details_text)

        # Add cancel button
        self.cancel_button = QPushButton("Cancel")
        layout.addWidget(self.cancel_button)

        # Center the window on screen
        self._center_on_screen()  # Changed to use the defined method
        
        # Force window to update
        self.update()
        self.repaint()

        self.detailed_status = QLabel("")
        self.detailed_status.setWordWrap(True)
        layout.addWidget(self.detailed_status)

    def update_detailed_status(self, message):
        self.detailed_status.setText(message)
        QApplication.processEvents()

    def update_status(self, message):
        self.status_label.setText(message)
        self.details_text.append(message)
        # Scroll to bottom
        scrollbar = self.details_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _center_on_screen(self):
        """Centers the window on the screen"""
        screen = QApplication.primaryScreen().geometry()
        window_size = self.geometry()
        x = (screen.width() - window_size.width()) // 2
        y = (screen.height() - window_size.height()) // 2
        self.move(x, y)

    def showEvent(self, event):
        super().showEvent(event)
        self.raise_()  # Bring window to front
        self.activateWindow()  # Activate the window

    def closeEvent(self, event):
        # logger.debug("ProcessingWindow close event triggered")  # Debug
        super().closeEvent(event)


class DirectoryListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Critical settings for drag and drop
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self.main_window = parent

    def dragEnterEvent(self, event):
        #print("Drag Enter Event Triggered")
        if event.mimeData().hasUrls():
         #   print("URLs Detected in Drag Event")
            event.acceptProposedAction()
        else:
          #  print("No URLs in Drag Event")
            event.ignore()

    def dragMoveEvent(self, event):
        #print("Drag Move Event Triggered")
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


class ConfigWindow(QWidget):
    def __init__(self, config_mgr=None):
        super().__init__()
        self.config_mgr = config_mgr or ConfigManager()
        self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
        self.setup_ui()
        self.load_config_values()

    def setup_ui(self):
        """Create fixed layout structure"""
        main_layout = QVBoxLayout(self)

        self.setup_outputs_section(main_layout)

        self.setup_fixity_section(main_layout)

        self.setup_tools_section(main_layout)

        self.connect_signals()
        
    # Outputs Section
    def setup_outputs_section(self, main_layout):
        outputs_group = QGroupBox("Outputs")
        outputs_layout = QVBoxLayout()
        
        # Create widgets
        self.access_file_cb = QCheckBox("access_file")
        self.report_cb = QCheckBox("report")
        self.qctools_ext_label = QLabel("qctools_ext")
        self.qctools_ext_input = QLineEdit()
        
        # Add to layout
        outputs_layout.addWidget(self.access_file_cb)
        outputs_layout.addWidget(self.report_cb)
        outputs_layout.addWidget(self.qctools_ext_label)
        outputs_layout.addWidget(self.qctools_ext_input)
        
        outputs_group.setLayout(outputs_layout)
        main_layout.addWidget(outputs_group)
    
    # Fixity Section
    def setup_fixity_section(self, main_layout):
        fixity_group = QGroupBox("Fixity")
        fixity_layout = QVBoxLayout()
        
        # Create checkboxes
        self.output_fixity_cb = QCheckBox("Output fixity (to .txt and .md5 files)")
        self.check_fixity_cb = QCheckBox("Validate fixity")
        self.embed_stream_cb = QCheckBox("Embed Stream fixity")
        self.validate_stream_cb = QCheckBox("Validate Stream fixity")
        self.overwrite_stream_cb = QCheckBox("Overwrite Stream fixity")
        
        # Add to layout
        fixity_layout.addWidget(self.output_fixity_cb)
        fixity_layout.addWidget(self.check_fixity_cb)
        fixity_layout.addWidget(self.embed_stream_cb)
        fixity_layout.addWidget(self.validate_stream_cb)
        fixity_layout.addWidget(self.overwrite_stream_cb)
        
        fixity_group.setLayout(fixity_layout)
        main_layout.addWidget(fixity_group)
        
    # Tools Section
    def setup_tools_section(self, main_layout):
        tools_group = QGroupBox("Tools")
        tools_layout = QVBoxLayout()
        
        # Setup basic tools
        basic_tools = ['exiftool', 'ffprobe', 'mediainfo', 'mediatrace', 'qctools']
        self.tool_widgets = {}
        
        for tool in basic_tools:
            tool_group = QGroupBox(tool)
            tool_layout = QVBoxLayout()
            
            if tool == 'qctools':
                run_cb = QCheckBox("Run Tool")
                self.tool_widgets[tool] = {'run': run_cb}
                tool_layout.addWidget(run_cb)
            else:
                check_cb = QCheckBox("Check Tool")
                run_cb = QCheckBox("Run Tool")
                self.tool_widgets[tool] = {'check': check_cb, 'run': run_cb}
                tool_layout.addWidget(check_cb)
                tool_layout.addWidget(run_cb)
            
            tool_group.setLayout(tool_layout)
            tools_layout.addWidget(tool_group)

        # MediaConch section
        mediaconch_group = QGroupBox("Mediaconch")
        mediaconch_layout = QVBoxLayout()
        
        self.run_mediaconch_cb = QCheckBox("Run Mediaconch")
        
        # Policy selection
        policy_container = QWidget()
        policy_layout = QVBoxLayout(policy_container)
        
        # Current policy display
        current_policy_widget = QWidget()
        current_policy_layout = QHBoxLayout(current_policy_widget)
        current_policy_layout.setContentsMargins(0, 0, 0, 0)
        
        self.policy_label = QLabel("Current policy: ")
        self.current_policy_display = QLabel()
        self.current_policy_display.setStyleSheet("font-weight: bold;")
        
        current_policy_layout.addWidget(self.policy_label)
        current_policy_layout.addWidget(self.current_policy_display)
        current_policy_layout.addStretch()
        
        self.policy_combo = QComboBox()
        self.import_policy_btn = QPushButton("Import New MediaConch Policy")
        
        policy_layout.addWidget(current_policy_widget)
        policy_layout.addWidget(QLabel("Available policies:"))
        policy_layout.addWidget(self.policy_combo)
        policy_layout.addWidget(self.import_policy_btn)
        
        mediaconch_layout.addWidget(self.run_mediaconch_cb)
        mediaconch_layout.addWidget(policy_container)
        mediaconch_group.setLayout(mediaconch_layout)
        tools_layout.addWidget(mediaconch_group)

        # QCT Parse section
        qct_group = QGroupBox("qct-parse")
        qct_layout = QVBoxLayout()
        
        # Checkboxes
        self.run_qctparse_cb = QCheckBox("Run Tool")
        self.bars_detection_cb = QCheckBox("barsDetection")
        self.evaluate_bars_cb = QCheckBox("evaluateBars")
        self.thumb_export_cb = QCheckBox("thumbExport")
        
        # Content Filter
        content_filter_label = QLabel("contentFilter")
        self.content_filter_combo = QComboBox()
        self.content_filter_combo.addItem("Select options...")
        self.content_filter_combo.addItems(["allBlack", "static"])
        
        # Profile
        profile_label = QLabel("profile")
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("Select options...")
        self.profile_combo.addItems(["default", "highTolerance", "midTolerance", "lowTolerance"])
        
        # Tagname
        tagname_label = QLabel("tagname")
        self.tagname_input = QLineEdit()
        self.tagname_input.setPlaceholderText("None")
        
        # Add widgets to layout
        qct_layout.addWidget(self.run_qctparse_cb)
        qct_layout.addWidget(self.bars_detection_cb)
        qct_layout.addWidget(self.evaluate_bars_cb)
        qct_layout.addWidget(self.thumb_export_cb)
        qct_layout.addWidget(content_filter_label)
        qct_layout.addWidget(self.content_filter_combo)
        qct_layout.addWidget(profile_label)
        qct_layout.addWidget(self.profile_combo)
        qct_layout.addWidget(tagname_label)
        qct_layout.addWidget(self.tagname_input)
        
        qct_group.setLayout(qct_layout)
        tools_layout.addWidget(qct_group)
        
        tools_group.setLayout(tools_layout)
        main_layout.addWidget(tools_group)

    def connect_signals(self):
        """Connect all widget signals to their handlers"""
        # Outputs section
        self.access_file_cb.stateChanged.connect(
            lambda state: self.on_checkbox_changed(state, ['outputs', 'access_file'])
        )
        self.report_cb.stateChanged.connect(
            lambda state: self.on_checkbox_changed(state, ['outputs', 'report'])
        )
        self.qctools_ext_input.textChanged.connect(
            lambda text: self.on_text_changed(['outputs', 'qctools_ext'], text)
        )
        
        # Fixity section
        fixity_checkboxes = {
            self.check_fixity_cb: 'check_fixity',
            self.validate_stream_cb: 'validate_stream_fixity',
            self.embed_stream_cb: 'embed_stream_fixity',
            self.output_fixity_cb: 'output_fixity',
            self.overwrite_stream_cb: 'overwrite_stream_fixity'
        }
        
        for checkbox, field in fixity_checkboxes.items():
            checkbox.stateChanged.connect(
                lambda state, f=field: self.on_checkbox_changed(state, ['fixity', f])
            )
        
        # Tools section
        for tool, widgets in self.tool_widgets.items():
            if tool == 'qctools':
                widgets['run'].stateChanged.connect(
                lambda state, t=tool: self.on_checkbox_changed(state, ['tools', t, 'run_tool'])
            )
            else:
                widgets['check'].stateChanged.connect(
                    lambda state, t=tool: self.on_checkbox_changed(state, ['tools', t, 'check_tool'])
                )
                widgets['run'].stateChanged.connect(
                    lambda state, t=tool: self.on_checkbox_changed(state, ['tools', t, 'run_tool'])
                )
        
        # MediaConch
        mediaconch = self.checks_config.tools.mediaconch
        self.run_mediaconch_cb.setChecked(mediaconch.run_mediaconch.lower() == 'yes')
        
        self.run_mediaconch_cb.stateChanged.connect(
            lambda state: self.on_checkbox_changed(state, ['tools', 'mediaconch', 'run_mediaconch'])
        )
        self.policy_combo.currentTextChanged.connect(self.on_mediaconch_policy_changed)
        self.import_policy_btn.clicked.connect(self.open_policy_file_dialog)
                    
        # QCT Parse
        self.run_qctparse_cb.stateChanged.connect(
            lambda state: self.on_checkbox_changed(state, ['tools', 'qct_parse', 'run_tool'])
        )
        self.bars_detection_cb.stateChanged.connect(
            lambda state: self.on_boolean_changed(state, ['tools', 'qct_parse', 'barsDetection'])
        )
        self.evaluate_bars_cb.stateChanged.connect(
            lambda state: self.on_boolean_changed(state, ['tools', 'qct_parse', 'evaluateBars'])
        )
        self.thumb_export_cb.stateChanged.connect(
            lambda state: self.on_boolean_changed(state, ['tools', 'qct_parse', 'thumbExport'])
        )
        self.content_filter_combo.currentTextChanged.connect(
            lambda text: self.on_qct_combo_changed(text, 'contentFilter')
        )
        self.profile_combo.currentTextChanged.connect(
            lambda text: self.on_qct_combo_changed(text, 'profile')
        )
        self.tagname_input.textChanged.connect(
            lambda text: self.on_tagname_changed(text)
        )

    def load_config_values(self):
        """Load current config values into UI elements"""
        # Set loading flag to True
        self.is_loading = True

        # Outputs
        self.access_file_cb.setChecked(self.checks_config.outputs.access_file.lower() == 'yes')
        self.report_cb.setChecked(self.checks_config.outputs.report.lower() == 'yes')
        self.qctools_ext_input.setText(self.checks_config.outputs.qctools_ext)
        
        # Fixity
        self.check_fixity_cb.setChecked(self.checks_config.fixity.check_fixity.lower() == 'yes')
        self.validate_stream_cb.setChecked(self.checks_config.fixity.validate_stream_fixity.lower() == 'yes')
        self.embed_stream_cb.setChecked(self.checks_config.fixity.embed_stream_fixity.lower() == 'yes')
        self.output_fixity_cb.setChecked(self.checks_config.fixity.output_fixity.lower() == 'yes')
        self.overwrite_stream_cb.setChecked(self.checks_config.fixity.overwrite_stream_fixity.lower() == 'yes')
        
        # Tools
        for tool, widgets in self.tool_widgets.items():
            tool_config = getattr(self.checks_config.tools, tool)
            if tool == 'qctools':
                widgets['run'].setChecked(tool_config.run_tool.lower() == 'yes')
            else:
                widgets['check'].setChecked(tool_config.check_tool.lower() == 'yes')
                widgets['run'].setChecked(tool_config.run_tool.lower() == 'yes')
        
        # MediaConch
        mediaconch = self.checks_config.tools.mediaconch
        self.run_mediaconch_cb.setChecked(mediaconch.run_mediaconch.lower() == 'yes')
        
        # Update current policy display
        self.update_current_policy_display(mediaconch.mediaconch_policy)
        
        # Load available policies
        available_policies = self.config_mgr.get_available_policies()
        self.policy_combo.clear()
        self.policy_combo.addItems(available_policies)
        
        # Temporarily block signals while setting the current text
        self.policy_combo.blockSignals(True)
        mediaconch = self.checks_config.tools.mediaconch
        if mediaconch.mediaconch_policy in available_policies:
            self.policy_combo.setCurrentText(mediaconch.mediaconch_policy)
        self.policy_combo.blockSignals(False)
        
        # QCT Parse
        qct = self.checks_config.tools.qct_parse
        self.run_qctparse_cb.setChecked(qct.run_tool.lower() == 'yes')
        self.bars_detection_cb.setChecked(qct.barsDetection)
        self.evaluate_bars_cb.setChecked(qct.evaluateBars)
        self.thumb_export_cb.setChecked(qct.thumbExport)
        
        if qct.contentFilter:
            self.content_filter_combo.setCurrentText(qct.contentFilter[0])
        if qct.profile:
            self.profile_combo.setCurrentText(qct.profile[0])
        if qct.tagname is not None:
            self.tagname_input.setText(qct.tagname)

        # Set loading flag back to False after everything is loaded
        self.is_loading = False

    def on_checkbox_changed(self, state, path):
        """Handle changes in yes/no checkboxes"""
        new_value = 'yes' if Qt.CheckState(state) == Qt.CheckState.Checked else 'no'
        
        if path[0] == "tools" and len(path) > 2:
            tool_name = path[1]
            field = path[2]
            updates = {'tools': {tool_name: {field: new_value}}}
        else:
            section = path[0]
            field = path[1]
            updates = {section: {field: new_value}}
            
        self.config_mgr.update_config('checks', updates)

    def on_boolean_changed(self, state, path):
        """Handle changes in boolean checkboxes"""
        new_value = Qt.CheckState(state) == Qt.CheckState.Checked
        
        if path[0] == "tools" and path[1] == "qct_parse":
            updates = {'tools': {'qct_parse': {path[2]: new_value}}}
            self.config_mgr.update_config('checks', updates)

    def on_text_changed(self, path, text):
        """Handle changes in text inputs"""
        updates = {path[0]: {path[1]: text}}
        self.config_mgr.update_config('checks', updates)

    def on_qct_combo_changed(self, text, field):
        """Handle changes in QCT Parse combo boxes"""
        value = [text] if text != "Select options..." else []
        updates = {'tools': {'qct_parse': {field: value}}}
        self.config_mgr.update_config('checks', updates)

    def on_tagname_changed(self, text):
        """Handle changes in tagname field"""
        updates = {'tools': {'qct_parse': {'tagname': text if text else None}}}
        self.config_mgr.update_config('checks', updates)

    def on_mediaconch_policy_changed(self, policy_name):
        """Handle selection of MediaConch policy"""
        if not self.is_loading and policy_name:
            self.config_mgr.update_config('checks', {
                'tools': {
                    'mediaconch': {
                        'mediaconch_policy': policy_name
                    }
                }
            })
            self.update_current_policy_display(policy_name)
            logger.info(f"Updated config to use policy file: {policy_name}")

    def update_current_policy_display(self, policy_name):
        """Update the display of the current policy"""
        if policy_name:
            self.current_policy_display.setText(policy_name)
        else:
            self.current_policy_display.setText("No policy selected")

    def open_policy_file_dialog(self):
        """Open file dialog for selecting MediaConch policy file"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("XML files (*.xml)")
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                policy_path = selected_files[0]
                # Call setup_mediaconch_policy with selected file
                new_policy_name = setup_mediaconch_policy(policy_path)
                if new_policy_name:
                    # Refresh the UI to show the new policy file
                    self.load_config_values()
                else:
                    # Show error message if policy setup failed
                    QMessageBox.critical(
                        self,
                        "Error",
                        "Failed to import MediaConch policy file. Check logs for details."
                    )

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.signals = ProcessingSignals()
        self.worker = None  # Initialize worker as None
        self.processing_window = None

        # Connect all signals
        self.setup_signal_connections()

        # Init processing window
        self.processing_window = None
        
        # Setup UI
        self.setup_ui()

    def setup_signal_connections(self):
        """Setup all signal connections"""
        # Processing window signals
        self.signals.started.connect(self.on_processing_started)
        self.signals.completed.connect(self.on_processing_completed)
        self.signals.error.connect(self.on_error)
        self.signals.status_update.connect(self.on_status_update)
        self.signals.cancelled.connect(self.on_processing_cancelled)
        
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
            # logger.debug(f"Error starting worker thread: {str(e)}")
            self.signals.error.emit(str(e))

    def on_worker_finished(self):
        """Handle worker thread completion"""
        if self.processing_window:
            self.processing_window.close()
            self.processing_window = None
        self.check_spex_button.setEnabled(True)
        
        # Clean up the worker
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

        # Slot methods
    def on_processing_started(self, message):
        """Handle processing start"""
        # logger.debug("Processing started with message:", message)
        if self.processing_window is None:
            self.processing_window = ProcessingWindow(self)
            self.processing_window.cancel_button.clicked.connect(self.cancel_processing)
        self.processing_window.update_status(message)
        self.processing_window.show()
        self.processing_window.raise_()
        self.check_spex_button.setEnabled(False)
        QApplication.processEvents()
        
    def on_processing_completed(self, message):
        if self.processing_window:
            self.processing_window.close()
            self.processing_window = None
        # Re-enable the Check Spex button
        self.check_spex_button.setEnabled(True)
        QMessageBox.information(self, "Complete", message)
    
    def on_processing_time(self, formatted_time):
        """Handle processing time message from worker"""
        QMessageBox.information(self, "Complete", f"Processing completed in {formatted_time}!")

    def on_error(self, error_message):
        """Handle errors"""
        if self.processing_window:
            self.processing_window.close()
            self.processing_window = None
        self.check_spex_button.setEnabled(True)
        QMessageBox.critical(self, "Error", error_message)
        
        # Clean up worker if it exists
        if self.worker:
            self.worker.quit()
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None

    def on_status_update(self, message):
        """Handle status updates"""
        if self.processing_window:
            self.processing_window.update_status(message)

    def cancel_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.processing_window.update_status("Cancelling processing...")
            self.processing_window.cancel_button.setEnabled(False)

    def on_processing_cancelled(self):
        if self.processing_window:
            self.processing_window.close()
            self.processing_window = None
        self.check_spex_button.setEnabled(True)
        QMessageBox.information(self, "Cancelled", "Processing was cancelled.")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        # Call our quit handling method
        self.on_quit_clicked()
        super().closeEvent(event)

    def on_tool_started(self, tool_name):
        if self.processing_window:
            self.processing_window.update_status(f"Starting {tool_name}")
        
    def on_tool_completed(self, message):
        if self.processing_window:
            self.processing_window.update_status(message)
            # Let UI update
            QApplication.processEvents()

    def on_fixity_progress(self, message):
        if self.processing_window:
            self.processing_window.update_detailed_status(message)

    def on_mediaconch_progress(self, message):
        if self.processing_window:
            self.processing_window.update_detailed_status(message)

    def on_metadata_progress(self, message):
        if self.processing_window:
            self.processing_window.update_detailed_status(message)

    def on_output_progress(self, message):
        if self.processing_window:
            self.processing_window.update_detailed_status(message)

    def add_custom_filename_button(self):
        custom_button = QPushButton("Create Custom Pattern...")
        custom_button.clicked.connect(self.show_custom_filename_dialog)
        # Add to the filename section layout that's already defined
        self.filename_section_layout.addWidget(custom_button)

    def show_custom_filename_dialog(self):
        dialog = CustomFilenameDialog(self)
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted:
            pattern = dialog.get_filename_pattern()
            if pattern:
                try:
                    # Use the first section's value for the custom name
                    first_section_key = next(iter(pattern.fn_sections))
                    first_section = pattern.fn_sections[first_section_key]
                    custom_name = f"Custom ({first_section.value})"
                    
                    # Check if this custom pattern already exists in the dropdown
                    found = False
                    for i in range(self.filename_profile_dropdown.count()):
                        if self.filename_profile_dropdown.itemText(i) == custom_name:
                            found = True
                            break
                    
                    # Only add if it's not already in the dropdown
                    if not found:
                        self.filename_profile_dropdown.addItem(custom_name)
                        self.filename_profile_dropdown.setCurrentText(custom_name)
                        
                        # Get the ConfigManager instance
                        config_manager = ConfigManager()
                        
                        # Get the current filename configuration
                        filename_config = config_manager.get_config('filename', FilenameConfig)
                        
                        # Create an updated dictionary of profiles
                        updated_profiles = dict(filename_config.filename_profiles)
                        updated_profiles[custom_name] = pattern
                        
                        # Create a new FilenameConfig with the updated profiles
                        new_config = FilenameConfig(
                            filename_profiles=updated_profiles
                        )
                        
                        # Set the config with the complete new object
                        config_manager.set_config('filename', new_config)
                        
                        # Save the last used configuration
                        config_manager.save_last_used_config('filename')
                        
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Error adding custom pattern to dropdown: {str(e)}")                        

        
    def setup_ui(self):
        self.config_mgr = ConfigManager()
        self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
        self.spex_config = self.config_mgr.get_config('spex', SpexConfig)

        self.resize(800, 900)
        
         # Initialize settings
        self.settings = QSettings('NMAAHC', 'AVSpex')
        self.selected_directories = []
        
        self.check_spex_clicked = False
        self.setWindowTitle("AV Spex")
        
        # Set up menu bar
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.file_menu = self.menu_bar.addMenu("File")
        self.import_action = self.file_menu.addAction("Import Directory")
        self.import_action.triggered.connect(self.import_directories)
        # Quit action
        self.quit_action = self.file_menu.addAction("Quit")
        self.quit_action.triggered.connect(self.on_quit_clicked)

        self.setup_main_layout()
        
        self.logo_setup()

        self.setup_tabs()

    def logo_setup(self):
        if getattr(sys, 'frozen', False):
            QTimer.singleShot(0, self._delayed_logo_setup)
        else:
            self._load_logo()

    def _delayed_logo_setup(self):
        self._load_logo()

    def _load_logo(self):
        logo_path = self.config_mgr.get_logo_path('JPCA_H_Branding_011025.png')
        image_layout = self.add_image_to_top(logo_path)
        self.main_layout.insertLayout(0, image_layout)  # Insert at index 0 (top)

    # Create a QTabWidget for tabs
    def setup_tabs(self):
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.setup_checks_tab()
        self.setup_spex_tab()

    # First tab: "checks"
    def setup_checks_tab(self):
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

        # Import directory button
        import_directories_button = QPushButton("Import Directory...")
        import_directories_button.clicked.connect(self.import_directories)
        vertical_layout.addWidget(import_directories_button)

        # Directory list
        directory_label = QLabel("Selected Directories:")
        self.directory_list = DirectoryListWidget(self)
        vertical_layout.addWidget(directory_label)
        vertical_layout.addWidget(self.directory_list)

        # Delete button
        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self.delete_selected_directory)
        vertical_layout.addWidget(delete_button)

        # Command Profile section
        command_profile_label = QLabel("Command profiles:")
        self.command_profile_dropdown = QComboBox()
        self.command_profile_dropdown.addItem("step1")
        self.command_profile_dropdown.addItem("step2")
        self.command_profile_dropdown.addItem("allOff")
        
        # Set initial dropdown state
        if self.checks_config.tools.exiftool.run_tool == "yes":
            self.command_profile_dropdown.setCurrentText("step1")
        elif self.checks_config.tools.exiftool.run_tool == "no":
            self.command_profile_dropdown.setCurrentText("step2")

        self.command_profile_dropdown.currentIndexChanged.connect(self.on_profile_selected)
        vertical_layout.addWidget(command_profile_label)
        vertical_layout.addWidget(self.command_profile_dropdown)

        # Config section
        command_checks_label = QLabel("Command options:")
        config_scroll_area = QScrollArea()
        self.config_widget = ConfigWindow(config_mgr=self.config_mgr)
        config_scroll_area.setWidgetResizable(True)
        config_scroll_area.setWidget(self.config_widget)

        # Set a minimum width for the config widget to ensure legibility
        config_scroll_area.setMinimumWidth(400)

        # Add checkboxes and label to the vertical layout
        vertical_layout.addWidget(command_checks_label)
        vertical_layout.addWidget(config_scroll_area)

        # Add scroll area to main layout
        checks_layout.addWidget(main_scroll_area)

        # Bottom button section
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        self.check_spex_button = QPushButton("Check Spex!")
        self.check_spex_button.clicked.connect(self.on_check_spex_clicked)
        bottom_row.addWidget(self.check_spex_button)
        checks_layout.addLayout(bottom_row)

    # Second tab: "spex"
    def setup_spex_tab(self):
        spex_tab = QWidget()
        spex_layout = QVBoxLayout(spex_tab)
        self.tabs.addTab(spex_tab, "Spex")

        filename_section_group = self.setup_filename_section()
        spex_layout.addWidget(filename_section_group)

        mediainfo_section_group = self.setup_mediainfo_section()
        spex_layout.addWidget(mediainfo_section_group)

        exiftool_section_group = self.setup_exiftool_section()
        spex_layout.addWidget(exiftool_section_group)

        ffprobe_section_group = self.setup_ffprobe_section()
        spex_layout.addWidget(ffprobe_section_group)

        mediatrace_section_group = self.setup_mediatrace_section()
        spex_layout.addWidget(mediatrace_section_group)

        qct_section_group = self.setup_qct_section()
        spex_layout.addWidget(qct_section_group)
    
    
    # Main layout
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

    # File name section
    def setup_filename_section(self):
        self.filename_config = self.config_mgr.get_config("filename", FilenameConfig)

        # Filename section
        filename_section_group = QGroupBox()
        filename_section_layout = QVBoxLayout()
        # Create a label to display the section name
        filename_section_label = QLabel(f"<b>Filename Values</b>")
        filename_section_layout.addWidget(filename_section_label)

        # Add a dropdown menu for command profiles
        filenames_profile_label = QLabel("Expected filename profiles:")
        filename_section_layout.addWidget(filenames_profile_label)

        self.filename_profile_dropdown = QComboBox()
        self.filename_profile_dropdown.addItem("Select a profile...")
        
        # Add any custom filename profiles from the config
        if hasattr(self.filename_config, 'filename_profiles') and self.filename_config.filename_profiles:
            for profile_name in self.filename_config.filename_profiles.keys():
                self.filename_profile_dropdown.addItem(profile_name)

        # Set initial state
        if self.spex_config.filename_values.fn_sections["section1"].value == "JPC":
            self.filename_profile_dropdown.setCurrentText("JPC Filename Profile")
        elif self.spex_config.filename_values.fn_sections["section1"].value == "2012":
            self.filename_profile_dropdown.setCurrentText("Bowser Filename Profile")
        else:
             self.filename_profile_dropdown.setCurrentText("Select a profile...")
            
        self.filename_profile_dropdown.currentIndexChanged.connect(self.on_filename_profile_changed)
        filename_section_layout.addWidget(self.filename_profile_dropdown)

        # Store the layout as an instance variable so it can be accessed by add_custom_filename_button
        self.filename_section_layout = filename_section_layout
        
        # Add the custom filename button
        self.add_custom_filename_button()
        
        # Open section button
        button = QPushButton("Open Section")
        button.clicked.connect(
            lambda: self.open_new_window('Filename Values', asdict(self.spex_config.filename_values))
        )
        filename_section_layout.addWidget(button)
        
        filename_section_group.setLayout(filename_section_layout)
        filename_section_group.setFixedHeight(150)
        return filename_section_group

    # Section setup functions for each tool (mediainfo, exiftool, ffprobe)
    def setup_mediainfo_section(self):
        group = QGroupBox()
        layout = QVBoxLayout()
        
        # Section label
        layout.addWidget(QLabel("<b>MediaInfo Values</b>"))
        # Create a toggle button to open a new window
        button = QPushButton("Open Section")
        button.clicked.connect(
            lambda: self.open_new_window('MediaInfo Values', self.spex_config.mediainfo_values)
        )
        layout.addWidget(button)
        
        group.setLayout(layout)
        group.setFixedHeight(100)
        return group
    
    def setup_exiftool_section(self):
        group = QGroupBox()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<b>Exiftool Values</b>"))
        button = QPushButton("Open Section")
        button.clicked.connect(
            lambda: self.open_new_window('Exiftool Values', asdict(self.spex_config.exiftool_values))
        )
        layout.addWidget(button)
        
        group.setLayout(layout)
        group.setFixedHeight(100)
        return group
    
    def setup_ffprobe_section(self):
        group = QGroupBox()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<b>FFprobe Values</b>"))
        button = QPushButton("Open Section")
        button.clicked.connect(
            lambda: self.open_new_window('FFprobe Values', self.spex_config.ffmpeg_values)
        )
        layout.addWidget(button)
        
        group.setLayout(layout)
        group.setFixedHeight(100)
        return group
    
    # Mediatrace section has custom dropdowns
    def setup_mediatrace_section(self):
        group = QGroupBox()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<b>Mediatrace Values</b>"))
        
        # Signalflow profile dropdown
        layout.addWidget(QLabel("Expected Signalflow profiles:"))
        self.signalflow_profile_dropdown = QComboBox()
        self.signalflow_profile_dropdown.addItem("JPC_AV_SVHS Signal Flow")
        self.signalflow_profile_dropdown.addItem("BVH3100 Signal Flow")
        
        # Set initial state
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
        layout.addWidget(self.signalflow_profile_dropdown)
        
        button = QPushButton("Open Section")
        button.clicked.connect(
            lambda: self.open_new_window('Mediatrace Values', asdict(self.spex_config.mediatrace_values))
        )
        layout.addWidget(button)
        
        group.setLayout(layout)
        group.setFixedHeight(150)
        return group
    
    def setup_qct_section(self):
        group = QGroupBox()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<b>qct-parse Values</b>"))
        button = QPushButton("Open Section")
        button.clicked.connect(
            lambda: self.open_new_window('Expected qct-parse options', asdict(self.spex_config.qct_parse_values))
        )
        layout.addWidget(button)
        
        group.setLayout(layout)
        group.setFixedHeight(100)
        return group
    
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

    def add_network_drive(self, file_dialog):
        """
        Open a dialog to add a network drive manually
        """
        network_dialog = QInputDialog(self)
        network_dialog.setWindowTitle("Add Network Drive")
        network_dialog.setLabelText("Enter network path (e.g., //servername/share):")
        network_dialog.setTextValue("//")
        
        if network_dialog.exec():
            network_path = network_dialog.textValue()
            
            # Basic validation of network path
            if network_path.startswith("//"):
                # Check if the network path exists and is accessible
                if os.path.exists(network_path):
                    file_dialog.setDirectory(network_path)
                else:
                    QMessageBox.warning(self, "Network Drive", 
                                        "Cannot access the specified network path. "
                                        "Please check the path and your network connection.")
            else:
                QMessageBox.warning(self, "Invalid Path", 
                                    "Please enter a valid network path starting with //")


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
        """Handle the Start button click."""
        # logger.debug("Check Spex button clicked")  # Debug line
        self.update_selected_directories()
        self.check_spex_clicked = True  # Mark that the button was clicked
        self.config_mgr.save_last_used_config('checks')
        self.config_mgr.save_last_used_config('spex')
        self.call_process_directories()


    def on_profile_selected(self, index):
        selected_profile = self.command_profile_dropdown.currentText()
        if selected_profile == "step1":
            profile = edit_config.profile_step1
        elif selected_profile == "step2":
            profile = edit_config.profile_step2
        elif selected_profile == "allOff":
            profile = edit_config.profile_allOff
        try:
            # Call the backend function to apply the selected profile
            edit_config.apply_profile(profile)
            logger.debug(f"Profile '{selected_profile}' applied successfully.")
            self.config_mgr.save_last_used_config('checks')
        except ValueError as e:
            logger.critical(f"Error: {e}")

        self.config_widget.load_config_values()


    def on_filename_profile_changed(self, index):
        filename_config = self.config_mgr.get_config("filename", FilenameConfig)
        jpc_filename_profile = filename_config.filename_profiles["JPC Filename Profile"]
        bowser_filename_profile = filename_config.filename_profiles["Bowser Filename Profile"]

        selected_option = self.filename_profile_dropdown.itemText(index)
        
        if selected_option == "JPC Filename Profile":
            edit_config.apply_filename_profile(jpc_filename_profile)
            self.config_mgr.save_last_used_config('spex')
        elif selected_option == "Bowser Filename Profile":
            edit_config.apply_filename_profile(bowser_filename_profile)
            self.config_mgr.save_last_used_config('spex')
        elif selected_option.startswith("Custom ("):
            for profile_name in self.filename_config.filename_profiles.keys():
                if selected_option == profile_name:
                    profile_class = filename_config.filename_profiles[profile_name]
                    edit_config.apply_filename_profile(profile_class)
                    self.config_mgr.save_last_used_config('spex')

    def on_signalflow_profile_changed(self, index):
        selected_option = self.signalflow_profile_dropdown.itemText(index)
        logger.debug(f"Selected signal flow profile: {selected_option}")

        if selected_option == "JPC_AV_SVHS Signal Flow":
            sn_config_changes = edit_config.JPC_AV_SVHS
        elif selected_option == "BVH3100 Signal Flow":
            sn_config_changes = edit_config.BVH3100
        else:
            logger.error("Signal flow identifier not recognized, config not updated")
            return

        # Update FFmpeg settings
        self.config_mgr.update_config('spex', {
            'ffmpeg_values': {
                'format': {
                    'tags': {
                        'ENCODER_SETTINGS': sn_config_changes
                    }
                }
            }
        })

        # Update MediaTrace settings 
        self.config_mgr.update_config('spex', {
            'mediatrace_values': {
                'ENCODER_SETTINGS': sn_config_changes
            }
        })


    def open_new_window(self, title, nested_dict):
        # Convert any dataclass instances in mediainfo_values to dictionaries
        if title == 'MediaInfo Values':
            nested_dict = {
                'expected_general': nested_dict['expected_general'],
                'expected_video': nested_dict['expected_video'], 
                'expected_audio': nested_dict['expected_audio']
            }
        # Convert ffmpeg_values dataclass instances
        elif title == 'FFprobe Values':
            nested_dict = {
                'video_stream': nested_dict['video_stream'],
                'audio_stream': nested_dict['audio_stream'],
                'format': nested_dict['format']
            }

        content_text = self.dict_to_string(nested_dict)
    
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
        
