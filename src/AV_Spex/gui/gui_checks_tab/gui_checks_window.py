from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QLineEdit,
    QLabel, QComboBox, QPushButton, QScrollArea, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette

from AV_Spex.gui.gui_theme_manager import ThemeManager, ThemeableMixin
from AV_Spex.utils.config_setup import ChecksConfig, SpexConfig
from AV_Spex.utils.config_manager import ConfigManager

from AV_Spex.processing.processing_mgmt import setup_mediaconch_policy

config_mgr= ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)

class ChecksWindow(QWidget, ThemeableMixin):
    """Configuration window for managing application settings."""
    
    def __init__(self):
        super().__init__()
        self.is_loading = False

        # Initialize themed_group_boxes before setup_theme_handling
        self.themed_group_boxes = {}
        
        # Setup theme handling
        self.setup_theme_handling()
        
        # Setup UI and load config
        self.setup_ui()
        self.load_config_values()

    def setup_ui(self):
        """Create fixed layout structure"""
        main_layout = QVBoxLayout(self)

        self.setup_validation_section(main_layout)
        self.setup_outputs_section(main_layout)
        self.setup_fixity_section(main_layout)
        self.setup_tools_section(main_layout)
        self.connect_signals()
        
    # Validation Section
    def setup_validation_section(self, main_layout):
        """Set up the validation section with palette-aware styling"""
        theme_manager = ThemeManager.instance()
        
        # Creating a new validation group
        self.validation_group = QGroupBox("Validation")
        theme_manager.style_groupbox(self.validation_group, "top left")
        self.themed_group_boxes['validation'] = self.validation_group

        validation_layout = QVBoxLayout()
        
        # Create checkbox with description on second line
        self.validate_filename_cb = QCheckBox("Validate Filename")
        self.validate_filename_cb.setStyleSheet("font-weight: bold;")
        validate_filename_desc = QLabel("Check if filename matches the expected pattern from the Filename profile")
        validate_filename_desc.setIndent(20)  # Indented to align with checkbox text
        
        # Add to layout
        validation_layout.addWidget(self.validate_filename_cb)
        validation_layout.addWidget(validate_filename_desc)
        
        self.validation_group.setLayout(validation_layout)
        main_layout.addWidget(self.validation_group)
    
    # Outputs Section
    def setup_outputs_section(self, main_layout):
        """Set up the outputs section with palette-aware styling"""
        theme_manager = ThemeManager.instance()
        
        # Creating a new outputs group
        self.outputs_group = QGroupBox("Outputs")
        theme_manager.style_groupbox(self.outputs_group, "top left")
        self.themed_group_boxes['outputs'] = self.outputs_group

        outputs_layout = QVBoxLayout()
        
        # Create widgets with descriptions on second line
        self.access_file_cb = QCheckBox("Access File")
        self.access_file_cb.setStyleSheet("font-weight: bold;")
        access_file_desc = QLabel("Creates a h264 access file of the input .mkv file")
        access_file_desc.setIndent(20)  # Indented to align with checkbox text
        
        self.report_cb = QCheckBox("HTML Report")
        self.report_cb.setStyleSheet("font-weight: bold;")
        report_desc = QLabel("Creates a .html report containing the results of Spex Checks")
        report_desc.setIndent(20)
        
        self.qctools_ext_label = QLabel("QCTools File Extension")
        self.qctools_ext_label.setStyleSheet("font-weight: bold;")
        qctools_ext_desc = QLabel("Set the extension for QCTools output")
        self.qctools_ext_input = QLineEdit()

        # Create a horizontal layout for the QCTools extension input
        qctools_ext_layout = QHBoxLayout()
        qctools_ext_layout.addWidget(self.qctools_ext_label)
        qctools_ext_layout.addWidget(self.qctools_ext_input)
        
        # Add to layout
        outputs_layout.addWidget(self.access_file_cb)
        outputs_layout.addWidget(access_file_desc)
        outputs_layout.addWidget(self.report_cb)
        outputs_layout.addWidget(report_desc)

        # Create a vertical layout just for the QCTools section
        qctools_section = QVBoxLayout()
        qctools_section.addLayout(qctools_ext_layout)
        qctools_ext_desc.setIndent(150)
        qctools_section.addWidget(qctools_ext_desc)
        # Add the QCTools section to the main outputs layout
        outputs_layout.addLayout(qctools_section)
        
        self.outputs_group.setLayout(outputs_layout)
        main_layout.addWidget(self.outputs_group)
    
    # Fixity Section - RESTRUCTURED with two algorithm dropdowns
    def setup_fixity_section(self, main_layout):
        """Set up the fixity section with palette-aware styling.
        
        Layout structure:
        - File Fixity row: [Checksum Algorithm dropdown] [Output fixity] [Validate fixity]
        - Stream Fixity row: [Stream Hash Algorithm dropdown] [Embed] [Overwrite] [Validate]
        """
        theme_manager = ThemeManager.instance()
        
        # Creating a new fixity group
        self.fixity_group = QGroupBox("Fixity")
        theme_manager.style_groupbox(self.fixity_group, "top left")
        self.themed_group_boxes['fixity'] = self.fixity_group
        
        fixity_layout = QVBoxLayout()
        
        # ========================================================================
        # FILE FIXITY SECTION
        # ========================================================================
        file_fixity_label = QLabel("File Fixity")
        file_fixity_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        fixity_layout.addWidget(file_fixity_label)
        
        # Container for file fixity row
        file_fixity_container = QWidget()
        file_fixity_layout = QHBoxLayout(file_fixity_container)
        file_fixity_layout.setContentsMargins(0, 5, 0, 10)
        
        # Checksum algorithm dropdown (left side)
        algorithm_widget = QWidget()
        algorithm_layout = QVBoxLayout(algorithm_widget)
        algorithm_layout.setContentsMargins(0, 0, 20, 0)
        algorithm_layout.setSpacing(2)
        
        algorithm_label = QLabel("Checksum Algorithm:")
        algorithm_label.setStyleSheet("font-weight: bold;")
        
        self.checksum_algorithm_combo = QComboBox()
        self.checksum_algorithm_combo.addItems(["md5", "sha256"])
        self.checksum_algorithm_combo.setMinimumWidth(120)
        
        algorithm_desc = QLabel("Hash algorithm for file checksums")
        algorithm_desc.setStyleSheet("color: gray; font-size: 10px;")
        
        algorithm_layout.addWidget(algorithm_label)
        algorithm_layout.addWidget(self.checksum_algorithm_combo)
        algorithm_layout.addWidget(algorithm_desc)
        
        file_fixity_layout.addWidget(algorithm_widget)
        
        # File fixity checkboxes (right side)
        file_checkboxes_widget = QWidget()
        file_checkboxes_layout = QVBoxLayout(file_checkboxes_widget)
        file_checkboxes_layout.setContentsMargins(0, 0, 0, 0)
        file_checkboxes_layout.setSpacing(5)
        
        # Output fixity checkbox
        self.output_fixity_cb = QCheckBox("Output fixity")
        self.output_fixity_cb.setStyleSheet("font-weight: bold;")
        output_fixity_desc = QLabel("Generate whole file checksum of .mkv files")
        output_fixity_desc.setIndent(20)
        output_fixity_desc.setStyleSheet("color: gray; font-size: 10px;")
        
        # Validate fixity checkbox
        self.check_fixity_cb = QCheckBox("Validate fixity")
        self.check_fixity_cb.setStyleSheet("font-weight: bold;")
        check_fixity_desc = QLabel("Validate .mkv files against existing checksum file")
        check_fixity_desc.setIndent(20)
        check_fixity_desc.setStyleSheet("color: gray; font-size: 10px;")
        
        file_checkboxes_layout.addWidget(self.output_fixity_cb)
        file_checkboxes_layout.addWidget(output_fixity_desc)
        file_checkboxes_layout.addWidget(self.check_fixity_cb)
        file_checkboxes_layout.addWidget(check_fixity_desc)
        
        file_fixity_layout.addWidget(file_checkboxes_widget)
        file_fixity_layout.addStretch()
        
        fixity_layout.addWidget(file_fixity_container)
        
        # Separator line
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: gray;")
        fixity_layout.addWidget(separator)
        
        # ========================================================================
        # STREAM FIXITY SECTION
        # ========================================================================
        stream_fixity_label = QLabel("Stream Fixity")
        stream_fixity_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        fixity_layout.addWidget(stream_fixity_label)
        
        # Container for stream fixity row
        stream_fixity_container = QWidget()
        stream_fixity_layout = QHBoxLayout(stream_fixity_container)
        stream_fixity_layout.setContentsMargins(0, 5, 0, 0)
        
        # Stream hash algorithm dropdown (left side)
        stream_algorithm_widget = QWidget()
        stream_algorithm_layout = QVBoxLayout(stream_algorithm_widget)
        stream_algorithm_layout.setContentsMargins(0, 0, 20, 0)
        stream_algorithm_layout.setSpacing(2)
        
        stream_algorithm_label = QLabel("Stream Hash Algorithm:")
        stream_algorithm_label.setStyleSheet("font-weight: bold;")
        
        self.stream_hash_algorithm_combo = QComboBox()
        self.stream_hash_algorithm_combo.addItems(["md5", "sha256"])
        self.stream_hash_algorithm_combo.setMinimumWidth(120)
        
        stream_algorithm_desc = QLabel("Hash algorithm for stream checksums")
        stream_algorithm_desc.setStyleSheet("color: gray; font-size: 10px;")
        
        stream_algorithm_layout.addWidget(stream_algorithm_label)
        stream_algorithm_layout.addWidget(self.stream_hash_algorithm_combo)
        stream_algorithm_layout.addWidget(stream_algorithm_desc)
        
        stream_fixity_layout.addWidget(stream_algorithm_widget)
        
        # Stream fixity checkboxes (right side)
        stream_checkboxes_widget = QWidget()
        stream_checkboxes_layout = QVBoxLayout(stream_checkboxes_widget)
        stream_checkboxes_layout.setContentsMargins(0, 0, 0, 0)
        stream_checkboxes_layout.setSpacing(5)
        
        # Embed stream fixity checkbox
        self.embed_stream_cb = QCheckBox("Embed Stream fixity")
        self.embed_stream_cb.setStyleSheet("font-weight: bold;")
        embed_stream_desc = QLabel("Embed video and audio stream checksums into .mkv tags")
        embed_stream_desc.setIndent(20)
        embed_stream_desc.setStyleSheet("color: gray; font-size: 10px;")
        
        # Overwrite stream fixity checkbox
        self.overwrite_stream_cb = QCheckBox("Overwrite Stream fixity")
        self.overwrite_stream_cb.setStyleSheet("font-weight: bold;")
        overwrite_stream_desc = QLabel("Embed stream checksums regardless if existing ones are found")
        overwrite_stream_desc.setIndent(20)
        overwrite_stream_desc.setStyleSheet("color: gray; font-size: 10px;")
        
        # Validate stream fixity checkbox
        self.validate_stream_cb = QCheckBox("Validate Stream fixity")
        self.validate_stream_cb.setStyleSheet("font-weight: bold;")
        validate_stream_desc = QLabel("Validate embedded stream fixity, will embed if none found")
        validate_stream_desc.setIndent(20)
        validate_stream_desc.setStyleSheet("color: gray; font-size: 10px;")
        
        stream_checkboxes_layout.addWidget(self.embed_stream_cb)
        stream_checkboxes_layout.addWidget(embed_stream_desc)
        stream_checkboxes_layout.addWidget(self.overwrite_stream_cb)
        stream_checkboxes_layout.addWidget(overwrite_stream_desc)
        stream_checkboxes_layout.addWidget(self.validate_stream_cb)
        stream_checkboxes_layout.addWidget(validate_stream_desc)
        
        stream_fixity_layout.addWidget(stream_checkboxes_widget)
        stream_fixity_layout.addStretch()
        
        fixity_layout.addWidget(stream_fixity_container)
        
        self.fixity_group.setLayout(fixity_layout)
        main_layout.addWidget(self.fixity_group)
        
    # Tools Section
    def setup_tools_section(self, main_layout):
        """Set up the tools section with palette-aware styling"""
        theme_manager = ThemeManager.instance()

        # Main Tools group box with centered title
        self.tools_group = QGroupBox("Tools")
        theme_manager.style_groupbox(self.tools_group, "top center")
        self.themed_group_boxes['tools'] = self.tools_group

        tools_layout = QVBoxLayout()
        
        # Dictionary to store references to all tool group boxes
        self.tool_group_boxes = {}
        
        # Setup basic tools
        basic_tools = ['exiftool', 'ffprobe', 'mediainfo', 'mediatrace', 'qctools']
        self.tool_widgets = {}
        
        # Individual tool group boxes with left-aligned titles
        for tool in basic_tools:
            # Create new group box for this tool
            tool_group = QGroupBox(tool)
            theme_manager.style_groupbox(tool_group, "top left")
            tool_layout = QVBoxLayout()
            
            # Store reference to this group box
            self.tool_group_boxes[tool] = tool_group
            self.themed_group_boxes[f'tool_{tool}'] = tool_group
            
            if tool.lower() == 'qctools':
                run_cb = QCheckBox("Run Tool")
                run_cb.setStyleSheet("font-weight: bold;")
                run_desc = QLabel("Run QCTools on input video file")
                run_desc.setIndent(20)
                
                self.tool_widgets[tool] = {'run': run_cb}
                tool_layout.addWidget(run_cb)
                tool_layout.addWidget(run_desc)
            else:
                check_cb = QCheckBox("Check Tool")
                check_cb.setStyleSheet("font-weight: bold;")
                check_desc = QLabel("Check the output of the tool against expected Spex")
                check_desc.setIndent(20)
                
                run_cb = QCheckBox("Run Tool")
                run_cb.setStyleSheet("font-weight: bold;")
                run_desc = QLabel(f"Run the tool on the input video")
                run_desc.setIndent(20)
                
                self.tool_widgets[tool] = {'check': check_cb, 'run': run_cb}
                tool_layout.addWidget(check_cb)
                tool_layout.addWidget(check_desc)
                tool_layout.addWidget(run_cb)
                tool_layout.addWidget(run_desc)
            
            tool_group.setLayout(tool_layout)
            tools_layout.addWidget(tool_group)

        # MediaConch section
        self.mediaconch_group = QGroupBox("Mediaconch")
        theme_manager.style_groupbox(self.mediaconch_group, "top left")
        self.themed_group_boxes['mediaconch'] = self.mediaconch_group

        mediaconch_layout = QVBoxLayout()

        self.run_mediaconch_cb = QCheckBox("Run Mediaconch")
        self.run_mediaconch_cb.setStyleSheet("font-weight: bold;")
        run_mediaconch_desc = QLabel("Run MediaConch validation on input files")
        run_mediaconch_desc.setIndent(20)

        # Policy selection
        policy_container = QWidget()
        policy_layout = QVBoxLayout(policy_container)

        # Current policy display
        current_policy_widget = QWidget()
        current_policy_layout = QHBoxLayout(current_policy_widget)
        current_policy_layout.setContentsMargins(0, 0, 0, 0)

        self.policy_label = QLabel("Current policy:")
        self.policy_label.setStyleSheet("font-weight: bold;")
        self.current_policy_display = QLabel()
        self.current_policy_display.setStyleSheet("font-weight: bold;")

        current_policy_layout.addWidget(self.policy_label)
        current_policy_layout.addWidget(self.current_policy_display)
        current_policy_layout.addStretch()

        self.policy_combo = QComboBox()
        
        policies_label = QLabel("Available policies:")
        policies_label.setStyleSheet("font-weight: bold;")
        
        self.import_policy_btn = QPushButton("Import New MediaConch Policy")
        theme_manager.style_button(self.import_policy_btn)
        
        import_policy_desc = QLabel("Import a custom policy file for MediaConch validation")

        policy_layout.addWidget(current_policy_widget)
        policy_layout.addWidget(policies_label)
        policy_layout.addWidget(self.policy_combo)
        policy_layout.addWidget(self.import_policy_btn)
        policy_layout.addWidget(import_policy_desc)

        mediaconch_layout.addWidget(self.run_mediaconch_cb)
        mediaconch_layout.addWidget(run_mediaconch_desc)
        mediaconch_layout.addWidget(policy_container)
        self.mediaconch_group.setLayout(mediaconch_layout)
        tools_layout.addWidget(self.mediaconch_group)

        # QCT Parse section - SIMPLIFIED WITHOUT CONTENT DETECTION AND TAG NAME
        self.qct_group = QGroupBox("qct-parse")
        theme_manager.style_groupbox(self.qct_group, "top left")
        self.themed_group_boxes['qct'] = self.qct_group

        qct_layout = QVBoxLayout()

        # Checkboxes with descriptions on second line
        self.run_qctparse_cb = QCheckBox("Run Tool")
        self.run_qctparse_cb.setStyleSheet("font-weight: bold;")
        run_qctparse_desc = QLabel("Run qct-parse tool on input video file")
        run_qctparse_desc.setIndent(20)

        self.bars_detection_cb = QCheckBox("Detect Color Bars")
        self.bars_detection_cb.setStyleSheet("font-weight: bold;")
        bars_detection_desc = QLabel("Detect color bars in the video content")
        bars_detection_desc.setIndent(20)

        self.evaluate_bars_cb = QCheckBox("Evaluate Color Bars")
        self.evaluate_bars_cb.setStyleSheet("font-weight: bold;")
        evaluate_bars_desc = QLabel("Compare content to color bars for validation")
        evaluate_bars_desc.setIndent(20)

        self.thumb_export_cb = QCheckBox("Thumbnail Export")
        self.thumb_export_cb.setStyleSheet("font-weight: bold;")
        thumb_export_desc = QLabel("Export thumbnails of failed frames for review")
        thumb_export_desc.setIndent(20)

        # Add all widgets to the qct layout 
        qct_layout.addWidget(self.run_qctparse_cb)
        qct_layout.addWidget(run_qctparse_desc)
        qct_layout.addWidget(self.bars_detection_cb)
        qct_layout.addWidget(bars_detection_desc)
        qct_layout.addWidget(self.evaluate_bars_cb)
        qct_layout.addWidget(evaluate_bars_desc)
        qct_layout.addWidget(self.thumb_export_cb)
        qct_layout.addWidget(thumb_export_desc)

        self.qct_group.setLayout(qct_layout)
        tools_layout.addWidget(self.qct_group)
        
        self.tools_group.setLayout(tools_layout)
        main_layout.addWidget(self.tools_group)

    def on_theme_changed(self, palette):
        """Handle theme changes for ChecksWindow"""
        # Apply the palette directly
        self.setPalette(palette)
        
        # Get the theme manager
        theme_manager = ThemeManager.instance()
        
        # Update all tracked group boxes with their specific title positions
        for key, group_box in self.themed_group_boxes.items():
            if key == 'tools':
                theme_manager.style_groupbox(group_box, "top center")
            else:
                # Preserve the title position if set
                position = group_box.property("title_position") or "top left"
                theme_manager.style_groupbox(group_box, position)
        
        # Style all buttons
        theme_manager.style_buttons(self)
            
        # Force repaint
        self.update()

    def connect_signals(self):
        """Connect all widget signals to their handlers"""
        # Validation section
        self.validate_filename_cb.stateChanged.connect(self.on_validate_filename_changed)
        
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
        
        # Fixity section - handle most checkboxes normally
        fixity_checkboxes = {
            self.check_fixity_cb: 'check_fixity',
            self.validate_stream_cb: 'validate_stream_fixity',
            self.embed_stream_cb: 'embed_stream_fixity',
            self.output_fixity_cb: 'output_fixity',
            # Note: overwrite_stream_cb is handled separately below
        }
        
        for checkbox, field in fixity_checkboxes.items():
            checkbox.stateChanged.connect(
                lambda state, f=field: self.on_checkbox_changed(state, ['fixity', f])
            )
        
        # Special handling for overwrite_stream_cb to auto-check embed_stream_cb
        self.overwrite_stream_cb.stateChanged.connect(self.on_overwrite_stream_changed)
        
        # Checksum algorithm dropdown (for file fixity)
        self.checksum_algorithm_combo.currentTextChanged.connect(self.on_checksum_algorithm_changed)
        
        # Stream hash algorithm dropdown (NEW)
        self.stream_hash_algorithm_combo.currentTextChanged.connect(self.on_stream_hash_algorithm_changed)
        
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
        self.run_mediaconch_cb.stateChanged.connect(
            lambda state: self.on_checkbox_changed(state, ['tools', 'mediaconch', 'run_mediaconch'])
        )
        self.policy_combo.currentTextChanged.connect(self.on_mediaconch_policy_changed)
        self.import_policy_btn.clicked.connect(self.open_policy_file_dialog)
                    
        #  Run QCT-Parse turns all the checks on:
        self.run_qctparse_cb.stateChanged.connect(self.on_run_qctparse_changed)

        self.bars_detection_cb.stateChanged.connect(self.on_bars_detection_changed)
        self.evaluate_bars_cb.stateChanged.connect(self.on_evaluate_bars_changed)
        self.thumb_export_cb.stateChanged.connect(
            lambda state: self.on_boolean_changed(state, ['tools', 'qct_parse', 'thumbExport'])
        )

    def load_config_values(self):
        """Load current config values into UI elements"""
        # Set loading flag to True
        self.is_loading = True

        checks_config = config_mgr.get_config('checks', ChecksConfig)

        # Validation
        self.validate_filename_cb.setChecked(checks_config.validate_filename)

        # Outputs - now using booleans directly
        self.access_file_cb.setChecked(checks_config.outputs.access_file)
        self.report_cb.setChecked(checks_config.outputs.report)
        self.qctools_ext_input.setText(checks_config.outputs.qctools_ext)
        
        # Fixity - now using booleans directly
        self.check_fixity_cb.setChecked(checks_config.fixity.check_fixity)
        self.validate_stream_cb.setChecked(checks_config.fixity.validate_stream_fixity)
        self.embed_stream_cb.setChecked(checks_config.fixity.embed_stream_fixity)
        self.output_fixity_cb.setChecked(checks_config.fixity.output_fixity)
        self.overwrite_stream_cb.setChecked(checks_config.fixity.overwrite_stream_fixity)
        
        # Checksum algorithm (for file fixity)
        self.checksum_algorithm_combo.blockSignals(True)
        algorithm = getattr(checks_config.fixity, 'checksum_algorithm', 'md5')
        index = self.checksum_algorithm_combo.findText(algorithm)
        if index >= 0:
            self.checksum_algorithm_combo.setCurrentIndex(index)
        else:
            self.checksum_algorithm_combo.setCurrentText('md5')
        self.checksum_algorithm_combo.blockSignals(False)
        
        # Stream hash algorithm (NEW)
        self.stream_hash_algorithm_combo.blockSignals(True)
        stream_algorithm = getattr(checks_config.fixity, 'stream_hash_algorithm', 'md5')
        stream_index = self.stream_hash_algorithm_combo.findText(stream_algorithm)
        if stream_index >= 0:
            self.stream_hash_algorithm_combo.setCurrentIndex(stream_index)
        else:
            self.stream_hash_algorithm_combo.setCurrentText('md5')
        self.stream_hash_algorithm_combo.blockSignals(False)
        
        # Tools - now using booleans directly
        for tool, widgets in self.tool_widgets.items():
            tool_config = getattr(checks_config.tools, tool)
            if tool == 'qctools':
                widgets['run'].setChecked(tool_config.run_tool)
            else:
                widgets['check'].setChecked(tool_config.check_tool)
                widgets['run'].setChecked(tool_config.run_tool)
        
        # MediaConch - now using boolean directly
        mediaconch = checks_config.tools.mediaconch
        self.run_mediaconch_cb.setChecked(mediaconch.run_mediaconch)
        
        # Update current policy display
        self.update_current_policy_display(mediaconch.mediaconch_policy)
        
        # Load available policies
        available_policies = config_mgr.get_available_policies()
        self.policy_combo.clear()
        self.policy_combo.addItems(available_policies)
        
        # Temporarily block signals while setting the current text
        self.policy_combo.blockSignals(True)
        if mediaconch.mediaconch_policy in available_policies:
            self.policy_combo.setCurrentText(mediaconch.mediaconch_policy)
        self.policy_combo.blockSignals(False)
        
        # QCT Parse - now using boolean directly for run_tool
        qct = checks_config.tools.qct_parse
        self.run_qctparse_cb.setChecked(qct.run_tool)
        # These are already booleans in the original config
        self.bars_detection_cb.setChecked(qct.barsDetection)
        self.evaluate_bars_cb.setChecked(qct.evaluateBars)
        self.thumb_export_cb.setChecked(qct.thumbExport)

        # Set initial enabled state for QCT Parse dependent checkboxes
        qct = checks_config.tools.qct_parse
        dependent_checkboxes = [
            self.bars_detection_cb,
            self.evaluate_bars_cb, 
            self.thumb_export_cb
        ]

        # Enable/disable dependent checkboxes based on run_tool state
        for checkbox in dependent_checkboxes:
            checkbox.setEnabled(qct.run_tool)

        # Additional logic for thumbnail dependency
        if qct.run_tool:
            # If run_tool is enabled, check if thumbnail should be enabled
            # Thumbnail is only enabled if at least one of bars detection or evaluate bars is checked
            thumbnail_should_be_enabled = qct.barsDetection or qct.evaluateBars
            self.thumb_export_cb.setEnabled(thumbnail_should_be_enabled)

        # Set loading flag back to False after everything is loaded
        self.is_loading = False

    def on_checkbox_changed(self, state, path):
        """Handle changes in boolean checkboxes (now saves booleans instead of yes/no)"""
        # Skip updates while loading
        if self.is_loading:
            return

        # Now using boolean values instead of 'yes'/'no'
        new_value = Qt.CheckState(state) == Qt.CheckState.Checked
        
        if path[0] == "tools" and len(path) > 2:
            tool_name = path[1]
            field = path[2]
            updates = {'tools': {tool_name: {field: new_value}}}
        else:
            section = path[0]
            field = path[1]
            updates = {section: {field: new_value}}
            
        config_mgr.update_config('checks', updates)

    def on_validate_filename_changed(self, state):
        """Handle changes in validate filename checkbox"""
        # Skip updates while loading
        if self.is_loading:
            return
        
        # Convert checkbox state to boolean
        new_value = Qt.CheckState(state) == Qt.CheckState.Checked
        
        # Update the top-level validate_filename field
        updates = {'validate_filename': new_value}
        config_mgr.update_config('checks', updates)

    def on_boolean_changed(self, state, path):
        """Handle changes in boolean checkboxes (for qct_parse fields that were already boolean)"""
        # Skip updates while loading
        if self.is_loading:
            return
        
        new_value = Qt.CheckState(state) == Qt.CheckState.Checked
        
        if path[0] == "tools" and path[1] == "qct_parse":
            updates = {'tools': {'qct_parse': {path[2]: new_value}}}
            config_mgr.update_config('checks', updates)

    def on_text_changed(self, path, text):
        """Handle changes in text inputs"""
        # Skip updates while loading
        if self.is_loading:
            return
        
        updates = {path[0]: {path[1]: text}}
        config_mgr.update_config('checks', updates)

    def on_checksum_algorithm_changed(self, algorithm):
        """Handle changes in checksum algorithm selection (for file fixity)"""
        # Skip updates while loading
        if self.is_loading:
            return
        
        updates = {'fixity': {'checksum_algorithm': algorithm}}
        config_mgr.update_config('checks', updates)

    def on_stream_hash_algorithm_changed(self, algorithm):
        """Handle changes in stream hash algorithm selection (NEW)"""
        # Skip updates while loading
        if self.is_loading:
            return
        
        updates = {'fixity': {'stream_hash_algorithm': algorithm}}
        config_mgr.update_config('checks', updates)

    def on_overwrite_stream_changed(self, state):
        """Handle changes in overwrite stream fixity checkbox with dependency logic"""
        # Skip updates while loading to avoid issues during initialization
        if self.is_loading:
            return
        
        # Handle the normal config update for overwrite stream fixity (now using boolean)
        new_value = Qt.CheckState(state) == Qt.CheckState.Checked
        updates = {'fixity': {'overwrite_stream_fixity': new_value}}
        config_mgr.update_config('checks', updates)
        
        # If overwrite stream fixity is being checked, also check embed stream fixity
        if Qt.CheckState(state) == Qt.CheckState.Checked:
            # Temporarily block signals to prevent recursive calls
            self.embed_stream_cb.blockSignals(True)
            self.embed_stream_cb.setChecked(True)
            self.embed_stream_cb.blockSignals(False)
            
            # Update the config for embed stream fixity as well (now using boolean)
            embed_updates = {'fixity': {'embed_stream_fixity': True}}
            config_mgr.update_config('checks', embed_updates)

    def on_qct_combo_changed(self, value, field):
        """Handle changes in QCT Parse combo boxes"""
        # Skip updates while loading
        if self.is_loading:
            return

        values = [value] if value is not None else []
        updates = {'tools': {'qct_parse': {field: values}}}
        config_mgr.update_config('checks', updates)

    def on_run_qctparse_changed(self, state):
        """Handle changes in run qct-parse checkbox with dependency logic"""
        # Skip updates while loading to avoid issues during initialization
        if self.is_loading:
            return
        
        # Handle the normal config update for run_tool (using boolean)
        new_value = Qt.CheckState(state) == Qt.CheckState.Checked
        updates = {'tools': {'qct_parse': {'run_tool': new_value}}}
        config_mgr.update_config('checks', updates)
        
        # Get list of dependent checkboxes
        dependent_checkboxes = [
            self.bars_detection_cb,
            self.evaluate_bars_cb, 
            self.thumb_export_cb
        ]
        
        if Qt.CheckState(state) == Qt.CheckState.Checked:
            # If run_tool is checked, enable bars detection first
            self.bars_detection_cb.setEnabled(True)
            self.bars_detection_cb.blockSignals(True)
            self.bars_detection_cb.setChecked(True)
            self.bars_detection_cb.blockSignals(False)
            
            # Enable and check evaluate bars and thumbnail export
            self.evaluate_bars_cb.setEnabled(True)
            self.evaluate_bars_cb.blockSignals(True)
            self.evaluate_bars_cb.setChecked(True)
            self.evaluate_bars_cb.blockSignals(False)
            
            self.thumb_export_cb.setEnabled(True)
            self.thumb_export_cb.blockSignals(True)
            self.thumb_export_cb.setChecked(True)
            self.thumb_export_cb.blockSignals(False)
            
            # Update the config for all dependent options
            dependent_updates = {
                'tools': {
                    'qct_parse': {
                        'barsDetection': True,
                        'evaluateBars': True,
                        'thumbExport': True
                    }
                }
            }
            config_mgr.update_config('checks', dependent_updates)
        else:
            # If run_tool is unchecked, disable (grey out) all dependent checkboxes
            for checkbox in dependent_checkboxes:
                checkbox.setEnabled(False)

    def on_bars_detection_changed(self, state):
        """Handle changes in bars detection checkbox with dependency logic"""
        # Skip updates while loading
        if self.is_loading:
            return
        
        # Update the config normally
        new_value = Qt.CheckState(state) == Qt.CheckState.Checked
        updates = {'tools': {'qct_parse': {'barsDetection': new_value}}}
        config_mgr.update_config('checks', updates)
        
        if Qt.CheckState(state) == Qt.CheckState.Checked:
            # If bars detection is checked, enable evaluate bars and thumbnail export
            self.evaluate_bars_cb.setEnabled(True)
            self.thumb_export_cb.setEnabled(True)
        else:
            # If bars detection is unchecked, disable and uncheck evaluate bars and thumbnail export
            self.evaluate_bars_cb.blockSignals(True)
            self.evaluate_bars_cb.setChecked(False)
            self.evaluate_bars_cb.setEnabled(False)
            self.evaluate_bars_cb.blockSignals(False)
            
            self.thumb_export_cb.blockSignals(True)
            self.thumb_export_cb.setChecked(False)
            self.thumb_export_cb.setEnabled(False)
            self.thumb_export_cb.blockSignals(False)
            
            # Update config for the dependent options that got unchecked
            dependent_updates = {
                'tools': {
                    'qct_parse': {
                        'evaluateBars': False,
                        'thumbExport': False
                    }
                }
            }
            config_mgr.update_config('checks', dependent_updates)
        
        # Check overall dependencies
        self.check_qct_dependencies()

    def on_evaluate_bars_changed(self, state):
        """Handle changes in evaluate bars checkbox with dependency logic"""
        # Skip updates while loading
        if self.is_loading:
            return
        
        # Update the config normally
        new_value = Qt.CheckState(state) == Qt.CheckState.Checked
        updates = {'tools': {'qct_parse': {'evaluateBars': new_value}}}
        config_mgr.update_config('checks', updates)
        
        # Check overall dependencies
        self.check_qct_dependencies()

    def check_qct_dependencies(self):
        """Check and enforce QCT Parse dependencies"""
        # If bars detection is off, then evaluate bars and thumbnail should already be disabled
        # This check is for the case where both bars detection and evaluate bars are unchecked
        if not self.bars_detection_cb.isChecked() and not self.evaluate_bars_cb.isChecked():
            # Uncheck run tool since no detection methods are active
            self.run_qctparse_cb.blockSignals(True)
            self.run_qctparse_cb.setChecked(False)
            self.run_qctparse_cb.blockSignals(False)
            
            # Disable all dependent options since run tool is now off
            self.bars_detection_cb.setEnabled(False)
            self.evaluate_bars_cb.setEnabled(False)
            self.thumb_export_cb.setEnabled(False)
            
            # Update config for run_tool change
            run_tool_updates = {
                'tools': {
                    'qct_parse': {
                        'run_tool': False
                    }
                }
            }
            config_mgr.update_config('checks', run_tool_updates)
    

    def on_tagname_changed(self, text):
        """Handle changes in tagname field"""
        # Skip updates while loading
        if self.is_loading:
            return

        updates = {'tools': {'qct_parse': {'tagname': text if text else None}}}
        config_mgr.update_config('checks', updates)

    def on_mediaconch_policy_changed(self, policy_name):
        """Handle selection of MediaConch policy"""
        # Skip updates while loading
        if self.is_loading:
            return
        
        if not self.is_loading and policy_name:
            config_mgr.update_config('checks', {
                'tools': {
                    'mediaconch': {
                        'mediaconch_policy': policy_name
                    }
                }
            })
            self.update_current_policy_display(policy_name)

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