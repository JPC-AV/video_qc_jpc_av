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

        self.setup_outputs_section(main_layout)
        self.setup_fixity_section(main_layout)
        self.setup_tools_section(main_layout)
        self.connect_signals()
        
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
    
    # Fixity Section
    def setup_fixity_section(self, main_layout):
        """Set up the fixity section with palette-aware styling"""
        theme_manager = ThemeManager.instance()
        
        # Creating a new fixity group
        self.fixity_group = QGroupBox("Fixity")
        theme_manager.style_groupbox(self.fixity_group, "top left")
        self.themed_group_boxes['fixity'] = self.fixity_group
        
        fixity_layout = QVBoxLayout()
        
        # Create checkboxes with descriptions on second line
        self.output_fixity_cb = QCheckBox("Output fixity")
        self.output_fixity_cb.setStyleSheet("font-weight: bold;")
        output_fixity_desc = QLabel("Generate whole file md5 checksum of .mkv files to .txt and .md5")
        output_fixity_desc.setIndent(20)
        
        self.check_fixity_cb = QCheckBox("Validate fixity")
        self.check_fixity_cb.setStyleSheet("font-weight: bold;")
        check_fixity_desc = QLabel("Validates fixity of .mkv files against a checksum file in the directory")
        check_fixity_desc.setIndent(20)
        
        self.embed_stream_cb = QCheckBox("Embed Stream fixity")
        self.embed_stream_cb.setStyleSheet("font-weight: bold;")
        embed_stream_desc = QLabel("Embeds video and audio stream checksums into .mkv tags")
        embed_stream_desc.setIndent(20)
        
        self.overwrite_stream_cb = QCheckBox("Overwrite Stream fixity")
        self.overwrite_stream_cb.setStyleSheet("font-weight: bold;")
        overwrite_stream_desc = QLabel("Embed stream checksums regardless if existing ones are found")
        overwrite_stream_desc.setIndent(20)
        
        self.validate_stream_cb = QCheckBox("Validate Stream fixity")
        self.validate_stream_cb.setStyleSheet("font-weight: bold;")
        validate_stream_desc = QLabel("Validates any embedded stream fixity, will embedded steam fixity if none is found")
        validate_stream_desc.setIndent(20)
        
        # Add to layout
        fixity_layout.addWidget(self.output_fixity_cb)
        fixity_layout.addWidget(output_fixity_desc)
        fixity_layout.addWidget(self.check_fixity_cb)
        fixity_layout.addWidget(check_fixity_desc)
        fixity_layout.addWidget(self.embed_stream_cb)
        fixity_layout.addWidget(embed_stream_desc)
        fixity_layout.addWidget(self.overwrite_stream_cb)
        fixity_layout.addWidget(overwrite_stream_desc)
        fixity_layout.addWidget(self.validate_stream_cb)
        fixity_layout.addWidget(validate_stream_desc)
        
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

        # QCT Parse section
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

        # Content Filter
        content_filter_label = QLabel("Content Detection")
        content_filter_label.setStyleSheet("font-weight: bold;")
        content_filter_desc = QLabel("Select type of content to detect in the video")
        self.content_filter_combo = QComboBox()
        self.content_filter_combo.addItem("Select options...", None)  # Store None as data

        # Create a mapping of display text to actual values
        content_filter_options = {
            "All Black Detection": "allBlack",
            "Static Content Detection": "static"
        }

        # Add items with display text and corresponding data value
        for display_text, value in content_filter_options.items():
            self.content_filter_combo.addItem(display_text, value)
                
        # Profile
        profile_label = QLabel("Profile")
        profile_label.setStyleSheet("font-weight: bold;")
        profile_desc = QLabel("Select tolerance profile for content analysis")
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("Select options...", None)  # Store None as data

        # Create a mapping of display text to actual values
        profile_options = {
            "Default Profile": "default",
            "High Tolerance": "highTolerance",
            "Medium Tolerance": "midTolerance",
            "Low Tolerance": "lowTolerance"
        }

        # Add items with display text and corresponding data value
        for display_text, value in profile_options.items():
            self.profile_combo.addItem(display_text, value)

        # Add all widgets to the qct layout
        qct_layout.addWidget(self.run_qctparse_cb)
        qct_layout.addWidget(run_qctparse_desc)
        qct_layout.addWidget(self.bars_detection_cb)
        qct_layout.addWidget(bars_detection_desc)
        qct_layout.addWidget(self.evaluate_bars_cb)
        qct_layout.addWidget(evaluate_bars_desc)
        qct_layout.addWidget(self.thumb_export_cb)
        qct_layout.addWidget(thumb_export_desc)
        qct_layout.addWidget(content_filter_label)
        qct_layout.addWidget(content_filter_desc)
        qct_layout.addWidget(self.content_filter_combo)
        qct_layout.addWidget(profile_label)
        qct_layout.addWidget(profile_desc)
        qct_layout.addWidget(self.profile_combo)

        self.qct_group.setLayout(qct_layout)
        tools_layout.addWidget(self.qct_group)
        
        # Tagname
        tagname_label = QLabel("Tag Name")
        tagname_label.setStyleSheet("font-weight: bold;")
        tagname_desc = QLabel("Input ad hoc tags using this format: YMIN, lt, 100 (tag name, lt or gt, number value)")
        self.tagname_input = QLineEdit()
        self.tagname_input.setPlaceholderText("None")
        qct_layout.addWidget(tagname_label)
        qct_layout.addWidget(tagname_desc)
        qct_layout.addWidget(self.tagname_input)
        
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
        self.content_filter_combo.currentIndexChanged.connect(
            lambda index: self.on_qct_combo_changed(self.content_filter_combo.itemData(index), 'contentFilter')
        )
        self.profile_combo.currentIndexChanged.connect(
            lambda index: self.on_qct_combo_changed(self.profile_combo.itemData(index), 'profile')
        )
        self.tagname_input.textChanged.connect(
            lambda text: self.on_tagname_changed(text)
        )

    def load_config_values(self):
        """Load current config values into UI elements"""
        # Set loading flag to True
        self.is_loading = True

        checks_config = config_mgr.get_config('checks', ChecksConfig)

        # Outputs
        self.access_file_cb.setChecked(checks_config.outputs.access_file.lower() == 'yes')
        self.report_cb.setChecked(checks_config.outputs.report.lower() == 'yes')
        self.qctools_ext_input.setText(checks_config.outputs.qctools_ext)
        
        # Fixity
        self.check_fixity_cb.setChecked(checks_config.fixity.check_fixity.lower() == 'yes')
        self.validate_stream_cb.setChecked(checks_config.fixity.validate_stream_fixity.lower() == 'yes')
        self.embed_stream_cb.setChecked(checks_config.fixity.embed_stream_fixity.lower() == 'yes')
        self.output_fixity_cb.setChecked(checks_config.fixity.output_fixity.lower() == 'yes')
        self.overwrite_stream_cb.setChecked(checks_config.fixity.overwrite_stream_fixity.lower() == 'yes')
        
        # Tools
        for tool, widgets in self.tool_widgets.items():
            tool_config = getattr(checks_config.tools, tool)
            if tool == 'qctools':
                widgets['run'].setChecked(tool_config.run_tool.lower() == 'yes')
            else:
                widgets['check'].setChecked(tool_config.check_tool.lower() == 'yes')
                widgets['run'].setChecked(tool_config.run_tool.lower() == 'yes')
        
        # MediaConch
        mediaconch = checks_config.tools.mediaconch
        self.run_mediaconch_cb.setChecked(mediaconch.run_mediaconch.lower() == 'yes')
        
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
        
        # QCT Parse
        qct = checks_config.tools.qct_parse
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
        # Skip updates while loading
        if self.is_loading:
            return

        new_value = 'yes' if Qt.CheckState(state) == Qt.CheckState.Checked else 'no'
        
        if path[0] == "tools" and len(path) > 2:
            tool_name = path[1]
            field = path[2]
            updates = {'tools': {tool_name: {field: new_value}}}
        else:
            section = path[0]
            field = path[1]
            updates = {section: {field: new_value}}
            
        config_mgr.update_config('checks', updates)

    def on_boolean_changed(self, state, path):
        """Handle changes in boolean checkboxes"""
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

    def on_qct_combo_changed(self, value, field):
        """Handle changes in QCT Parse combo boxes"""
        # Skip updates while loading
        if self.is_loading:
            return

        values = [value] if value is not None else []
        updates = {'tools': {'qct_parse': {field: values}}}
        config_mgr.update_config('checks', updates)

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