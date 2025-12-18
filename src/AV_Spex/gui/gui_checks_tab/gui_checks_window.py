from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QLineEdit,
    QLabel, QComboBox, QPushButton, QScrollArea, QFileDialog, QMessageBox, QGridLayout
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
        
        # Add to layout
        outputs_layout.addWidget(self.access_file_cb)
        outputs_layout.addWidget(access_file_desc)
        outputs_layout.addWidget(self.report_cb)
        outputs_layout.addWidget(report_desc)
        
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
        validate_stream_desc = QLabel("Validates any embedded stream fixity, will embedded stream fixity if none is found")
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
        basic_tools = ['exiftool', 'ffprobe', 'mediainfo', 'mediatrace']
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
        
        # Tools section
        for tool, widgets in self.tool_widgets.items():
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
        
        # Fixity - now using booleans directly
        self.check_fixity_cb.setChecked(checks_config.fixity.check_fixity)
        self.validate_stream_cb.setChecked(checks_config.fixity.validate_stream_fixity)
        self.embed_stream_cb.setChecked(checks_config.fixity.embed_stream_fixity)
        self.output_fixity_cb.setChecked(checks_config.fixity.output_fixity)
        self.overwrite_stream_cb.setChecked(checks_config.fixity.overwrite_stream_fixity)
        
        # Tools - now using booleans directly
        for tool, widgets in self.tool_widgets.items():
            tool_config = getattr(checks_config.tools, tool)
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
        elif len(path) == 3:  # Handle nested structures like outputs.frame_analysis.field
            section = path[0]
            subsection = path[1]
            field = path[2]
            updates = {section: {subsection: {field: new_value}}}
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

    def on_frame_analysis_mode_changed(self, index):
        """Handle border detection mode changes"""
        if self.is_loading:
            return
        
        mode = self.border_mode_combo.itemData(index)
        
        # Show/hide appropriate parameter widgets
        if mode == "simple":
            self.simple_params_widget.setVisible(True)
            self.sophisticated_params_widget.setVisible(False)
            self.signalstats_widget.setVisible(False)
        else:  # sophisticated
            self.simple_params_widget.setVisible(False)
            self.sophisticated_params_widget.setVisible(True)
            self.signalstats_widget.setVisible(True)
        
        # Update config
        updates = {'outputs': {'frame_analysis': {'border_detection_mode': mode}}}
        config_mgr.update_config('checks', updates)

    def on_frame_analysis_param_changed(self, param_name, value):
        """Handle frame analysis parameter changes"""
        if self.is_loading:
            return
        
        # Convert to appropriate type
        if param_name in ['simple_border_pixels', 'sophisticated_threshold', 'sophisticated_edge_sample_width',
                        'sophisticated_sample_frames', 'sophisticated_padding', 'sophisticated_viz_time',
                        'sophisticated_search_window', 'brng_duration_limit', 'signalstats_start_time',
                        'signalstats_duration']:
            try:
                # Handle empty string case
                value = int(value) if value.strip() else 0
            except ValueError:
                return  # Don't update config if conversion fails
        
        updates = {'outputs': {'frame_analysis': {param_name: value}}}
        config_mgr.update_config('checks', updates)

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