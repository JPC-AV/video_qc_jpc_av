
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, 
    QScrollArea, QPushButton, QComboBox, QCheckBox, QGroupBox,
    QMessageBox, QDialog, QTextEdit, QGridLayout, QListWidget
)
from PyQt6.QtCore import Qt

from AV_Spex.utils.config_manager import ConfigManager
from AV_Spex.utils import config_edit
from AV_Spex.utils.config_setup import (
    ChecksProfile, OutputsConfig, FixityConfig, ToolsConfig,
    BasicToolConfig, QCToolsConfig, MediaConchConfig, QCTParseToolConfig
)
from AV_Spex.gui.gui_theme_manager import ThemeManager, ThemeableMixin

class CustomProfileDialog(QDialog, ThemeableMixin):
    def __init__(self, parent=None, edit_profile=None):
        super().__init__(parent)
        self.profile = None
        self.edit_mode = edit_profile is not None
        self.setWindowTitle("Custom Profile Editor" if self.edit_mode else "Create Custom Profile")
        self.setModal(True)

        # Add theme handling
        self.setup_theme_handling()

        # Set minimum size for the dialog
        self.setMinimumSize(700, 800)
        
        # Initialize layout
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Profile name and description
        self.setup_profile_info_section(layout)
        
        # Scrollable area for configuration sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.config_layout = QVBoxLayout(scroll_widget)
        self.config_layout.setSpacing(10)
        scroll.setWidget(scroll_widget)
        
        # Configuration sections
        self.setup_outputs_section()
        self.setup_fixity_section()
        self.setup_tools_section()
        
        # Set scroll area height
        scroll.setMinimumHeight(500)
        layout.addWidget(scroll)
        
        # Dialog buttons
        self.setup_dialog_buttons(layout)
        
        self.setLayout(layout)
        
        # Load existing profile if in edit mode
        if edit_profile:
            self.load_existing_profile(edit_profile)
    
    def _apply_initial_theme_styling(self):
        """Apply initial theme styling using ThemeManager."""
        theme_manager = ThemeManager.instance()
        
        # Style all group boxes
        for group_box in self.findChildren(QGroupBox):
            theme_manager.style_groupbox(group_box)
        
        # Style all buttons
        theme_manager.style_buttons(self)
        
        # Style all comboboxes
        theme_manager.style_comboboxes(self)
    
    def setup_profile_info_section(self, layout):
        """Setup the profile name and description section."""
        info_group = QGroupBox("Profile Information")
        info_layout = QVBoxLayout()
        
        # Profile name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Profile Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter profile name...")
        name_layout.addWidget(self.name_input)
        info_layout.addLayout(name_layout)
        
        # Profile description
        desc_layout = QVBoxLayout()
        desc_layout.addWidget(QLabel("Description (optional):"))
        self.description_input = QTextEdit()
        self.description_input.setMaximumHeight(60)
        self.description_input.setPlaceholderText("Enter profile description...")
        desc_layout.addWidget(self.description_input)
        info_layout.addLayout(desc_layout)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
    
    def setup_outputs_section(self):
        """Setup the outputs configuration section."""
        outputs_group = QGroupBox("Output Settings")
        outputs_layout = QGridLayout()
        
        # Access file
        outputs_layout.addWidget(QLabel("Access File:"), 0, 0)
        self.access_file_combo = QComboBox()
        self.access_file_combo.addItems(["no", "yes"])
        outputs_layout.addWidget(self.access_file_combo, 0, 1)
        
        # Report
        outputs_layout.addWidget(QLabel("Report:"), 1, 0)
        self.report_combo = QComboBox()
        self.report_combo.addItems(["no", "yes"])
        outputs_layout.addWidget(self.report_combo, 1, 1)
        
        # QCTools extension
        outputs_layout.addWidget(QLabel("QCTools Extension:"), 2, 0)
        self.qctools_ext_input = QLineEdit()
        self.qctools_ext_input.setText("qctools.xml.gz")
        outputs_layout.addWidget(self.qctools_ext_input, 2, 1)
        
        outputs_group.setLayout(outputs_layout)
        self.config_layout.addWidget(outputs_group)
    
    def setup_fixity_section(self):
        """Setup the fixity configuration section."""
        fixity_group = QGroupBox("Fixity Settings")
        fixity_layout = QGridLayout()
        
        # Create combo boxes for each fixity setting
        self.fixity_combos = {}
        fixity_options = [
            ("check_fixity", "Check Fixity:", 0),
            ("validate_stream_fixity", "Validate Stream Fixity:", 1),
            ("embed_stream_fixity", "Embed Stream Fixity:", 2),
            ("output_fixity", "Output Fixity:", 3),
            ("overwrite_stream_fixity", "Overwrite Stream Fixity:", 4)
        ]
        
        for setting, label, row in fixity_options:
            fixity_layout.addWidget(QLabel(label), row, 0)
            combo = QComboBox()
            combo.addItems(["no", "yes"])
            self.fixity_combos[setting] = combo
            fixity_layout.addWidget(combo, row, 1)
        
        fixity_group.setLayout(fixity_layout)
        self.config_layout.addWidget(fixity_group)
    
    def setup_tools_section(self):
        """Setup the tools configuration section."""
        tools_group = QGroupBox("Tools Settings")
        tools_layout = QVBoxLayout()
        
        # Basic tools (exiftool, ffprobe, mediainfo, mediatrace)
        basic_tools = ["exiftool", "ffprobe", "mediainfo", "mediatrace"]
        self.basic_tool_combos = {}
        
        for tool in basic_tools:
            tool_group = QGroupBox(tool.title())
            tool_layout = QGridLayout()
            
            # Check tool
            tool_layout.addWidget(QLabel("Check Tool:"), 0, 0)
            check_combo = QComboBox()
            check_combo.addItems(["no", "yes"])
            tool_layout.addWidget(check_combo, 0, 1)
            
            # Run tool
            tool_layout.addWidget(QLabel("Run Tool:"), 1, 0)
            run_combo = QComboBox()
            run_combo.addItems(["no", "yes"])
            tool_layout.addWidget(run_combo, 1, 1)
            
            self.basic_tool_combos[tool] = {
                'check_tool': check_combo,
                'run_tool': run_combo
            }
            
            tool_group.setLayout(tool_layout)
            tools_layout.addWidget(tool_group)
        
        # MediaConch
        self.setup_mediaconch_section(tools_layout)
        
        # QCTools
        self.setup_qctools_section(tools_layout)
        
        # QCT Parse
        self.setup_qct_parse_section(tools_layout)
        
        tools_group.setLayout(tools_layout)
        self.config_layout.addWidget(tools_group)
    
    def setup_mediaconch_section(self, parent_layout):
        """Setup MediaConch specific settings."""
        mediaconch_group = QGroupBox("MediaConch")
        mediaconch_layout = QGridLayout()
        
        # Policy dropdown (replacing the text input)
        mediaconch_layout.addWidget(QLabel("Policy:"), 0, 0)
        self.mediaconch_policy_combo = QComboBox()
        
        # Load available policies from config manager
        config_mgr = ConfigManager()
        available_policies = config_mgr.get_available_policies()
        self.mediaconch_policy_combo.addItems(available_policies)
        
        mediaconch_layout.addWidget(self.mediaconch_policy_combo, 0, 1)
        
        # Run MediaConch
        mediaconch_layout.addWidget(QLabel("Run MediaConch:"), 1, 0)
        self.mediaconch_run_combo = QComboBox()
        self.mediaconch_run_combo.addItems(["no", "yes"])
        mediaconch_layout.addWidget(self.mediaconch_run_combo, 1, 1)
        
        mediaconch_group.setLayout(mediaconch_layout)
        parent_layout.addWidget(mediaconch_group)
    
    def setup_qctools_section(self, parent_layout):
        """Setup QCTools specific settings."""
        qctools_group = QGroupBox("QCTools")
        qctools_layout = QGridLayout()
        
        # Run tool
        qctools_layout.addWidget(QLabel("Run Tool:"), 0, 0)
        self.qctools_run_combo = QComboBox()
        self.qctools_run_combo.addItems(["no", "yes"])
        qctools_layout.addWidget(self.qctools_run_combo, 0, 1)
        
        qctools_group.setLayout(qctools_layout)
        parent_layout.addWidget(qctools_group)
    
    def setup_qct_parse_section(self, parent_layout):
        """Setup QCT Parse specific settings."""
        qct_parse_group = QGroupBox("QCT Parse")
        qct_parse_layout = QGridLayout()
        
        # Run tool
        qct_parse_layout.addWidget(QLabel("Run Tool:"), 0, 0)
        self.qct_parse_run_combo = QComboBox()
        self.qct_parse_run_combo.addItems(["no", "yes"])
        qct_parse_layout.addWidget(self.qct_parse_run_combo, 0, 1)
        
        # Bars Detection
        qct_parse_layout.addWidget(QLabel("Bars Detection:"), 1, 0)
        self.bars_detection_check = QCheckBox()
        qct_parse_layout.addWidget(self.bars_detection_check, 1, 1)
        
        # Evaluate Bars
        qct_parse_layout.addWidget(QLabel("Evaluate Bars:"), 2, 0)
        self.evaluate_bars_check = QCheckBox()
        qct_parse_layout.addWidget(self.evaluate_bars_check, 2, 1)
        
        # Thumb Export
        qct_parse_layout.addWidget(QLabel("Thumb Export:"), 3, 0)
        self.thumb_export_check = QCheckBox()
        qct_parse_layout.addWidget(self.thumb_export_check, 3, 1)
        
        # Tag name
        qct_parse_layout.addWidget(QLabel("Tag Name:"), 4, 0)
        self.tagname_input = QLineEdit()
        self.tagname_input.setPlaceholderText("Optional tag name...")
        qct_parse_layout.addWidget(self.tagname_input, 4, 1)
        
        qct_parse_group.setLayout(qct_parse_layout)
        parent_layout.addWidget(qct_parse_group)
    
    def setup_dialog_buttons(self, layout):
        """Setup dialog action buttons."""
        button_layout = QHBoxLayout()
        
        # Load from current config button
        load_current_button = QPushButton("Load from Current Config")
        load_current_button.clicked.connect(self.load_from_current_config)
        button_layout.addWidget(load_current_button)
        
        button_layout.addStretch()  # Add stretch to push save/cancel to the right
        
        # Save and Cancel buttons
        save_button = QPushButton("Save Profile")
        save_button.clicked.connect(self.on_save_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def load_from_current_config(self):
        """Load settings from the current checks configuration."""
        try:
            current_config = config_edit.config_mgr.get_config('checks', config_edit.ChecksConfig)
            
            # Load outputs
            self.access_file_combo.setCurrentText(current_config.outputs.access_file)
            self.report_combo.setCurrentText(current_config.outputs.report)
            self.qctools_ext_input.setText(current_config.outputs.qctools_ext)
            
            # Load fixity
            self.fixity_combos['check_fixity'].setCurrentText(current_config.fixity.check_fixity)
            self.fixity_combos['validate_stream_fixity'].setCurrentText(current_config.fixity.validate_stream_fixity)
            self.fixity_combos['embed_stream_fixity'].setCurrentText(current_config.fixity.embed_stream_fixity)
            self.fixity_combos['output_fixity'].setCurrentText(current_config.fixity.output_fixity)
            self.fixity_combos['overwrite_stream_fixity'].setCurrentText(current_config.fixity.overwrite_stream_fixity)
            
            # Load basic tools
            for tool_name in self.basic_tool_combos:
                tool_config = getattr(current_config.tools, tool_name)
                self.basic_tool_combos[tool_name]['check_tool'].setCurrentText(tool_config.check_tool)
                self.basic_tool_combos[tool_name]['run_tool'].setCurrentText(tool_config.run_tool)
            
            # Load MediaConch - updated to use combo box
            if current_config.tools.mediaconch.mediaconch_policy:
                # Check if the policy exists in the dropdown, if so select it
                index = self.mediaconch_policy_combo.findText(current_config.tools.mediaconch.mediaconch_policy)
                if index >= 0:
                    self.mediaconch_policy_combo.setCurrentIndex(index)
                else:
                    # If policy doesn't exist in dropdown, add it and select it
                    self.mediaconch_policy_combo.addItem(current_config.tools.mediaconch.mediaconch_policy)
                    self.mediaconch_policy_combo.setCurrentText(current_config.tools.mediaconch.mediaconch_policy)
            
            self.mediaconch_run_combo.setCurrentText(current_config.tools.mediaconch.run_mediaconch)
            
            # Load QCTools
            self.qctools_run_combo.setCurrentText(current_config.tools.qctools.run_tool)
            
            # Load QCT Parse
            qct_config = current_config.tools.qct_parse
            self.qct_parse_run_combo.setCurrentText("yes" if qct_config.run_tool == "yes" else "no")
            self.bars_detection_check.setChecked(qct_config.barsDetection)
            self.evaluate_bars_check.setChecked(qct_config.evaluateBars)
            self.thumb_export_check.setChecked(qct_config.thumbExport)
            if qct_config.tagname:
                self.tagname_input.setText(qct_config.tagname)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load current config: {str(e)}")
    
    def load_existing_profile(self, profile):
        """Load an existing profile into the dialog."""
        # Load profile info
        self.name_input.setText(profile.name)
        self.description_input.setPlainText(profile.description)
        
        # Load outputs
        self.access_file_combo.setCurrentText(profile.outputs.access_file)
        self.report_combo.setCurrentText(profile.outputs.report)
        self.qctools_ext_input.setText(profile.outputs.qctools_ext)
        
        # Load fixity
        self.fixity_combos['check_fixity'].setCurrentText(profile.fixity.check_fixity)
        self.fixity_combos['validate_stream_fixity'].setCurrentText(profile.fixity.validate_stream_fixity)
        self.fixity_combos['embed_stream_fixity'].setCurrentText(profile.fixity.embed_stream_fixity)
        self.fixity_combos['output_fixity'].setCurrentText(profile.fixity.output_fixity)
        self.fixity_combos['overwrite_stream_fixity'].setCurrentText(profile.fixity.overwrite_stream_fixity)
        
        # Load basic tools
        for tool_name in self.basic_tool_combos:
            tool_config = getattr(profile.tools, tool_name)
            self.basic_tool_combos[tool_name]['check_tool'].setCurrentText(tool_config.check_tool)
            self.basic_tool_combos[tool_name]['run_tool'].setCurrentText(tool_config.run_tool)
        
        # Load MediaConch - updated to use combo box
        if profile.tools.mediaconch.mediaconch_policy:
            # Check if the policy exists in the dropdown, if so select it
            index = self.mediaconch_policy_combo.findText(profile.tools.mediaconch.mediaconch_policy)
            if index >= 0:
                self.mediaconch_policy_combo.setCurrentIndex(index)
            else:
                # If policy doesn't exist in dropdown, add it and select it
                self.mediaconch_policy_combo.addItem(profile.tools.mediaconch.mediaconch_policy)
                self.mediaconch_policy_combo.setCurrentText(profile.tools.mediaconch.mediaconch_policy)
        
        self.mediaconch_run_combo.setCurrentText(profile.tools.mediaconch.run_mediaconch)
        
        # Load QCTools
        self.qctools_run_combo.setCurrentText(profile.tools.qctools.run_tool)
        
        # Load QCT Parse
        qct_config = profile.tools.qct_parse
        self.qct_parse_run_combo.setCurrentText("yes" if qct_config.run_tool == "yes" else "no")
        self.bars_detection_check.setChecked(qct_config.barsDetection)
        self.evaluate_bars_check.setChecked(qct_config.evaluateBars)
        self.thumb_export_check.setChecked(qct_config.thumbExport)
        if qct_config.tagname:
            self.tagname_input.setText(qct_config.tagname)
    
    def get_profile_from_form(self):
        """Create a ChecksProfile from the form data."""
        # Validate required fields
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Profile name is required.")
            return None
        
        # Create outputs config
        outputs = OutputsConfig(
            access_file=self.access_file_combo.currentText(),
            report=self.report_combo.currentText(),
            qctools_ext=self.qctools_ext_input.text()
        )
        
        # Create fixity config
        fixity = FixityConfig(
            check_fixity=self.fixity_combos['check_fixity'].currentText(),
            validate_stream_fixity=self.fixity_combos['validate_stream_fixity'].currentText(),
            embed_stream_fixity=self.fixity_combos['embed_stream_fixity'].currentText(),
            output_fixity=self.fixity_combos['output_fixity'].currentText(),
            overwrite_stream_fixity=self.fixity_combos['overwrite_stream_fixity'].currentText()
        )
        
        # Create tools config
        tools = ToolsConfig(
            exiftool=BasicToolConfig(
                check_tool=self.basic_tool_combos['exiftool']['check_tool'].currentText(),
                run_tool=self.basic_tool_combos['exiftool']['run_tool'].currentText()
            ),
            ffprobe=BasicToolConfig(
                check_tool=self.basic_tool_combos['ffprobe']['check_tool'].currentText(),
                run_tool=self.basic_tool_combos['ffprobe']['run_tool'].currentText()
            ),
            mediaconch=MediaConchConfig(
                mediaconch_policy=self.mediaconch_policy_combo.currentText(),
                run_mediaconch=self.mediaconch_run_combo.currentText()
            ),
            mediainfo=BasicToolConfig(
                check_tool=self.basic_tool_combos['mediainfo']['check_tool'].currentText(),
                run_tool=self.basic_tool_combos['mediainfo']['run_tool'].currentText()
            ),
            mediatrace=BasicToolConfig(
                check_tool=self.basic_tool_combos['mediatrace']['check_tool'].currentText(),
                run_tool=self.basic_tool_combos['mediatrace']['run_tool'].currentText()
            ),
            qctools=QCToolsConfig(
                run_tool=self.qctools_run_combo.currentText()
            ),
            qct_parse=QCTParseToolConfig(
                run_tool=self.qct_parse_run_combo.currentText(),
                barsDetection=self.bars_detection_check.isChecked(),
                evaluateBars=self.evaluate_bars_check.isChecked(),
                contentFilter=[],  # Empty for now, could be extended
                profile=[],  # Empty for now, could be extended
                tagname=self.tagname_input.text() if self.tagname_input.text() else None,
                thumbExport=self.thumb_export_check.isChecked()
            )
        )
        
        # Create and return the profile
        return ChecksProfile(
            name=name,
            description=self.description_input.toPlainText().strip(),
            outputs=outputs,
            fixity=fixity,
            tools=tools
        )
    
    def on_save_clicked(self):
        """Handle save button click."""
        profile = self.get_profile_from_form()
        if profile:
            try:
                config_edit.save_custom_profile(profile)
                self.profile = profile
                self.accept()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save profile: {str(e)}")
    
    def get_profile(self):
        """Return the created/edited profile."""
        return self.profile
    
    def on_theme_changed(self, palette):
        """Apply theme changes to this dialog."""
        # Apply the palette directly
        self.setPalette(palette)
        
        # Get the theme manager
        theme_manager = ThemeManager.instance()
        
        # Update all group boxes
        for group_box in self.findChildren(QGroupBox):
            theme_manager.style_groupbox(group_box)
        
        # Update all buttons
        theme_manager.style_buttons(self)
        
        # Force repaint
        self.update()
    
    def closeEvent(self, event):
        """Clean up theme connections before closing."""
        self.cleanup_theme_handling()
        super().closeEvent(event)


class ProfileSelectionDialog(QDialog, ThemeableMixin):
    """Dialog for selecting and managing custom profiles."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_profile = None
        self.setWindowTitle("Manage Profiles")
        self.setModal(True)
        self.setup_theme_handling()
        self.setMinimumSize(400, 500)
        
        layout = QVBoxLayout()
        
        # Profile list
        layout.addWidget(QLabel("Available Profiles:"))
        self.profile_list = QListWidget()
        self.profile_list.itemDoubleClicked.connect(self.on_apply_profile)
        layout.addWidget(self.profile_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Left side buttons
        left_buttons = QHBoxLayout()
        create_button = QPushButton("Create New")
        create_button.clicked.connect(self.create_new_profile)
        edit_button = QPushButton("Edit")
        edit_button.clicked.connect(self.edit_profile)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_profile)
        
        left_buttons.addWidget(create_button)
        left_buttons.addWidget(edit_button)
        left_buttons.addWidget(delete_button)
        
        # Right side buttons
        right_buttons = QHBoxLayout()
        apply_button = QPushButton("Apply Profile")
        apply_button.clicked.connect(self.on_apply_profile)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        
        right_buttons.addWidget(apply_button)
        right_buttons.addWidget(close_button)
        
        button_layout.addLayout(left_buttons)
        button_layout.addStretch()
        button_layout.addLayout(right_buttons)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Apply initial theme styling
        self._apply_initial_theme_styling()
        
        self.refresh_profile_list()
    
    def _apply_initial_theme_styling(self):
        """Apply initial theme styling using ThemeManager."""
        theme_manager = ThemeManager.instance()
        
        # Style all buttons
        theme_manager.style_buttons(self)
    
    def refresh_profile_list(self):
        """Refresh the list of available profiles."""
        self.profile_list.clear()
        
        # Add built-in profiles
        builtin_profiles = ["Step 1 Profile", "Step 2 Profile", "All Off Profile"]
        for profile_name in builtin_profiles:
            self.profile_list.addItem(f"[Built-in] {profile_name}")
        
        # Add custom profiles
        custom_profiles = config_edit.get_available_custom_profiles()
        for profile_name in custom_profiles:
            self.profile_list.addItem(f"[Custom] {profile_name}")
    
    def create_new_profile(self):
        """Create a new custom profile."""
        dialog = CustomProfileDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_profile_list()
    
    def edit_profile(self):
        """Edit the selected custom profile."""
        current_item = self.profile_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a profile to edit.")
            return
        
        item_text = current_item.text()
        if not item_text.startswith("[Custom]"):
            QMessageBox.warning(self, "Cannot Edit", "Built-in profiles cannot be edited.")
            return
        
        profile_name = item_text.replace("[Custom] ", "")
        profile = config_edit.get_custom_profile(profile_name)
        
        if profile:
            dialog = CustomProfileDialog(self, edit_profile=profile)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.refresh_profile_list()
    
    def delete_profile(self):
        """Delete the selected custom profile."""
        current_item = self.profile_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a profile to delete.")
            return
        
        item_text = current_item.text()
        if not item_text.startswith("[Custom]"):
            QMessageBox.warning(self, "Cannot Delete", "Built-in profiles cannot be deleted.")
            return
        
        profile_name = item_text.replace("[Custom] ", "")
        
        reply = QMessageBox.question(
            self, "Confirm Delete", 
            f"Are you sure you want to delete the profile '{profile_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if config_edit.delete_custom_profile(profile_name):
                self.refresh_profile_list()
                QMessageBox.information(self, "Success", f"Profile '{profile_name}' deleted.")
    
    def on_apply_profile(self):
        """Apply the selected profile."""
        current_item = self.profile_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a profile to apply.")
            return
        
        item_text = current_item.text()
        
        try:
            if item_text.startswith("[Built-in]"):
                profile_name = item_text.replace("[Built-in] ", "")
                # Apply built-in profile
                if profile_name == "Step 1 Profile":
                    config_edit.apply_profile(config_edit.profile_step1)
                elif profile_name == "Step 2 Profile":
                    config_edit.apply_profile(config_edit.profile_step2)
                elif profile_name == "All Off Profile":
                    config_edit.apply_profile(config_edit.profile_allOff)
            else:
                profile_name = item_text.replace("[Custom] ", "")
                # Apply custom profile
                config_edit.apply_custom_profile(profile_name)
            
            # Update the GUI to reflect the new configuration
            main_window = self.parent()
            if main_window:
                # Refresh the config widget to show the new settings (handles most dropdowns)
                if hasattr(main_window, 'config_widget') and main_window.config_widget:
                    main_window.config_widget.load_config_values()
                
                # Handle the main profile dropdown separately
                if hasattr(main_window, 'checks_tab') and main_window.checks_tab:
                    main_window.checks_tab.profile_handlers.refresh_profile_dropdown()
                    
                    # Set the dropdown to show the applied profile
                    dropdown = main_window.checks_profile_dropdown
                    if item_text.startswith("[Built-in]"):
                        # For built-in profiles, use the simplified name
                        if profile_name == "Step 1 Profile":
                            dropdown.setCurrentText("Step 1")
                        elif profile_name == "Step 2 Profile":
                            dropdown.setCurrentText("Step 2")
                        elif profile_name == "All Off Profile":
                            dropdown.setCurrentText("All Off")
                    else:
                        # For custom profiles, use the [Custom] prefix format
                        dropdown.setCurrentText(f"[Custom] {profile_name}")
            
            QMessageBox.information(self, "Success", f"Applied profile: {profile_name}")
            self.selected_profile = profile_name
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply profile: {str(e)}")
    
    def on_theme_changed(self, palette):
        """Apply theme changes to this dialog."""
        # Apply the palette directly
        self.setPalette(palette)
        
        # Get the theme manager
        theme_manager = ThemeManager.instance()
        
        # Update all buttons
        theme_manager.style_buttons(self)
        
        # Force repaint
        self.update()
    
    def closeEvent(self, event):
        """Clean up theme connections before closing."""
        self.cleanup_theme_handling()
        super().closeEvent(event)