from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QLineEdit, QLabel, 
    QScrollArea, QFileDialog, QMenuBar, QListWidget, QPushButton, QFrame, QComboBox, QTabWidget,
    QTextEdit, QAbstractItemView, QInputDialog, QMessageBox, QProgressBar, QDialog
)
from PyQt6.QtCore import Qt, QSettings, QDir, QTimer
from PyQt6.QtGui import QPixmap, QPalette

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette

from dataclasses import asdict

from AV_Spex.gui.gui_theme_manager import ThemeManager, ThemeableMixin
from AV_Spex.gui.gui_custom_filename import CustomFilenameDialog
from AV_Spex.gui.gui_custom_signalflow import CustomSignalflowDialog
from AV_Spex.gui.gui_custom_exiftool import CustomExiftoolDialog 

from AV_Spex.utils.config_manager import ConfigManager
from AV_Spex.utils.config_setup import (
    SpexConfig, ChecksConfig, FilenameConfig, 
    SignalflowConfig, SignalflowProfile, ExiftoolConfig, ExiftoolProfile 
)
from AV_Spex.utils.log_setup import logger

from AV_Spex.utils import config_edit

config_mgr = ConfigManager()
spex_config = config_mgr.get_config('spex', SpexConfig)

class SpexTab(ThemeableMixin):
    """Spex tab with nested handler classes for hierarchical organization"""
    
    class ProfileHandlers:
        """Profile selection handlers for the Spex tab"""
        
        def __init__(self, parent_tab):
            self.parent_tab = parent_tab
            self.main_window = parent_tab.main_window
        
        def on_filename_profile_changed(self, index):
            filename_config = config_mgr.get_config("filename", FilenameConfig)
            jpc_filename_profile = filename_config.filename_profiles["JPC Filename Profile"]
            bowser_filename_profile = filename_config.filename_profiles["Bowser Filename Profile"]

            selected_option = self.parent_tab.filename_profile_dropdown.itemText(index)
            
            if selected_option == "JPC Filename Profile":
                config_edit.apply_filename_profile(jpc_filename_profile)
                config_mgr.save_config('spex', is_last_used=True)
            elif selected_option == "Bowser Filename Profile":
                config_edit.apply_filename_profile(bowser_filename_profile)
                config_mgr.save_config('spex', is_last_used=True)
            else:
                for profile_name in filename_config.filename_profiles.keys():
                    if selected_option == profile_name:
                        profile_class = filename_config.filename_profiles[profile_name]
                        config_edit.apply_filename_profile(profile_class)
                        config_mgr.save_config('spex', is_last_used=True)

        def on_signalflow_profile_changed(self, index):
            """Handle signal flow profile selection change."""
            signalflow_config = config_mgr.get_config("signalflow", SignalflowConfig)

            selected_option = self.main_window.signalflow_profile_dropdown.itemText(index)
            logger.debug(f"Selected signal flow profile: {selected_option}")

            if selected_option == "JPC_AV_SVHS Signal Flow":
                sn_config_changes = config_edit.JPC_AV_SVHS
            elif selected_option == "BVH3100 Signal Flow":
                sn_config_changes = config_edit.BVH3100
            else:
                # Handle custom profile
                if hasattr(signalflow_config, 'signalflow_profiles') and signalflow_config.signalflow_profiles:
                    for profile_name, profile_data in signalflow_config.signalflow_profiles.items():
                        if selected_option == profile_name:
                            config_edit.apply_signalflow_profile(profile_data)
                            config_mgr.save_config('spex', is_last_used=True)
                            return
                logger.error(f"Could not find profile: {selected_option}")
                return

            if sn_config_changes:
                config_edit.apply_signalflow_profile(sn_config_changes)
                config_mgr.save_config('spex', is_last_used=True)

        def on_exiftool_profile_changed(self, index):
            """Handle exiftool profile selection change"""
            try:
                exiftool_config = config_mgr.get_config("exiftool", ExiftoolConfig)
            except:
                return
            
            selected_option = self.parent_tab.exiftool_profile_dropdown.itemText(index)
            
            if selected_option != "Select a profile..." and hasattr(exiftool_config, 'exiftool_profiles'):
                if selected_option in exiftool_config.exiftool_profiles:
                    profile = exiftool_config.exiftool_profiles[selected_option]
                    from AV_Spex.utils import config_edit
                    config_edit.apply_exiftool_profile(profile)
                    config_mgr.save_config('spex', is_last_used=True)
    
    def __init__(self, main_window):
        self.main_window = main_window
        
        # Initialize nested handler classes
        self.profile_handlers = self.ProfileHandlers(self)

        # Initialize theme handling 
        self.setup_theme_handling()
    
    def setup_spex_tab(self):
        """Set up or update the Spex tab with theme-aware styling"""
        # Get the theme manager instance
        theme_manager = ThemeManager.instance()
        
        # Initialize or reset the group boxes collection
        self.main_window.spex_tab_group_boxes = []
        
        # Create the tab widget
        spex_tab = QWidget()
        spex_layout = QVBoxLayout(spex_tab)
        self.main_window.tabs.addTab(spex_tab, "Spex")
        
        # Create scroll area for vertical scrolling
        main_scroll_area = QScrollArea(self.main_window)
        main_scroll_area.setWidgetResizable(True)
        main_widget = QWidget(self.main_window)
        main_scroll_area.setWidget(main_widget)
        vertical_layout = QVBoxLayout(main_widget)

        self.filename_section_group = self.setup_filename_section()
        vertical_layout.addWidget(self.filename_section_group)
        
        # 2. MediaInfo section
        self.mediainfo_group = QGroupBox("MediaInfo Values")
        theme_manager.style_groupbox(self.mediainfo_group, "top center")
        self.main_window.spex_tab_group_boxes.append(self.mediainfo_group)
        
        mediainfo_layout = QVBoxLayout()
        
        mediainfo_button = QPushButton("Open Section")
        mediainfo_button.clicked.connect(
            lambda: self.open_new_window('MediaInfo Values', 'mediainfo_values')
        )
        
        mediainfo_layout.addWidget(mediainfo_button)
        self.mediainfo_group.setLayout(mediainfo_layout)
        vertical_layout.addWidget(self.mediainfo_group)
        
        # Style the button
        theme_manager.style_buttons(mediainfo_button)
        
        # 3. Exiftool section
        self.exiftool_group = self.setup_exiftool_section()
        vertical_layout.addWidget(self.exiftool_group)
        
        # 4. FFprobe section
        self.ffprobe_group = QGroupBox("FFprobe Values")
        theme_manager.style_groupbox(self.ffprobe_group, "top center")
        self.main_window.spex_tab_group_boxes.append(self.ffprobe_group)
        
        ffprobe_layout = QVBoxLayout()
        
        ffprobe_button = QPushButton("Open Section")
        ffprobe_button.clicked.connect(
            lambda: self.open_new_window('FFprobe Values', 'ffmpeg_values')
        )
        
        ffprobe_layout.addWidget(ffprobe_button)
        self.ffprobe_group.setLayout(ffprobe_layout)
        vertical_layout.addWidget(self.ffprobe_group)
        
        # Style the button
        theme_manager.style_buttons(ffprobe_button)
        
         # 5. Mediatrace section
        self.mediatrace_group = self.setup_mediatrace_section()
        vertical_layout.addWidget(self.mediatrace_group)
        
        # 6. QCT section
        self.qct_group = QGroupBox("qct-parse Values")
        theme_manager.style_groupbox(self.qct_group, "top center")
        self.main_window.spex_tab_group_boxes.append(self.qct_group)
        
        qct_layout = QVBoxLayout()
        
        qct_button = QPushButton("Open Section")
        qct_button.clicked.connect(
            lambda: self.open_new_window('Expected qct-parse options', 'qct_parse_values')
        )
        
        qct_layout.addWidget(qct_button)
        self.qct_group.setLayout(qct_layout)
        vertical_layout.addWidget(self.qct_group)
        
        # Style the button
        theme_manager.style_buttons(qct_button)
        
        # Add scroll area to main layout
        spex_layout.addWidget(main_scroll_area)

    def setup_exiftool_section(self):
        """Setup the Exiftool section with profiles"""
        # Load exiftool config
        try:
            self.exiftool_config = config_mgr.get_config("exiftool", ExiftoolConfig)
        except:
            # If config doesn't exist yet, create a default one
            self.exiftool_config = ExiftoolConfig()
        
        # Create and style the group box
        exiftool_group = QGroupBox("Exiftool Values")
        theme_manager = ThemeManager.instance()
        theme_manager.style_groupbox(exiftool_group, "top center")
        self.main_window.spex_tab_group_boxes.append(exiftool_group)
        
        # Create layout
        exiftool_layout = QVBoxLayout()
        
        # Add a dropdown menu for exiftool profiles
        exiftool_profile_label = QLabel("Expected Exiftool profiles:")
        exiftool_profile_label.setStyleSheet("font-weight: bold;")
        exiftool_layout.addWidget(exiftool_profile_label)
        
        self.exiftool_profile_dropdown = QComboBox()
        self.exiftool_profile_dropdown.addItem("Select a profile...")
        
        # Add any existing exiftool profiles from config
        if hasattr(self.exiftool_config, 'exiftool_profiles') and self.exiftool_config.exiftool_profiles:
            for profile_name in self.exiftool_config.exiftool_profiles.keys():
                self.exiftool_profile_dropdown.addItem(profile_name)
        
        # Set initial state based on current config - UPDATED LOGIC
        matched_profile = self._find_matching_exiftool_profile()
        if matched_profile:
            self.exiftool_profile_dropdown.setCurrentText(matched_profile)
        else:
            self.exiftool_profile_dropdown.setCurrentText("Select a profile...")
        
        self.exiftool_profile_dropdown.currentIndexChanged.connect(self.on_exiftool_profile_changed)
        exiftool_layout.addWidget(self.exiftool_profile_dropdown)
        
        # Store the layout as instance variable
        self.exiftool_section_layout = exiftool_layout
        
        # Add custom exiftool button
        self.add_custom_exiftool_button()
        self.add_edit_exiftool_button()
        
        # Open section button
        exiftool_button = QPushButton("Open Section")
        exiftool_button.clicked.connect(
            lambda: self.open_new_window('Exiftool Values', 'exiftool_values')
        )
        exiftool_layout.addWidget(exiftool_button)
        
        # Set the layout for the group
        exiftool_group.setLayout(exiftool_layout)
        exiftool_group.setMinimumHeight(225)
        
        # Style the buttons
        theme_manager.style_buttons(exiftool_layout)
        
        return exiftool_group

    def _find_matching_exiftool_profile(self):
        """Find which profile matches the current exiftool values"""
        if not hasattr(self.exiftool_config, 'exiftool_profiles') or not self.exiftool_config.exiftool_profiles:
            return None
        
        current_values = spex_config.exiftool_values
        
        # Compare current values with each profile
        for profile_name, profile in self.exiftool_config.exiftool_profiles.items():
            if self._compare_exiftool_values(current_values, profile):
                return profile_name
        
        return None

    def _compare_exiftool_values(self, current, profile):
        """Compare two exiftool value sets to see if they match"""
        from dataclasses import asdict
        
        # Convert both to dictionaries for comparison
        current_dict = asdict(current) if hasattr(current, '__dataclass_fields__') else current.__dict__
        profile_dict = asdict(profile) if hasattr(profile, '__dataclass_fields__') else profile.__dict__
        
        # Compare all fields
        for key in current_dict.keys():
            if key in profile_dict:
                if current_dict[key] != profile_dict[key]:
                    return False
        
        return True

    def add_custom_exiftool_button(self):
        """Add a button to create custom exiftool profiles"""
        custom_button = QPushButton("Create Custom Profile...")
        custom_button.clicked.connect(self.show_custom_exiftool_dialog)
        self.exiftool_section_layout.addWidget(custom_button)

    def add_edit_exiftool_button(self):
        """Add a button to edit existing exiftool profiles"""
        edit_button = QPushButton("Edit Selected Profile...")
        edit_button.clicked.connect(self.show_edit_exiftool_dialog)
        self.exiftool_section_layout.addWidget(edit_button)

    def show_custom_exiftool_dialog(self, edit_mode=False, profile_name=None):
        """Show the custom exiftool dialog"""
        from AV_Spex.gui.gui_custom_exiftool import CustomExiftoolDialog
        
        dialog = CustomExiftoolDialog(self.main_window, edit_mode=edit_mode, profile_name=profile_name)
        
        if edit_mode and profile_name:
            # Load the existing profile data
            try:
                exiftool_config = config_mgr.get_config("exiftool", ExiftoolConfig)
                if profile_name in exiftool_config.exiftool_profiles:
                    profile_data = exiftool_config.exiftool_profiles[profile_name]
                    dialog.load_profile_data(profile_data)
            except Exception as e:
                QMessageBox.warning(self.main_window, "Error", 
                                f"Error loading profile: {str(e)}")
                return
        else:
            # Load current exiftool values as defaults for new profile
            dialog.load_profile_data(spex_config.exiftool_values)
        
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted:
            profile_info = dialog.get_profile()
            if profile_info:
                try:
                    profile_name = profile_info['name']
                    profile_data = profile_info['data']
                    is_edit = profile_info.get('is_edit', False)
                    
                    # Get the ConfigManager instance
                    config_manager = ConfigManager()
                    config_manager.refresh_configs()
                    
                    # Get the exiftool configuration
                    try:
                        exiftool_config = config_manager.get_config('exiftool', ExiftoolConfig)
                    except:
                        exiftool_config = ExiftoolConfig()
                    
                    # Update the profiles
                    if not hasattr(exiftool_config, 'exiftool_profiles'):
                        exiftool_config.exiftool_profiles = {}
                    
                    # Add or update the profile
                    exiftool_config.exiftool_profiles[profile_name] = profile_data
                    
                    # Update the cached config
                    config_manager._configs['exiftool'] = exiftool_config
                    
                    # Save the updated config
                    config_manager.save_config('exiftool', is_last_used=True)
                    
                    # Update dropdown if it's a new profile
                    if not is_edit:
                        # Check if profile already exists in dropdown
                        found = False
                        for i in range(self.exiftool_profile_dropdown.count()):
                            if self.exiftool_profile_dropdown.itemText(i) == profile_name:
                                found = True
                                break
                        
                        if not found:
                            self.exiftool_profile_dropdown.addItem(profile_name)
                    
                    # Set as current selection
                    self.exiftool_profile_dropdown.setCurrentText(profile_name)
                    
                    # Apply the profile
                    from AV_Spex.utils import config_edit
                    config_edit.apply_exiftool_profile(profile_data)
                    config_manager.save_config('spex', is_last_used=True)
                    
                    action = "updated" if is_edit else "added"
                    logger.debug(f"Successfully {action} exiftool profile '{profile_name}'")
                    QMessageBox.information(self.main_window, "Success", 
                                        f"Profile '{profile_name}' {action} successfully!")
                    
                except Exception as e:
                    QMessageBox.warning(self.main_window, "Error", 
                                    f"Error saving profile: {str(e)}")

    def show_edit_exiftool_dialog(self):
        """Show the dialog to edit the currently selected exiftool profile"""
        selected_profile = self.exiftool_profile_dropdown.currentText()
        
        if selected_profile == "Select a profile...":
            QMessageBox.information(self.main_window, "No Profile Selected", 
                                "Please select a profile to edit from the dropdown.")
            return
        
        # Show the dialog in edit mode
        self.show_custom_exiftool_dialog(edit_mode=True, profile_name=selected_profile)
    
    def on_exiftool_profile_changed(self, index):
        """Handle exiftool profile selection change"""
        try:
            exiftool_config = config_mgr.get_config("exiftool", ExiftoolConfig)
        except:
            # If config doesn't exist, return
            return
        
        selected_option = self.exiftool_profile_dropdown.itemText(index)
        
        if selected_option != "Select a profile..." and hasattr(exiftool_config, 'exiftool_profiles'):
            if selected_option in exiftool_config.exiftool_profiles:
                profile = exiftool_config.exiftool_profiles[selected_option]
                from AV_Spex.utils import config_edit
                config_edit.apply_exiftool_profile(profile)
                config_mgr.save_config('spex', is_last_used=True)
                logger.debug(f"Applied exiftool profile: {selected_option}")

    def open_new_window(self, title, config_attribute_name):
        """Open a new window to display configuration details."""
        checks_config = config_mgr.get_config('checks', ChecksConfig)
        spex_config = config_mgr.get_config('spex', SpexConfig)

        # Get the fresh config data using the attribute name
        config_data = getattr(spex_config, config_attribute_name)
        
        # Prepare the data using the helper function
        nested_dict = self.prepare_config_data(title, config_data)
        
        # Convert the dictionary to a string representation
        content_text = self.dict_to_string(nested_dict)

        # Create and configure the window
        self.main_window.new_window = QWidget()
        self.main_window.new_window.setWindowTitle(title)
        self.main_window.new_window.setLayout(QVBoxLayout())
        
        scroll_area = QScrollArea(self.main_window.new_window)
        scroll_area.setWidgetResizable(True)
        
        content_widget = QTextEdit()
        content_widget.setPlainText(content_text)
        content_widget.setReadOnly(True)
        content_widget.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)

        # Let the theme manager style the text edit
        theme_manager = ThemeManager.instance()
        theme_manager.style_console_text(content_widget)
        
        scroll_area.setWidget(content_widget)
        
        self.main_window.new_window.layout().addWidget(scroll_area)
        self.main_window.new_window.resize(600, 400)
        self.main_window.new_window.show()

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
    
    def setup_filename_section(self):
        self.filename_config = config_mgr.get_config("filename", FilenameConfig)

        # Filename section
        filename_section_group = QGroupBox("Filename Values")
        theme_manager = ThemeManager.instance()
        theme_manager.style_groupbox(filename_section_group, "top center")
        self.main_window.spex_tab_group_boxes.append(filename_section_group)
        
        filename_section_layout = QVBoxLayout()
        
        # Add a dropdown menu for command profiles
        filenames_profile_label = QLabel("Expected filename profiles:")
        filenames_profile_label.setStyleSheet("font-weight: bold;")
        filename_section_layout.addWidget(filenames_profile_label)

        self.filename_profile_dropdown = QComboBox()
        self.filename_profile_dropdown.addItem("Select a profile...")
        
        # Add any custom filename profiles from the config
        if hasattr(self.filename_config, 'filename_profiles') and self.filename_config.filename_profiles:
            for profile_name in self.filename_config.filename_profiles.keys():
                self.filename_profile_dropdown.addItem(profile_name)

        # Set initial state
        if spex_config.filename_values.fn_sections["section1"].value == "JPC":
            self.filename_profile_dropdown.setCurrentText("JPC Filename Profile")
        elif spex_config.filename_values.fn_sections["section1"].value == "2012":
            self.filename_profile_dropdown.setCurrentText("Bowser Filename Profile")
        else:
             self.filename_profile_dropdown.setCurrentText("Select a profile...")
            
        self.filename_profile_dropdown.currentIndexChanged.connect(self.profile_handlers.on_filename_profile_changed)
        filename_section_layout.addWidget(self.filename_profile_dropdown)

        # Store the layout as an instance variable so it can be accessed by add_custom_filename_button
        self.filename_section_layout = filename_section_layout
        
        # Add the custom filename button
        self.add_custom_filename_button()
        
        # Open section button
        button = QPushButton("Open Section")
        button.clicked.connect(
            lambda: self.open_new_window('Filename Values', 'filename_values')
        )
        filename_section_layout.addWidget(button)
        
        filename_section_group.setLayout(filename_section_layout)
        filename_section_group.setMinimumHeight(200)
        
        # Style the button using theme manager
        theme_manager = ThemeManager.instance()
        theme_manager.style_buttons(self.filename_section_layout)

        return filename_section_group
    
    def add_custom_filename_button(self):
        custom_button = QPushButton("Create Custom Pattern...")
        custom_button.clicked.connect(self.show_custom_filename_dialog)
        # Add to the filename section layout that's already defined
        self.filename_section_layout.addWidget(custom_button)

    def show_custom_filename_dialog(self):
        dialog = CustomFilenameDialog(self.main_window)
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
                        # Add to dropdown UI
                        self.filename_profile_dropdown.addItem(custom_name)
                        self.filename_profile_dropdown.setCurrentText(custom_name)
                        
                        # Get the ConfigManager instance and refresh configs
                        config_manager = ConfigManager()
                        config_manager.refresh_configs()
                        
                        # Get the current filename configuration
                        filename_config = config_manager.get_config('filename', FilenameConfig)
                        
                        # Create an updated dictionary of profiles
                        updated_profiles = dict(filename_config.filename_profiles)
                        updated_profiles[custom_name] = pattern
                        
                        # Update the configuration
                        filename_config.filename_profiles = updated_profiles
                        
                        # Update the cached config directly
                        config_manager._configs['filename'] = filename_config
                        
                        # Save the updated config to disk
                        config_manager.save_config('filename', is_last_used=True)
                        
                        logger.debug(f"Added custom filename pattern '{custom_name}' to configuration")
                        
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Error adding custom pattern to dropdown: {str(e)}")        
    
    def setup_mediatrace_section(self):
        """Setup the Mediatrace section with signalflow profiles"""
        # Create and style the group box
        mediatrace_group = QGroupBox("Mediatrace Values")
        theme_manager = ThemeManager.instance()
        theme_manager.style_groupbox(mediatrace_group, "top center")
        self.main_window.spex_tab_group_boxes.append(mediatrace_group)
        
        # Create layout
        mediatrace_layout = QVBoxLayout()
        
        # Add a dropdown menu for signal flow profiles
        signalflow_label = QLabel("Expected Signalflow profiles:")
        signalflow_label.setStyleSheet("font-weight: bold;")
        mediatrace_layout.addWidget(signalflow_label)

        # Create the dropdown
        self.main_window.signalflow_profile_dropdown = QComboBox()
        self.main_window.signalflow_profile_dropdown.addItem("Select a profile...")
        
        # Try to load profiles from the dedicated signalflow config
        try:
            signalflow_config = config_mgr.get_config('signalflow', SignalflowConfig)
            if hasattr(signalflow_config, 'signalflow_profiles') and signalflow_config.signalflow_profiles:
                for profile_name in signalflow_config.signalflow_profiles.keys():
                    self.main_window.signalflow_profile_dropdown.addItem(profile_name)
        except Exception as e:
            logger.warning(f"Could not load signalflow config: {e}")
            # Fall back to hardcoded profiles
            self.main_window.signalflow_profile_dropdown.addItem("JPC_AV_SVHS Signal Flow")
            self.main_window.signalflow_profile_dropdown.addItem("BVH3100 Signal Flow")
        
        # Set initial state based on config
        encoder_settings = spex_config.mediatrace_values.ENCODER_SETTINGS
        if isinstance(encoder_settings, dict):
            source_vtr = encoder_settings.get('Source_VTR', [])
        else:
            source_vtr = encoder_settings.Source_VTR
            
        if any("SVO5800" in vtr for vtr in source_vtr):
            self.main_window.signalflow_profile_dropdown.setCurrentText("JPC_AV_SVHS Signal Flow")
        elif any("Sony BVH3100" in vtr for vtr in source_vtr):
            self.main_window.signalflow_profile_dropdown.setCurrentText("BVH3100 Signal Flow")
        else:
            # Check if it matches any custom profile
            found_match = False
            if hasattr(spex_config, 'signalflow_profiles'):
                for profile_name, profile_data in spex_config.signalflow_profiles.items():
                    if 'Source_VTR' in profile_data and source_vtr and any(vtr in str(source_vtr) for vtr in profile_data['Source_VTR']):
                        self.main_window.signalflow_profile_dropdown.setCurrentText(profile_name)
                        found_match = True
                        break
            
            if not found_match:
                self.main_window.signalflow_profile_dropdown.setCurrentText("Select a profile...")
                
        # Connect the dropdown to the handler
        self.main_window.signalflow_profile_dropdown.currentIndexChanged.connect(
            self.profile_handlers.on_signalflow_profile_changed
        )
        mediatrace_layout.addWidget(self.main_window.signalflow_profile_dropdown)
        
        # Store the layout as an instance variable so it can be accessed by other methods
        self.mediatrace_section_layout = mediatrace_layout
        
        # Add the custom signalflow button
        self.add_custom_signalflow_button()
        
        # Open section button
        mediatrace_button = QPushButton("Open Section")
        mediatrace_button.clicked.connect(
            lambda: self.open_new_window('Mediatrace Values', 'mediatrace_values')
        )
        mediatrace_layout.addWidget(mediatrace_button)
        
        # Set the layout for the group
        mediatrace_group.setLayout(mediatrace_layout)
        mediatrace_group.setMinimumHeight(200)
        
        # Style the buttons using theme manager
        theme_manager.style_buttons(mediatrace_layout)
        
        return mediatrace_group
    
    def add_custom_signalflow_button(self):
        """Add a button to create custom signal flow profiles"""
        custom_button = QPushButton("Create Custom Signalflow...")
        custom_button.clicked.connect(self.show_custom_signalflow_dialog)
        # Add to the mediatrace section layout that's already defined
        self.mediatrace_section_layout.addWidget(custom_button)
        
    def show_custom_signalflow_dialog(self):
        """Show the custom signal flow dialog"""
        dialog = CustomSignalflowDialog(self.main_window)
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted:
            profile = dialog.profile
            if profile:
                try:
                    # Get the profile name for the dropdown
                    profile_name = f"Custom ({profile['name']})"
                    
                    # Check if this profile already exists in the dropdown
                    found = False
                    for i in range(self.main_window.signalflow_profile_dropdown.count()):
                        if self.main_window.signalflow_profile_dropdown.itemText(i) == profile_name:
                            found = True
                            break
                    
                    # Only add if it's not already in the dropdown
                    if not found:
                        # Add to dropdown UI
                        self.main_window.signalflow_profile_dropdown.addItem(profile_name)
                        self.main_window.signalflow_profile_dropdown.setCurrentText(profile_name)
                        
                        # Get the ConfigManager instance and refresh configs
                        config_manager = ConfigManager()
                        config_manager.refresh_configs()
                        
                        # Try to save to the dedicated signalflow config
                        try:
                            # Get the signalflow configuration
                            signalflow_config = config_manager.get_config('signalflow', SignalflowConfig)
                            
                            # Create or update the signalflow_profiles dictionary
                            if not hasattr(signalflow_config, 'signalflow_profiles'):
                                signalflow_config.signalflow_profiles = {}
                            
                            # Create an updated dictionary of profiles
                            updated_profiles = dict(signalflow_config.signalflow_profiles)
                            
                            # Create a SignalflowProfile object
                            new_profile = SignalflowProfile(
                                name=profile['name'],
                                Source_VTR=profile['Source_VTR'],
                                TBC_Framesync=profile.get('TBC_Framesync', []),
                                ADC=profile.get('ADC', []),
                                Capture_Device=profile['Capture_Device'],
                                Computer=profile['Computer']
                            )
                            
                            # Add the new profile
                            updated_profiles[profile_name] = new_profile
                            
                            # Update the configuration
                            signalflow_config.signalflow_profiles = updated_profiles
                            
                            # Update the cached config directly
                            config_manager._configs['signalflow'] = signalflow_config
                            
                            # Save the updated config to disk
                            config_manager.save_config('signalflow', is_last_used=True)
                            
                            logger.debug(f"Added custom signal flow profile '{profile_name}' to signalflow configuration")
                            
                        except Exception as e:
                            logger.warning(f"Could not save to signalflow config: {e}")
                            # Fall back to storing in SpexConfig (legacy method)
                            spex_config = config_manager.get_config('spex', SpexConfig)
                            
                            # Create or update the signalflow_profiles dictionary
                            if not hasattr(spex_config, 'signalflow_profiles'):
                                spex_config.signalflow_profiles = {}
                            
                            # Create an updated dictionary of profiles
                            updated_profiles = dict(spex_config.signalflow_profiles)
                            updated_profiles[profile_name] = profile
                            
                            # Update the configuration
                            spex_config.signalflow_profiles = updated_profiles
                            
                            # Update the cached config directly
                            config_manager._configs['spex'] = spex_config
                            
                            # Save the updated config to disk
                            config_manager.save_config('spex', is_last_used=True)
                            
                            logger.debug(f"Added custom signal flow profile '{profile_name}' to spex configuration (fallback)")
                        
                        # Apply the new profile
                        config_edit.apply_signalflow_profile(profile)
                        
                except Exception as e:
                    QMessageBox.warning(self.main_window, "Error", f"Error adding custom profile to dropdown: {str(e)}")
    
    def on_theme_changed(self, palette):
        """Handle theme changes for this tab"""
        theme_manager = ThemeManager.instance()
        
        # Update all group boxes
        for group_box in self.main_window.spex_tab_group_boxes:
            if group_box is not None:
                theme_manager.style_groupbox(group_box)
        
        # Update any buttons within the tab groups
        if hasattr(self, 'filename_section_group'):
            theme_manager.style_buttons(self.filename_section_group)
        if hasattr(self, 'mediainfo_group'):
            theme_manager.style_buttons(self.mediainfo_group)
        if hasattr(self, 'exiftool_group'):
            theme_manager.style_buttons(self.exiftool_group)
        if hasattr(self, 'ffprobe_group'):
            theme_manager.style_buttons(self.ffprobe_group)
        if hasattr(self, 'mediatrace_group'):
            theme_manager.style_buttons(self.mediatrace_group)
        if hasattr(self, 'qct_group'):
            theme_manager.style_buttons(self.qct_group)
        if hasattr(self, 'exiftool_group'):
            theme_manager.style_buttons(self.exiftool_group)
    