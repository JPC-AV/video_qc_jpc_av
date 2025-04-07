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

from ..gui.gui_theme_manager import ThemeManager, ThemeableMixin
from ..gui.gui_custom_filename import CustomFilenameDialog

from ..utils.config_manager import ConfigManager
from ..utils.config_setup import SpexConfig, ChecksConfig, FilenameConfig

from ..utils.log_setup import logger

from ..utils import config_edit

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

            selected_option = self.main_window.filename_profile_dropdown.itemText(index)
            
            if selected_option == "JPC Filename Profile":
                config_edit.apply_filename_profile(jpc_filename_profile)
                config_mgr.save_last_used_config('spex')
            elif selected_option == "Bowser Filename Profile":
                config_edit.apply_filename_profile(bowser_filename_profile)
                config_mgr.save_last_used_config('spex')
            elif selected_option.startswith("Custom ("):
                for profile_name in filename_config.filename_profiles.keys():
                    if selected_option == profile_name:
                        profile_class = filename_config.filename_profiles[profile_name]
                        config_edit.apply_filename_profile(profile_class)
                        config_mgr.save_last_used_config('spex')

        def on_signalflow_profile_changed(self, index):
            """Handle signal flow profile selection change."""
            selected_option = self.main_window.signalflow_profile_dropdown.itemText(index)
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
        spex_layout.addWidget(self.filename_section_group)
        
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
        self.exiftool_group = QGroupBox("Exiftool Values")
        theme_manager.style_groupbox(self.exiftool_group, "top center")
        self.main_window.spex_tab_group_boxes.append(self.exiftool_group)
        
        exiftool_layout = QVBoxLayout()
        
        exiftool_button = QPushButton("Open Section")
        exiftool_button.clicked.connect(
            lambda: self.open_new_window('Exiftool Values', 'exiftool_values')
        )
        
        exiftool_layout.addWidget(exiftool_button)
        self.exiftool_group.setLayout(exiftool_layout)
        vertical_layout.addWidget(self.exiftool_group)
        
        # Style the button
        theme_manager.style_buttons(exiftool_button)
        
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
        self.mediatrace_group = QGroupBox("Mediatrace Values")
        theme_manager.style_groupbox(self.mediatrace_group, "top center")
        self.main_window.spex_tab_group_boxes.append(self.mediatrace_group)
        
        mediatrace_layout = QVBoxLayout()
        
        # Signalflow profile dropdown
        signalflow_label = QLabel("Expected Signalflow profiles:")
        signalflow_label.setStyleSheet("font-weight: bold;")
        self.main_window.signalflow_profile_dropdown = QComboBox()
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
            
        self.main_window.signalflow_profile_dropdown.currentIndexChanged.connect(self.profile_handlers.on_signalflow_profile_changed)
        
        mediatrace_button = QPushButton("Open Section")
        mediatrace_button.clicked.connect(
            lambda: self.open_new_window('Mediatrace Values', 'mediatrace_values')
        )
        
        mediatrace_layout.addWidget(signalflow_label)
        mediatrace_layout.addWidget(self.main_window.signalflow_profile_dropdown)
        mediatrace_layout.addWidget(mediatrace_button)
        self.mediatrace_group.setLayout(mediatrace_layout)
        vertical_layout.addWidget(self.mediatrace_group)
        
        # Style the button
        theme_manager.style_buttons(self.mediatrace_group)
        
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
        self.filename_profile_dropdown.addItem("JPC file names")
        self.filename_profile_dropdown.addItem("Bowser file names")
        
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
    