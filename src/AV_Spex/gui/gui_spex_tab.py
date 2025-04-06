from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QScrollArea, QComboBox, QPushButton,
    QFrame, QTextEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette

from ..gui.gui_theme_manager import ThemeManager, ThemeableMixin
from ..utils.config_manager import ConfigManager
from ..utils.config_setup import SpexConfig, ChecksConfig
from ..utils import config_edit
from ..utils.log_setup import logger

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
            """Handle filename profile selection change."""
            selected_option = self.main_window.filename_profile_dropdown.itemText(index)
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
            
            config_mgr.update_config('spex', updates)
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
        
        # 1. Filename section
        self.filename_group = QGroupBox("Filename Values")
        theme_manager.style_groupbox(self.filename_group, "top center")
        self.main_window.spex_tab_group_boxes.append(self.filename_group)
        
        filename_layout = QVBoxLayout()
        
        # Profile dropdown
        profile_label = QLabel("Expected filename profiles:")
        profile_label.setStyleSheet("font-weight: bold;")
        self.main_window.filename_profile_dropdown = QComboBox()
        self.main_window.filename_profile_dropdown.addItem("Bowser file names")
        self.main_window.filename_profile_dropdown.addItem("JPC file names")
        
        # Set initial state based on config
        if spex_config.filename_values.Collection == "JPC":
            self.main_window.filename_profile_dropdown.setCurrentText("JPC file names")
        elif spex_config.filename_values.Collection == "2012_79":
            self.main_window.filename_profile_dropdown.setCurrentText("Bowser file names")
        
        self.main_window.filename_profile_dropdown.currentIndexChanged.connect(self.profile_handlers.on_filename_profile_changed)
        
        # Open section button
        open_button = QPushButton("Open Section")
        open_button.clicked.connect(
            lambda: self.open_new_window('Filename Values', 'filename_values')
        )
        
        # Add widgets to layout
        filename_layout.addWidget(profile_label)
        filename_layout.addWidget(self.main_window.filename_profile_dropdown)
        filename_layout.addWidget(open_button)
        self.filename_group.setLayout(filename_layout)
        vertical_layout.addWidget(self.filename_group)
        
        # Style the button using theme manager
        theme_manager.style_buttons(open_button)
        
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
    
    def on_theme_changed(self, palette):
        """Handle theme changes for this tab"""
        theme_manager = ThemeManager.instance()
        
        # Update all group boxes
        for group_box in self.main_window.spex_tab_group_boxes:
            if group_box is not None:
                theme_manager.style_groupbox(group_box)
        
        # Update any buttons within the tab groups
        if hasattr(self, 'filename_group'):
            theme_manager.style_buttons(self.filename_group)
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
    