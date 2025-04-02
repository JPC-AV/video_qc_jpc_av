from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QScrollArea, QComboBox, QPushButton,
    QFrame, QTextEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette

from ...gui.gui_theme_manager import ThemeManager

from ...utils.config_manager import ConfigManager
from ...utils.config_setup import SpexConfig, ChecksConfig

config_mgr = ConfigManager()

class SpexTabSetup:
    """Setup and handlers for the Spex tab"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def setup_spex_tab(self):
        """Set up or update the Spex tab with theme-aware styling"""
        # Get the theme manager instance
        theme_manager = ThemeManager.instance()
        
        # Initialize or reset the group boxes collection
        self.parent.spex_tab_group_boxes = []
        
        # Create the tab widget
        spex_tab = QWidget()
        spex_layout = QVBoxLayout(spex_tab)
        self.parent.tabs.addTab(spex_tab, "Spex")
        
        # Create scroll area for vertical scrolling
        main_scroll_area = QScrollArea(self.parent)
        main_scroll_area.setWidgetResizable(True)
        main_widget = QWidget(self.parent)
        main_scroll_area.setWidget(main_widget)
        vertical_layout = QVBoxLayout(main_widget)
        
        # 1. Filename section
        self.filename_group = QGroupBox("Filename Values")
        theme_manager.style_groupbox(self.filename_group, "top center")
        self.parent.spex_tab_group_boxes.append(self.filename_group)
        
        filename_layout = QVBoxLayout()
        
        # Profile dropdown
        profile_label = QLabel("Expected filename profiles:")
        profile_label.setStyleSheet("font-weight: bold;")
        self.parent.filename_profile_dropdown = QComboBox()
        self.parent.filename_profile_dropdown.addItem("Bowser file names")
        self.parent.filename_profile_dropdown.addItem("JPC file names")
        
        # Set initial state based on config
        if self.parent.spex_config.filename_values.Collection == "JPC":
            self.parent.filename_profile_dropdown.setCurrentText("JPC file names")
        elif self.parent.spex_config.filename_values.Collection == "2012_79":
            self.parent.filename_profile_dropdown.setCurrentText("Bowser file names")
        
        self.parent.filename_profile_dropdown.currentIndexChanged.connect(self.parent.spex_profile_handlers.on_filename_profile_changed)
        
        # Open section button
        open_button = QPushButton("Open Section")
        open_button.clicked.connect(
            lambda: self.open_new_window('Filename Values', 'filename_values')
        )
        
        # Add widgets to layout
        filename_layout.addWidget(profile_label)
        filename_layout.addWidget(self.parent.filename_profile_dropdown)
        filename_layout.addWidget(open_button)
        self.filename_group.setLayout(filename_layout)
        vertical_layout.addWidget(self.filename_group)
        
        # Style the button using theme manager
        theme_manager.style_buttons(self.filename_group)
        
        # 2. MediaInfo section
        self.mediainfo_group = QGroupBox("MediaInfo Values")
        theme_manager.style_groupbox(self.mediainfo_group, "top center")
        self.parent.spex_tab_group_boxes.append(self.mediainfo_group)
        
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
        self.parent.spex_tab_group_boxes.append(self.exiftool_group)
        
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
        self.parent.spex_tab_group_boxes.append(self.ffprobe_group)
        
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
        self.parent.spex_tab_group_boxes.append(self.mediatrace_group)
        
        mediatrace_layout = QVBoxLayout()
        
        # Signalflow profile dropdown
        signalflow_label = QLabel("Expected Signalflow profiles:")
        signalflow_label.setStyleSheet("font-weight: bold;")
        self.parent.signalflow_profile_dropdown = QComboBox()
        self.parent.signalflow_profile_dropdown.addItem("JPC_AV_SVHS Signal Flow")
        self.parent.signalflow_profile_dropdown.addItem("BVH3100 Signal Flow")
        
        # Set initial state based on config
        encoder_settings = self.parent.spex_config.mediatrace_values.ENCODER_SETTINGS
        if isinstance(encoder_settings, dict):
            source_vtr = encoder_settings.get('Source_VTR', [])
        else:
            source_vtr = encoder_settings.Source_VTR
            
        if any("SVO5800" in vtr for vtr in source_vtr):
            self.parent.signalflow_profile_dropdown.setCurrentText("JPC_AV_SVHS Signal Flow")
        elif any("Sony BVH3100" in vtr for vtr in source_vtr):
            self.parent.signalflow_profile_dropdown.setCurrentText("BVH3100 Signal Flow")
            
        self.parent.signalflow_profile_dropdown.currentIndexChanged.connect(self.parent.spex_profile_handlers.on_signalflow_profile_changed)
        
        mediatrace_button = QPushButton("Open Section")
        mediatrace_button.clicked.connect(
            lambda: self.open_new_window('Mediatrace Values', 'mediatrace_values')
        )
        
        mediatrace_layout.addWidget(signalflow_label)
        mediatrace_layout.addWidget(self.parent.signalflow_profile_dropdown)
        mediatrace_layout.addWidget(mediatrace_button)
        self.mediatrace_group.setLayout(mediatrace_layout)
        vertical_layout.addWidget(self.mediatrace_group)
        
        # Style the button
        theme_manager.style_buttons(self.mediatrace_group)
        
        # 6. QCT section
        self.qct_group = QGroupBox("qct-parse Values")
        theme_manager.style_groupbox(self.qct_group, "top center")
        self.parent.spex_tab_group_boxes.append(self.qct_group)
        
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

    def open_new_window(self, title, config_attribute_name):
        """Open a new window to display configuration details."""
        self.checks_config = config_mgr.get_config('checks', ChecksConfig)
        self.spex_config = config_mgr.get_config('spex', SpexConfig)

        # Get the fresh config data using the attribute name
        config_data = getattr(self.spex_config, config_attribute_name)
        
        # Prepare the data using the helper function
        nested_dict = self.prepare_config_data(title, config_data)
        
        # Convert the dictionary to a string representation
        content_text = self.dict_to_string(nested_dict)

        # Create and configure the window
        self.parent.new_window = QWidget()
        self.parent.new_window.setWindowTitle(title)
        self.parent.new_window.setLayout(QVBoxLayout())
        
        scroll_area = QScrollArea(self.parent.new_window)
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
        
        self.parent.new_window.layout().addWidget(scroll_area)
        self.parent.new_window.resize(600, 400)
        self.parent.new_window.show()

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