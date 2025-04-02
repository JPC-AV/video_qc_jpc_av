from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QScrollArea, QComboBox, QFrame
)
from PyQt6.QtCore import Qt

from ...gui.gui_theme_manager import ThemeManager
from ...gui.gui_checks_tab.gui_checks_window import ChecksWindow

class ChecksTabSetup:
    """Setup and handlers for the Checks tab"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def setup_checks_tab(self):
        """Set up or update the Checks tab with theme-aware styling"""
        # Get the theme manager instance
        theme_manager = ThemeManager.instance()
        
        # If we're here, we're creating the tab from scratch or recreating it
        # Initialize or reset the group boxes collection
        self.parent.checks_tab_group_boxes = []
    
        # Create the tab
        checks_tab = QWidget()
        checks_layout = QVBoxLayout(checks_tab)
        self.parent.tabs.addTab(checks_tab, "Checks")

        # Scroll Area for Vertical Scrolling in "Checks" Tab
        main_scroll_area = QScrollArea(self.parent)
        main_scroll_area.setWidgetResizable(True)
        main_widget = QWidget(self.parent)
        main_scroll_area.setWidget(main_widget)

        # Vertical layout for the main content in "Checks"
        vertical_layout = QVBoxLayout(main_widget)


        # 1. Checks Profile section
        self.profile_group = QGroupBox("Checks Profiles")
        theme_manager.style_groupbox(self.profile_group, "top center")
        self.parent.checks_tab_group_boxes.append(self.profile_group)
        
        profile_layout = QVBoxLayout()
        
        command_profile_label = QLabel("Select a Checks profile:")
        command_profile_label.setStyleSheet("font-weight: bold;")
        command_profile_desc = QLabel("Choose from a preset Checks profile to apply a set of Checks to run on your Spex")
        
        self.parent.command_profile_dropdown = QComboBox()
        self.parent.command_profile_dropdown.addItem("Step 1")
        self.parent.command_profile_dropdown.addItem("Step 2")
        self.parent.command_profile_dropdown.addItem("All Off")
        
        # Set initial dropdown state
        if self.parent.checks_config.tools.exiftool.run_tool == "yes":
            self.parent.command_profile_dropdown.setCurrentText("Step 1")
        elif self.parent.checks_config.tools.exiftool.run_tool == "no":
            self.parent.command_profile_dropdown.setCurrentText("Step 2")

        self.parent.command_profile_dropdown.currentIndexChanged.connect(self.parent.checks_profile_handlers.on_profile_selected)

        # Add widgets to layout
        profile_layout.addWidget(command_profile_label)
        profile_layout.addWidget(command_profile_desc)
        profile_layout.addWidget(self.parent.command_profile_dropdown)
        
        self.profile_group.setLayout(profile_layout)
        vertical_layout.addWidget(self.profile_group)

        # 3. Config section
        self.config_group = QGroupBox("Checks Options")
        theme_manager.style_groupbox(self.config_group, "top center")
        self.parent.checks_tab_group_boxes.append(self.config_group)
        
        config_layout = QVBoxLayout()
        
        config_scroll_area = QScrollArea()
        config_scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
            }
        """)
        self.parent.config_widget = ChecksWindow(config_mgr=self.parent.config_mgr)
        config_scroll_area.setWidgetResizable(True)
        config_scroll_area.setWidget(self.parent.config_widget)

        # Set a minimum width for the config widget to ensure legibility
        config_scroll_area.setMinimumWidth(450)

        config_layout.addWidget(config_scroll_area)
        self.config_group.setLayout(config_layout)
        vertical_layout.addWidget(self.config_group)

        # Add scroll area to main layout
        checks_layout.addWidget(main_scroll_area)