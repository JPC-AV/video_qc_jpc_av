from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QScrollArea, QComboBox, QFrame,
    QMessageBox
)
from PyQt6.QtCore import Qt

from ...gui.gui_theme_manager import ThemeManager
from ...gui.gui_checks_tab.gui_checks_window import ChecksWindow
from ...utils import config_edit
from ...utils.log_setup import logger

class ChecksTab:
    """Checks tab with nested handler classes for hierarchical organization"""
    
    class ProfileHandlers:
        """Profile selection handlers for the Checks tab"""
        
        def __init__(self, parent_tab):
            self.parent_tab = parent_tab
            self.main_window = parent_tab.main_window
        
        def on_profile_selected(self, index):
            """Handle profile selection from dropdown."""
            selected_profile = self.main_window.command_profile_dropdown.currentText()
            if selected_profile == "Step 1":
                profile = config_edit.profile_step1
            elif selected_profile == "Step 2":
                profile = config_edit.profile_step2
            elif selected_profile == "All Off":
                profile = config_edit.profile_allOff
            try:
                # Call the backend function to apply the selected profile
                config_edit.apply_profile(profile)
                logger.debug(f"Profile '{selected_profile}' applied successfully.")
                self.main_window.config_mgr.save_last_used_config('checks')
            except ValueError as e:
                logger.critical(f"Error: {e}")

            self.main_window.config_widget.load_config_values()
    
    def __init__(self, main_window):
        self.main_window = main_window
        
        # Initialize nested handler classes
        self.profile_handlers = self.ProfileHandlers(self)
    
    def setup_checks_tab(self):
        """Set up or update the Checks tab with theme-aware styling"""
        # Get the theme manager instance
        theme_manager = ThemeManager.instance()
        
        # If we're here, we're creating the tab from scratch or recreating it
        # Initialize or reset the group boxes collection
        self.main_window.checks_tab_group_boxes = []
    
        # Create the tab
        checks_tab = QWidget()
        checks_layout = QVBoxLayout(checks_tab)
        self.main_window.tabs.addTab(checks_tab, "Checks")

        # Scroll Area for Vertical Scrolling in "Checks" Tab
        main_scroll_area = QScrollArea(self.main_window)
        main_scroll_area.setWidgetResizable(True)
        main_widget = QWidget(self.main_window)
        main_scroll_area.setWidget(main_widget)

        # Vertical layout for the main content in "Checks"
        vertical_layout = QVBoxLayout(main_widget)

        # 1. Checks Profile section
        self.profile_group = QGroupBox("Checks Profiles")
        theme_manager.style_groupbox(self.profile_group, "top center")
        self.main_window.checks_tab_group_boxes.append(self.profile_group)
        
        profile_layout = QVBoxLayout()
        
        command_profile_label = QLabel("Select a Checks profile:")
        command_profile_label.setStyleSheet("font-weight: bold;")
        command_profile_desc = QLabel("Choose from a preset Checks profile to apply a set of Checks to run on your Spex")
        
        self.main_window.command_profile_dropdown = QComboBox()
        self.main_window.command_profile_dropdown.addItem("Step 1")
        self.main_window.command_profile_dropdown.addItem("Step 2")
        self.main_window.command_profile_dropdown.addItem("All Off")
        
        # Set initial dropdown state
        if self.main_window.checks_config.tools.exiftool.run_tool == "yes":
            self.main_window.command_profile_dropdown.setCurrentText("Step 1")
        elif self.main_window.checks_config.tools.exiftool.run_tool == "no":
            self.main_window.command_profile_dropdown.setCurrentText("Step 2")

        self.main_window.command_profile_dropdown.currentIndexChanged.connect(self.profile_handlers.on_profile_selected)

        # Add widgets to layout
        profile_layout.addWidget(command_profile_label)
        profile_layout.addWidget(command_profile_desc)
        profile_layout.addWidget(self.main_window.command_profile_dropdown)
        
        self.profile_group.setLayout(profile_layout)
        vertical_layout.addWidget(self.profile_group)

        # 3. Config section
        self.config_group = QGroupBox("Checks Options")
        theme_manager.style_groupbox(self.config_group, "top center")
        self.main_window.checks_tab_group_boxes.append(self.config_group)
        
        config_layout = QVBoxLayout()
        
        config_scroll_area = QScrollArea()
        config_scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
            }
        """)
        self.main_window.config_widget = ChecksWindow(config_mgr=self.main_window.config_mgr)
        config_scroll_area.setWidgetResizable(True)
        config_scroll_area.setWidget(self.main_window.config_widget)

        # Set a minimum width for the config widget to ensure legibility
        config_scroll_area.setMinimumWidth(450)

        config_layout.addWidget(config_scroll_area)
        self.config_group.setLayout(config_layout)
        vertical_layout.addWidget(self.config_group)

        # Add scroll area to main layout
        checks_layout.addWidget(main_scroll_area)