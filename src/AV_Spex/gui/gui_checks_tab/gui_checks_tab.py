from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QScrollArea, QComboBox, QFrame,
    QMessageBox, QInputDialog, QPushButton
)
from PyQt6.QtCore import Qt

from AV_Spex.gui.gui_theme_manager import ThemeManager, ThemeableMixin
from AV_Spex.gui.gui_checks_tab.gui_checks_window import ChecksWindow
from AV_Spex.gui.gui_custom_profiles import ProfileSelectionDialog, CustomProfileDialog
from AV_Spex.utils.config_manager import ConfigManager
from AV_Spex.utils.config_setup import SpexConfig, ChecksConfig
from AV_Spex.utils import config_edit
from AV_Spex.utils.log_setup import logger

config_mgr = ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)

class ChecksTab(ThemeableMixin):
    """Checks tab with nested handler classes for hierarchical organization"""
    
    class ProfileHandlers:
        """Profile selection handlers for the Checks tab"""
        
        def __init__(self, parent_tab):
            self.parent_tab = parent_tab
            self.main_window = parent_tab.main_window
        
        def on_profile_selected(self, index):
            """Handle profile selection from dropdown."""
            selected_profile = self.main_window.checks_profile_dropdown.currentText()
            
            # Handle built-in profiles
            if selected_profile == "Step 1":
                profile = config_edit.profile_step1
                config_edit.apply_profile(profile)
            elif selected_profile == "Step 2":
                profile = config_edit.profile_step2
                config_edit.apply_profile(profile)
            elif selected_profile == "All Off":
                profile = config_edit.profile_allOff
                config_edit.apply_profile(profile)
            else:
                # Handle custom profiles - check if it's a custom profile
                if selected_profile.startswith("[Custom] "):
                    profile_name = selected_profile.replace("[Custom] ", "")
                    config_edit.apply_custom_profile(profile_name)
                else:
                    logger.warning(f"Unknown profile selected: {selected_profile}")
                    return

            try:
                logger.debug(f"Profile '{selected_profile}' applied successfully.")
                
                # Update Checks tab configuration
                self.main_window.config_widget.load_config_values()
                
                # Update Complex tab configuration if it exists
                if hasattr(self.main_window, 'complex_widget') and self.main_window.complex_widget:
                    if hasattr(self.main_window.complex_widget, 'load_config_values'):
                        self.main_window.complex_widget.load_config_values()
                    else:
                        logger.warning("Complex widget exists but doesn't have load_config_values method")
                        
            except Exception as e:
                logger.critical(f"Error applying profile: {e}")

        def on_manage_profiles_clicked(self):
            """Open the profile management dialog."""
            dialog = ProfileSelectionDialog(self.main_window)
            if dialog.exec():
                # Refresh the dropdown after dialog closes
                self.refresh_profile_dropdown()
                
                # Reload config values in case a profile was applied
                self.main_window.config_widget.load_config_values()
                
                # Also reload Complex tab configuration if it exists
                if hasattr(self.main_window, 'complex_widget') and self.main_window.complex_widget:
                    if hasattr(self.main_window.complex_widget, 'load_config_values'):
                        self.main_window.complex_widget.load_config_values()

        def on_create_profile_clicked(self):
            """Create a new profile from current configuration."""
            # Get available custom profiles to suggest a name
            custom_profiles = config_edit.get_available_custom_profiles()
            suggested_name = f"Custom Profile {len(custom_profiles) + 1}"
            
            name, ok = QInputDialog.getText(
                self.main_window, 
                'Create Profile', 
                'Enter a name for the new profile:',
                text=suggested_name
            )
            
            if ok and name:
                try:
                    # Create profile from current config
                    new_profile = config_edit.create_profile_from_current_config(
                        name.strip(), 
                        "Created from current configuration"
                    )
                    
                    # Save the profile
                    config_edit.save_custom_profile(new_profile)
                    
                    # Refresh dropdown to show new profile
                    self.refresh_profile_dropdown()
                    
                    # Set the dropdown to the new profile
                    self.main_window.checks_profile_dropdown.setCurrentText(f"[Custom] {name.strip()}")
                    
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.information(
                        self.main_window, 
                        "Success", 
                        f"Created profile '{name}' from current configuration"
                    )
                    
                except Exception as e:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.critical(self.main_window, "Error", f"Failed to create profile: {str(e)}")

        def refresh_profile_dropdown(self):
            """Refresh the profile dropdown with current profiles."""
            # Block signals while updating
            self.main_window.checks_profile_dropdown.blockSignals(True)
            
            # Store current selection
            current_text = self.main_window.checks_profile_dropdown.currentText()
            
            # Clear and repopulate
            self.main_window.checks_profile_dropdown.clear()
            
            # Add built-in profiles
            self.main_window.checks_profile_dropdown.addItem("Step 1")
            self.main_window.checks_profile_dropdown.addItem("Step 2") 
            self.main_window.checks_profile_dropdown.addItem("All Off")
            
            # Add custom profiles
            custom_profiles = config_edit.get_available_custom_profiles()
            for profile_name in custom_profiles:
                self.main_window.checks_profile_dropdown.addItem(f"[Custom] {profile_name}")
            
            # Restore selection if it still exists
            index = self.main_window.checks_profile_dropdown.findText(current_text)
            if index >= 0:
                self.main_window.checks_profile_dropdown.setCurrentIndex(index)
            else:
                # If previous selection doesn't exist, try to set based on current config
                self.set_dropdown_from_config()
            
            # Re-enable signals
            self.main_window.checks_profile_dropdown.blockSignals(False)

        def set_dropdown_from_config(self):
            """Set dropdown selection based on current configuration."""
            checks_config = config_mgr.get_config('checks', ChecksConfig)
            
            # Simple heuristic to determine which profile is closest to current config
            if checks_config.tools.exiftool.run_tool == "yes":
                self.main_window.checks_profile_dropdown.setCurrentText("Step 1")
            elif checks_config.tools.qctools.run_tool == "yes":
                self.main_window.checks_profile_dropdown.setCurrentText("Step 2")
            else:
                # Check if it matches any custom profile
                # For now, just default to first item
                if self.main_window.checks_profile_dropdown.count() > 0:
                    self.main_window.checks_profile_dropdown.setCurrentIndex(0)
    
    def __init__(self, main_window):
        self.main_window = main_window
        
        # Initialize tab UI elements as instance attributes
        self.profile_group = None
        self.config_group = None
        
        # Initialize nested handler classes
        self.profile_handlers = self.ProfileHandlers(self)
        
        # Initialize theme handling
        self.setup_theme_handling()

    def on_theme_changed(self, palette):
        """Handle theme changes for this tab"""
        theme_manager = ThemeManager.instance()
        
        # Update all group boxes
        for group_box in self.main_window.checks_tab_group_boxes:
            if group_box is not None:
                # Preserve the title position when refreshing style
                group_box_title_pos = group_box.property("title_position") or "top center"
                theme_manager.style_groupbox(group_box, group_box_title_pos)
        
        # Update the config widget if it exists
        if hasattr(self.main_window, 'config_widget') and self.main_window.config_widget:
            # If the config widget has its own theme handling, let it handle the change
            if hasattr(self.main_window.config_widget, 'on_theme_changed'):
                self.main_window.config_widget.on_theme_changed(palette)
    
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
        
        checks_profile_label = QLabel("Select a Checks profile:")
        checks_profile_label.setStyleSheet("font-weight: bold;")
        checks_profile_desc = QLabel("Choose from a preset Checks profile to apply a set of Checks to run on your Spex")
        
        # Profile selection layout
        profile_selection_layout = QHBoxLayout()
        
        self.main_window.checks_profile_dropdown = QComboBox()
        self.main_window.checks_profile_dropdown.addItem("Step 1")
        self.main_window.checks_profile_dropdown.addItem("Step 2")
        self.main_window.checks_profile_dropdown.addItem("All Off")
        
        # Add custom profiles to dropdown
        self.profile_handlers.refresh_profile_dropdown()
        
        # Set initial dropdown state
        self.profile_handlers.set_dropdown_from_config()

        self.main_window.checks_profile_dropdown.currentIndexChanged.connect(self.profile_handlers.on_profile_selected)
        
        # Set minimum width to make dropdown longer horizontally
        self.main_window.checks_profile_dropdown.setMinimumWidth(350)
        
        profile_selection_layout.addWidget(self.main_window.checks_profile_dropdown)
        profile_selection_layout.addStretch()  # Push buttons to the right
        
        # Profile management buttons
        manage_profiles_btn = QPushButton("Manage Profiles")
        manage_profiles_btn.clicked.connect(self.profile_handlers.on_manage_profiles_clicked)
        
        create_profile_btn = QPushButton("Save Current as Profile")
        create_profile_btn.clicked.connect(self.profile_handlers.on_create_profile_clicked)
        
        # Apply theme styling to the new buttons
        theme_manager.style_button(manage_profiles_btn)
        theme_manager.style_button(create_profile_btn)
        
        profile_selection_layout.addWidget(manage_profiles_btn)
        profile_selection_layout.addWidget(create_profile_btn)

        # Add widgets to layout
        profile_layout.addWidget(checks_profile_label)
        profile_layout.addWidget(checks_profile_desc)
        profile_layout.addLayout(profile_selection_layout)
        
        self.profile_group.setLayout(profile_layout)
        vertical_layout.addWidget(self.profile_group)

        # 3. Config section (unchanged)
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
        self.main_window.config_widget = ChecksWindow()
        config_scroll_area.setWidgetResizable(True)
        config_scroll_area.setWidget(self.main_window.config_widget)

        # Set a minimum width for the config widget to ensure legibility
        config_scroll_area.setMinimumWidth(450)

        config_layout.addWidget(config_scroll_area)
        self.config_group.setLayout(config_layout)
        vertical_layout.addWidget(self.config_group)

        # Add scroll area to main layout
        checks_layout.addWidget(main_scroll_area)