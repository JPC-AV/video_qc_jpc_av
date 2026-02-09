import json

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QScrollArea, QComboBox, QFrame,
    QMessageBox, QInputDialog, QPushButton, QFileDialog
)
from PyQt6.QtCore import Qt

from AV_Spex.gui.gui_theme_manager import ThemeManager, ThemeableMixin
from AV_Spex.gui.gui_checks_tab.gui_checks_window import ChecksWindow
from AV_Spex.gui.gui_custom_profiles import ProfileSelectionDialog, CustomProfileDialog
from AV_Spex.utils.config_manager import ConfigManager
from AV_Spex.utils.config_setup import SpexConfig, ChecksConfig
from AV_Spex.utils.config_io import ConfigIO
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
                self.main_window.config_widget.load_config_values()
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
                    
                    QMessageBox.information(
                        self.main_window, 
                        "Success", 
                        f"Created profile '{name}' from current configuration"
                    )
                    
                except Exception as e:
                    QMessageBox.critical(self.main_window, "Error", f"Failed to create profile: {str(e)}")

        def on_export_profile_clicked(self):
            """Export the currently selected custom profile to a JSON file."""
            selected_text = self.main_window.checks_profile_dropdown.currentText()
            
            # Only allow exporting custom profiles
            if not selected_text.startswith("[Custom] "):
                QMessageBox.information(
                    self.main_window,
                    "Export Profile",
                    "Only custom profiles can be exported.\n"
                    "Select a custom profile from the dropdown first."
                )
                return
            
            profile_name = selected_text.replace("[Custom] ", "")
            
            # Open save dialog
            safe_name = profile_name.replace(' ', '_').replace('/', '_')
            suggested_filename = f"av_spex_profile_{safe_name}.json"
            
            filepath, _ = QFileDialog.getSaveFileName(
                self.main_window,
                "Export Profile",
                suggested_filename,
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not filepath:
                return  # User cancelled
            
            try:
                config_io = ConfigIO(config_mgr)
                result = config_io.save_single_profile(
                    'profiles_checks', profile_name, filepath
                )
                
                if result:
                    QMessageBox.information(
                        self.main_window,
                        "Export Successful",
                        f"Profile '{profile_name}' exported to:\n{result}"
                    )
                else:
                    QMessageBox.warning(
                        self.main_window,
                        "Export Failed",
                        f"Profile '{profile_name}' could not be found for export."
                    )
            except Exception as e:
                logger.error(f"Error exporting profile: {e}")
                QMessageBox.critical(
                    self.main_window,
                    "Export Error",
                    f"Failed to export profile:\n{str(e)}"
                )

        def on_import_profile_clicked(self):
            """Import custom profile(s) from a JSON file."""
            filepath, _ = QFileDialog.getOpenFileName(
                self.main_window,
                "Import Profile",
                "",
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not filepath:
                return  # User cancelled
            
            try:
                config_io = ConfigIO(config_mgr)
                import_results = config_io.import_configs(filepath)
                
                # Refresh the dropdown to show any newly imported profiles
                self.refresh_profile_dropdown()
                
                # Reload config values in case the active config was updated
                self.main_window.config_widget.load_config_values()
                
                # Build user-facing summary
                renamed = import_results.get('renamed_profiles', [])
                
                if renamed:
                    # Show notification about renamed profiles
                    self._show_rename_notification(renamed)
                else:
                    QMessageBox.information(
                        self.main_window,
                        "Import Successful",
                        f"Profile(s) imported successfully from:\n{filepath}"
                    )
                    
            except json.JSONDecodeError:
                QMessageBox.critical(
                    self.main_window,
                    "Import Error",
                    "The selected file is not valid JSON.\n"
                    "Please select a valid AV Spex profile or config export file."
                )
            except Exception as e:
                logger.error(f"Error importing profile: {e}")
                QMessageBox.critical(
                    self.main_window,
                    "Import Error",
                    f"Failed to import profile:\n{str(e)}"
                )

        def _show_rename_notification(self, renamed_profiles):
            """
            Show a notification dialog listing profiles that were renamed
            during import due to name collisions.
            
            Args:
                renamed_profiles: List of (original_name, new_name) tuples
            """
            rename_lines = []
            for original, renamed in renamed_profiles:
                rename_lines.append(f'  "{original}"  →  "{renamed}"')
            
            rename_text = "\n".join(rename_lines)
            
            msg = QMessageBox(self.main_window)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("Import Successful — Profiles Renamed")
            msg.setText(
                "Some imported profiles were renamed to avoid "
                "conflicts with existing profiles:"
            )
            msg.setInformativeText(rename_text)
            msg.setDetailedText(
                "When an imported profile has the same name as an existing profile, "
                "AV Spex adds an '(imported)' suffix to the imported profile to "
                "preserve both versions.\n\n"
                "You can rename imported profiles through Manage Profiles."
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()

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
            
            # Update export button enabled state
            self._update_export_button_state()

        def set_dropdown_from_config(self):
            """Set dropdown selection based on current configuration."""
            checks_config = config_mgr.get_config('checks', ChecksConfig)
            
            # Simple heuristic to determine which profile is closest to current config
            if checks_config.tools.exiftool.run_tool:  
                self.main_window.checks_profile_dropdown.setCurrentText("Step 1")
            elif checks_config.tools.qctools.run_tool:  
                self.main_window.checks_profile_dropdown.setCurrentText("Step 2")
            else:
                # Check if it matches any custom profile
                # For now, just default to first item
                if self.main_window.checks_profile_dropdown.count() > 0:
                    self.main_window.checks_profile_dropdown.setCurrentIndex(0)

        def _update_export_button_state(self):
            """Enable/disable the export button based on whether a custom profile is selected."""
            if hasattr(self.main_window, 'export_profile_btn'):
                selected = self.main_window.checks_profile_dropdown.currentText()
                is_custom = selected.startswith("[Custom] ")
                self.main_window.export_profile_btn.setEnabled(is_custom)
    
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
        
        # Profile selection layout (Row 1: dropdown + manage/create buttons)
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
        
        # Also update export button state when dropdown selection changes
        self.main_window.checks_profile_dropdown.currentIndexChanged.connect(
            lambda _: self.profile_handlers._update_export_button_state()
        )
        
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

        # Profile import/export layout (Row 2: export + import buttons)
        profile_io_layout = QHBoxLayout()
        profile_io_layout.addStretch()  # Right-align the buttons
        
        # Export button — only enabled when a custom profile is selected
        self.main_window.export_profile_btn = QPushButton("Export Profile")
        self.main_window.export_profile_btn.setToolTip(
            "Export the selected custom profile to a JSON file for sharing"
        )
        self.main_window.export_profile_btn.clicked.connect(
            self.profile_handlers.on_export_profile_clicked
        )
        theme_manager.style_button(self.main_window.export_profile_btn)
        
        # Import button — always enabled
        import_profile_btn = QPushButton("Import Profile")
        import_profile_btn.setToolTip(
            "Import a custom profile from a JSON file"
        )
        import_profile_btn.clicked.connect(
            self.profile_handlers.on_import_profile_clicked
        )
        theme_manager.style_button(import_profile_btn)
        
        profile_io_layout.addWidget(self.main_window.export_profile_btn)
        profile_io_layout.addWidget(import_profile_btn)
        
        # Set initial export button state
        self.profile_handlers._update_export_button_state()

        # Add widgets to layout
        profile_layout.addWidget(checks_profile_label)
        profile_layout.addWidget(checks_profile_desc)
        profile_layout.addLayout(profile_selection_layout)
        profile_layout.addLayout(profile_io_layout)
        
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