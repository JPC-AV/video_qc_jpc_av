from PyQt6.QtWidgets import QFileDialog, QMessageBox
import os

from ..utils.config_io import ConfigIO
from ..utils.log_setup import logger

class ConfigHandlers:
    """Configuration import/export/reset handlers"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def export_selected_config(self):
        selected_option = self.parent.export_config_dropdown.currentText()
        # Skip export if the placeholder option is selected
        if selected_option == "Export Config Type...":
            return
        elif selected_option == "Export Checks Config":
            self.export_config('checks')
        elif selected_option == "Export Spex Config":
            self.export_config('spex')
        elif selected_option == "Export All Config":
            self.export_config('all')

    def import_config(self):
        """Import configuration from a file."""
        file_dialog = QFileDialog(self.parent, "Import Configuration")
        file_dialog.setNameFilter("Config Files (*.json);;All Files (*)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            try:
                # Use the ConfigIO class to import config
                config_io = ConfigIO(self.parent.config_mgr)
                config_io.import_configs(file_path)
                
                # Reload UI components to reflect new settings
                self.parent.config_widget.load_config_values()

                # Ensure recent config ref
                self.parent.checks_config = self.parent.config_mgr.get_config('checks', ChecksConfig)
                self.parent.spex_config = self.parent.config_mgr.get_config('spex', SpexConfig)

                # Spex dropdowns
                # file name dropdown
                if self.parent.spex_config.filename_values.Collection == "JPC":
                    self.parent.filename_profile_dropdown.setCurrentText("JPC file names")
                elif self.parent.spex_config.filename_values.Collection == "2012_79":
                    self.parent.filename_profile_dropdown.setCurrentText("Bowser file names")
                
                # Signalflow profile dropdown
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
                
                QMessageBox.information(self.parent, "Success", f"Configuration imported successfully from {file_path}")
            except Exception as e:
                logger.error(f"Error importing config: {str(e)}")
                QMessageBox.critical(self.parent, "Error", f"Error importing configuration: {str(e)}")

    def export_config(self, config_type):
        """Export configuration to a file."""
        file_dialog = QFileDialog(self.parent, "Export Configuration")
        file_dialog.setNameFilter("JSON Files (*.json);;All Files (*)")
        file_dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        file_dialog.setDefaultSuffix("json")
        
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            try:
                # Use the ConfigIO class to export config
                config_io = ConfigIO(self.parent.config_mgr)
                config_io.save_configs(file_path, config_type)
                
                QMessageBox.information(self.parent, "Success", f"Configuration exported successfully to {file_path}")
            except Exception as e:
                logger.error(f"Error exporting config: {str(e)}")
                QMessageBox.critical(self.parent, "Error", f"Error exporting configuration: {str(e)}")

    def reset_config(self):
        """Reset configuration to default values."""
        # Ask for confirmation
        result = QMessageBox.question(
            self.parent,
            "Confirm Reset",
            "Are you sure you want to reset all configuration to default values? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if result == QMessageBox.StandardButton.Yes:
            try:
                # Reset config by removing user config files
                user_config_dir = self.parent.config_mgr._user_config_dir
                try:
                    # Remove the user config files
                    os.remove(os.path.join(user_config_dir, "last_used_checks_config.json"))
                    os.remove(os.path.join(user_config_dir, "last_used_spex_config.json"))
                    
                    QMessageBox.information(self.parent, "Success", "Configuration has been reset to default values")
                except FileNotFoundError:
                    # It's okay if the files don't exist
                    QMessageBox.information(self.parent, "Information", "Already using default configuration")
            except Exception as e:
                logger.error(f"Error resetting config: {str(e)}")
                QMessageBox.critical(self.parent, "Error", f"Error resetting configuration: {str(e)}")

            self.parent.config_mgr._configs = {}
            self.parent.config_mgr = ConfigManager()
            self.parent.checks_config = self.parent.config_mgr.get_config('checks', ChecksConfig)
            self.parent.spex_config = self.parent.config_mgr.get_config('spex', SpexConfig)
            self.parent.config_mgr.save_last_used_config('checks')

            # Reload UI components to reflect new settings
            self.parent.config_widget.load_config_values()

            # Spex dropdowns
            # file name dropdown
            if self.parent.spex_config.filename_values.Collection == "JPC":
                self.parent.filename_profile_dropdown.setCurrentText("JPC file names")
            elif self.parent.spex_config.filename_values.Collection == "2012_79":
                self.parent.filename_profile_dropdown.setCurrentText("Bowser file names")
            
            # Signalflow profile dropdown
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