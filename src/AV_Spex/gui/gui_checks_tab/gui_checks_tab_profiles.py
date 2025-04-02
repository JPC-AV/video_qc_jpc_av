from PyQt6.QtWidgets import QMessageBox
from ...utils import config_edit
from ...utils.log_setup import logger

class ChecksProfileHandlers:
    """Profile selection handlers for the Checks tab"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def on_profile_selected(self, index):
        """Handle profile selection from dropdown."""
        selected_profile = self.parent.command_profile_dropdown.currentText()
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
            self.parent.config_mgr.save_last_used_config('checks')
        except ValueError as e:
            logger.critical(f"Error: {e}")

        self.parent.config_widget.load_config_values()