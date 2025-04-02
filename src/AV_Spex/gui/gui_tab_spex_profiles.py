from ..utils import config_edit
from ..utils.log_setup import logger

class SpexProfileHandlers:
    """Profile selection handlers for the Spex tab"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def on_filename_profile_changed(self, index):
        """Handle filename profile selection change."""
        selected_option = self.parent.filename_profile_dropdown.itemText(index)
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
        
        self.parent.config_mgr.update_config('spex', updates)
        self.parent.config_mgr.save_last_used_config('spex')

    def on_signalflow_profile_changed(self, index):
        """Handle signal flow profile selection change."""
        selected_option = self.parent.signalflow_profile_dropdown.itemText(index)
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