from PyQt6.QtWidgets import QLabel, QHBoxLayout
from PyQt6.QtCore import Qt

import os

from ...gui.gui_theme_manager import ThemeManager
from ...utils.config_manager import ConfigManager

config_mgr = ConfigManager()

class MainWindowTheme:
    """Theme handling helper methods for the main window"""
    
    def __init__(self, main_window):
        self.main_window = main_window
    
    def on_theme_changed(self, palette):
        """Handle theme changes across the application."""
        # Get the theme manager
        theme_manager = ThemeManager.instance()
        
        # Apply palette to main window
        self.main_window.setPalette(palette)
        
        # Update the tabs
        if hasattr(self.main_window, 'tabs'):
            theme_manager.style_tabs(self.main_window.tabs)

        # Style comboboxes
        if hasattr(self.main_window, 'export_config_dropdown'):
            theme_manager.style_combobox(self.main_window.export_config_dropdown)
        
        # Style the special buttons
        self._style_special_buttons()
        
        # Refresh the logo
        self._refresh_logo()
        
        # Force repaint
        self.main_window.update()
    
    def _style_special_buttons(self):
        """Apply special styling to buttons that need custom styling"""
        theme_manager = ThemeManager.instance()
        
        # Style the 'Check Spex' button
        if hasattr(self.main_window, 'check_spex_button'):
            theme_manager.style_button(self.main_window.check_spex_button, special_style="check_spex")
        
        # Style the 'Show Processing Window' button
        if hasattr(self.main_window, 'open_processing_button'):
            theme_manager.style_button(self.main_window.open_processing_button, special_style="processing_window")
            
        # Style the 'Cancel Processing' button
        if hasattr(self.main_window, 'cancel_processing_button'):
            theme_manager.style_button(self.main_window.cancel_processing_button, special_style="cancel_processing")
        
        # Style the progress indicator
        if hasattr(self.main_window, 'processing_indicator'):
            theme_manager.style_progress_bar(self.main_window.processing_indicator)
    
    def _refresh_logo(self):
        """Refresh the logo when theme changes"""
        # First, find and remove the existing logo layout
        for i in range(self.main_window.main_layout.count()):
            item = self.main_window.main_layout.itemAt(i)
            # Check if this layout item contains our logo
            if item and item.layout() and item.layout().count() > 0:
                widget = item.layout().itemAt(0).widget()
                if isinstance(widget, QLabel) and widget.pixmap() is not None:
                    # Remove the existing logo layout
                    self._remove_layout_item(self.main_window.main_layout, i)
                    break
        
        # Now load the new theme-appropriate logo
        self._load_logo()

    def _remove_layout_item(self, layout, index):
        """Helper method to remove an item from a layout"""
        if index >= 0 and index < layout.count():
            item = layout.takeAt(index)
            if item:
                # If the item has a layout, we need to clear it first
                if item.layout():
                    while item.layout().count():
                        child = item.layout().takeAt(0)
                        if child.widget():
                            child.widget().deleteLater()
                # If the item has a widget, delete it
                if item.widget():
                    item.widget().deleteLater()
                # Delete the item itself
                del item
    
    def _load_logo(self):
        """Load and display the logo based on current theme"""
        # Get ThemeManager instance
        theme_manager = ThemeManager.instance()
        
        # Define light and dark logo paths
        light_logo_path = config_mgr.get_logo_path('Branding_avspex_noJPC_030725.png')
        dark_logo_path = config_mgr.get_logo_path('Branding_avspex_noJPC_inverted_032325.png')
        
        # Get appropriate logo for current theme
        logo_path = theme_manager.get_theme_appropriate_logo(light_logo_path, dark_logo_path)
        
        # Create and add image layout
        image_layout = self.add_image_to_top(logo_path)
        self.main_window.main_layout.insertLayout(0, image_layout)  # Insert at index 0 (top)
    
    def add_image_to_top(self, logo_path):
        """Add image to the top of the main layout."""
        image_layout = QHBoxLayout()
        
        label = QLabel()
        label.setMinimumHeight(100)
        
        # Use the ThemeManager to load the logo
        theme_manager = ThemeManager.instance()
        theme_manager.load_logo(label, logo_path, width=self.main_window.width())
        
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(label)
        return image_layout