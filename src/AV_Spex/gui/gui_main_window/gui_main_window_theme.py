from PyQt6.QtWidgets import QLabel, QHBoxLayout
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap

import os

from ...gui.gui_theme_manager import ThemeManager

class MainWindowTheme:
    """Theme handling helper methods for the main window"""
    
    def __init__(self, main_window):
        self.main_window = main_window
    
    def on_theme_changed(self, palette):
        """Handle theme changes across the application."""
        
        # Apply palette to main window
        self.main_window.setPalette(palette)
        
        # Get the theme manager
        theme_manager = ThemeManager.instance()
        
        # Update the tabs
        if hasattr(self.main_window, 'tabs'):
            self.main_window.tabs.setStyleSheet(theme_manager.get_tab_style())

        if hasattr(self.main_window, 'export_config_dropdown'):
            theme_manager.style_combobox(self.main_window.export_config_dropdown)
        
        # Style the special buttons
        self._style_special_buttons()
        
        # Refresh the logo
        self._refresh_logo()
        
        # Force repaint
        self.main_window.update()
    
    def _style_special_buttons(self):
        """Apply special styling to buttons that need it"""
        # Style the 'Check Spex' button
        if hasattr(self.main_window, 'check_spex_button'):
            self.main_window.check_spex_button.setStyleSheet("""
                QPushButton {
                    font-weight: bold;
                    padding: 8px 16px;
                    font-size: 14px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:disabled {
                    background-color: #A5D6A7; 
                    color: #E8F5E9;             
                    opacity: 0.8;               
                }
            """)
        
        # Style the 'Show Processing Window' button
        if hasattr(self.main_window, 'open_processing_button'):
            self.main_window.open_processing_button.setStyleSheet("""
                QPushButton {
                    font-weight: bold;
                    padding: 8px 16px;
                    font-size: 14px;
                    background-color: white;
                    color: #4CAF50;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #d2ffed;
                }
                QPushButton:disabled {
                    background-color: #E8F5E9; 
                    color: #A5D6A7;             
                    opacity: 0.8;               
                }
            """)
        
        # Style the progress indicator
        if hasattr(self.main_window, 'processing_indicator'):
            self.main_window.processing_indicator.setStyleSheet("""
                QProgressBar {
                    background-color: palette(Base);
                    text-align: center;
                    padding: 1px;
                }
                QProgressBar::chunk {
                    background-color: palette(Highlight);
                }
            """)
        
        # Refresh the logo
        self._refresh_logo()
        
        # Force repaint
        self.main_window.update()

    def _refresh_logo(self):
        """Refresh the logo when theme changes"""
        # First, find and remove the existing logo layout
        for i in range(self.main_window.main_layout.count()):
            item = self.main_window.main_layout.itemAt(i)
            # Check if this layout item contains our logo (you might need to adapt this check)
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
    
    def _delayed_logo_setup(self):
        """Delayed logo setup for frozen applications"""
        self._load_logo()

    def _load_logo(self):
        """Load and display the logo based on current theme"""
        # Get ThemeManager instance
        theme_manager = ThemeManager.instance()
        
        # Define light and dark logo paths
        light_logo_path = self.main_window.config_mgr.get_logo_path('Branding_avspex_noJPC_030725.png')
        dark_logo_path = self.main_window.config_mgr.get_logo_path('Branding_avspex_noJPC_inverted_032325.png')
        
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
        
        if logo_path and os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                # Scale pixmap to window width while keeping aspect ratio
                scaled_pixmap = pixmap.scaledToWidth(self.main_window.width(), Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
            else:
                print(f"Failed to load image at path: {logo_path}")
        else:
            print(f"Invalid logo path: {logo_path}")
        
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(label)
        return image_layout