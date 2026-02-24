from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QScrollArea
)
from PyQt6.QtCore import Qt

from AV_Spex.gui.gui_theme_manager import ThemeManager, ThemeableMixin
from AV_Spex.gui.gui_complex_window import ComplexWindow
from AV_Spex.utils.config_manager import ConfigManager
from AV_Spex.utils.config_setup import ChecksConfig
from AV_Spex.utils.log_setup import logger

config_mgr = ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)

class ComplexTab(ThemeableMixin):
    """Complex tab for advanced configuration options"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        
        # Initialize tab UI elements as instance attributes
        self.complex_group = None
        
        # Initialize theme handling
        self.setup_theme_handling()

    def on_theme_changed(self, palette):
        """Handle theme changes for this tab"""
        theme_manager = ThemeManager.instance()
        
        # Update all group boxes
        for group_box in self.main_window.complex_tab_group_boxes:
            if group_box is not None:
                # Preserve the title position when refreshing style
                group_box_title_pos = group_box.property("title_position") or "top center"
                theme_manager.style_groupbox(group_box, group_box_title_pos)
        
        # Update the complex widget if it exists
        if hasattr(self.main_window, 'complex_widget') and self.main_window.complex_widget:
            # If the complex widget has its own theme handling, let it handle the change
            if hasattr(self.main_window.complex_widget, 'on_theme_changed'):
                self.main_window.complex_widget.on_theme_changed(palette)
    
    def setup_complex_tab(self):
        """Set up the Complex tab with theme-aware styling"""
        # Get the theme manager instance
        theme_manager = ThemeManager.instance()
        
        # Initialize or reset the group boxes collection
        self.main_window.complex_tab_group_boxes = []
    
        # Create the tab
        complex_tab = QWidget()
        complex_layout = QVBoxLayout(complex_tab)
        self.main_window.tabs.addTab(complex_tab, "Complex")

        # Scroll Area for Vertical Scrolling in "Complex" Tab
        main_scroll_area = QScrollArea(self.main_window)
        main_scroll_area.setWidgetResizable(True)
        main_widget = QWidget(self.main_window)
        main_scroll_area.setWidget(main_widget)

        # Vertical layout for the main content in "Complex"
        vertical_layout = QVBoxLayout(main_widget)

        # Complex Options section
        self.complex_group = QGroupBox("Complex Analysis Options")
        theme_manager.style_groupbox(self.complex_group, "top center")
        self.main_window.complex_tab_group_boxes.append(self.complex_group)
        
        complex_options_layout = QVBoxLayout()
        
        complex_scroll_area = QScrollArea()
        complex_scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
            }
        """)
        self.main_window.complex_widget = ComplexWindow()
        complex_scroll_area.setWidgetResizable(True)
        complex_scroll_area.setWidget(self.main_window.complex_widget)

        # Set a minimum width for the complex widget to ensure legibility
        complex_scroll_area.setMinimumWidth(450)

        complex_options_layout.addWidget(complex_scroll_area)
        self.complex_group.setLayout(complex_options_layout)
        vertical_layout.addWidget(self.complex_group)

        # Add scroll area to main layout
        complex_layout.addWidget(main_scroll_area)