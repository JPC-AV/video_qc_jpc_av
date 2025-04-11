from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QScrollArea, QMenuBar, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QPixmap

import os
import sys

from ...gui.gui_theme_manager import ThemeManager

class MainWindowUI:
    """UI components and setup for the main window"""
    
    def __init__(self, main_window):
        self.main_window = main_window
    
    def setup_ui(self):
        """Set up the main UI components"""
        self.main_window.setMinimumSize(700, 800)

        ## self.main_window.windowFlags() retrieves the current window flags
        ## Qt.WindowType.WindowMaximizeButtonHint enables the maximize button in the window's title bar.
        self.main_window.setWindowFlags(self.main_window.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        
        self.main_window.setWindowTitle("AV Spex")
        
        # Set up menu bar
        self.menu_bar = QMenuBar(self.main_window)
        self.main_window.setMenuBar(self.menu_bar)
        
        # App menu 
        self.app_menu = self.menu_bar.addMenu("AV Spex")
        self.about_action = self.app_menu.addAction("About AV Spex")
        self.about_action.triggered.connect(self.main_window.import_tab.dialog_handlers.show_about_dialog)
        
        # Add a separator
        self.app_menu.addSeparator()
        
        # Add Quit action to the app menu
        self.quit_action = self.app_menu.addAction("Quit")
        self.quit_action.triggered.connect(self.main_window.signals_handler.on_quit_clicked)
        
        # File menu (comes after the app menu)
        self.file_menu = self.menu_bar.addMenu("File")
        self.import_action = self.file_menu.addAction("Import Directory")
        self.import_action.triggered.connect(self.main_window.import_tab.import_directories)

        self.setup_main_layout()
        
        self.logo_setup()

        self.setup_tabs()
    
    def setup_main_layout(self):
        """Set up the main window layout structure"""
        # Create and set central widget
        self.main_window.central_widget = QWidget()
        self.main_window.setCentralWidget(self.main_window.central_widget)

        # Set minimum size for the main window
        self.main_window.setMinimumSize(500, 800)  # Width: 500px, Height: 800px

        # Create main vertical layout
        self.main_window.main_layout = QVBoxLayout(self.main_window.central_widget)

        # Set layout margins and spacing
        self.main_window.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_window.main_layout.setSpacing(10)
    
    def logo_setup(self):
        """Set up the logo display"""
        if getattr(sys, 'frozen', False):
            QTimer.singleShot(0, self.main_window.theme._delayed_logo_setup)
        else:
            self.main_window.theme._load_logo()
    
    def setup_tabs(self):
        """Set up tab styling"""
        theme_manager = ThemeManager.instance()
        
        # Create new tabs
        self.main_window.tabs = QTabWidget()
        self.main_window.tabs.setStyleSheet(theme_manager.get_tab_style())

        self.main_window.main_layout.addWidget(self.main_window.tabs)

        # Set up individual tabs
        self.main_window.import_tab.setup_import_tab()
        self.main_window.checks_tab.setup_checks_tab()
        self.main_window.spex_tab.setup_spex_tab()