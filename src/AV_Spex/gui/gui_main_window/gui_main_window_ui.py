from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QScrollArea, QMenuBar, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QPixmap, QKeySequence

import os
import sys
import platform

from AV_Spex.gui.gui_theme_manager import ThemeManager

class MainWindowUI:
    """UI components and setup for the main window"""
    
    def __init__(self, main_window):
        self.main_window = main_window
    
    def setup_ui(self):
        """Set up the main UI components"""
        print("🔍 DEBUG: MainWindowUI.setup_ui() called")
        self.main_window.setMinimumSize(750, 800)

        ## self.main_window.windowFlags() retrieves the current window flags
        ## Qt.WindowType.WindowMaximizeButtonHint enables the maximize button in the window's title bar.
        self.main_window.setWindowFlags(self.main_window.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        
        self.main_window.setWindowTitle("AV Spex")
        
        # Set up menu bar
        self.menu_bar = QMenuBar(self.main_window)
        self.main_window.setMenuBar(self.menu_bar)
        
        # Platform-specific menu setup
        if platform.system() == 'Darwin':
            # On macOS, the application menu is handled differently
            # File menu (this becomes the first menu after the Apple menu on macOS)
            self.file_menu = self.menu_bar.addMenu("File")
            self.import_action = self.file_menu.addAction("Import Directory")
            self.import_action.triggered.connect(self.main_window.import_tab.import_directories)
            
            # Add Quit action to the File menu on macOS
            self.file_menu.addSeparator()
            self.quit_action = self.file_menu.addAction("Quit")
            self.quit_action.setShortcut(QKeySequence.StandardKey.Quit)
            self.quit_action.triggered.connect(self.main_window.signals_handler.on_quit_clicked)
            
            # About action in its own menu
            self.help_menu = self.menu_bar.addMenu("Help")
            self.about_action = self.help_menu.addAction("About AV Spex")
            self.about_action.triggered.connect(self.main_window.import_tab.dialog_handlers.show_about_dialog)
        else:
            # For Windows/Linux (if we expand support)
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
        
        print("🔍 DEBUG: MainWindowUI.setup_ui() completed")
    
    def setup_main_layout(self):
        """Set up the main window layout structure"""
        print("🔍 DEBUG: Setting up main layout")
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
        print("🔍 DEBUG: Setting up logo")
        self.main_window.theme._load_logo()
    
    def setup_tabs(self):
        """Set up tab styling"""
        print("🔍 DEBUG: Setting up tabs")
        theme_manager = ThemeManager.instance()
        
        # Create new tabs
        self.main_window.tabs = QTabWidget()
        self.main_window.tabs.setStyleSheet(theme_manager.get_tab_style())

        self.main_window.main_layout.addWidget(self.main_window.tabs)

        # Set up individual tabs
        print("🔍 DEBUG: Setting up import tab")
        self.main_window.import_tab.setup_import_tab()
        print("🔍 DEBUG: Setting up checks tab")
        self.main_window.checks_tab.setup_checks_tab()
        print("🔍 DEBUG: Setting up spex tab")
        self.main_window.spex_tab.setup_spex_tab()