from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
    QLabel, QScrollArea, QMenuBar, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QPixmap

import os
import sys

from ...gui.gui_theme_manager import ThemeManager
from ...gui.gui_main_window.gui_main_window_signals import MainWindowSignals
from ...gui.gui_import_tab.gui_import_tab_dialog_handler import DialogHandlers
from ...gui.gui_import_tab.gui_import_tab import ImportTabSetup

class MainWindowUI:
    """UI components and setup for the main window"""
    
    def __init__(self, parent):
        self.parent = parent
        self.signals_handler = MainWindowSignals
        self.dialog_handlers = DialogHandlers
        self.import_tab = ImportTabSetup
    
    def setup_ui(self):
        """Set up the main UI components"""
        self.parent.setMinimumSize(700, 800)

        ## self.parent.windowFlags() retrieves the current window flags
        ## Qt.WindowType.WindowMaximizeButtonHint enables the maximize button in the window's title bar.
        self.parent.setWindowFlags(self.parent.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        
        self.parent.setWindowTitle("AV Spex")
        
        # Set up menu bar
        self.menu_bar = QMenuBar(self.parent)
        self.parent.setMenuBar(self.menu_bar)
        
        # App menu 
        self.app_menu = self.menu_bar.addMenu("AV Spex")
        self.about_action = self.app_menu.addAction("About AV Spex")
        self.about_action.triggered.connect(self.dialog_handlers.show_about_dialog)
        
        # Add a separator
        self.app_menu.addSeparator()
        
        # Add Quit action to the app menu
        self.quit_action = self.app_menu.addAction("Quit")
        self.quit_action.triggered.connect(self.signals_handler.on_quit_clicked)
        
        # File menu (comes after the app menu)
        self.file_menu = self.menu_bar.addMenu("File")
        self.import_action = self.file_menu.addAction("Import Directory")
        self.import_action.triggered.connect(self.import_tab.import_directories)

        self.setup_main_layout()
        
        self.logo_setup()

        self.setup_tabs()
    
    def setup_main_layout(self):
        """Set up the main window layout structure"""
        # Create and set central widget
        self.parent.central_widget = QWidget()
        self.parent.setCentralWidget(self.parent.central_widget)

        # Create main vertical layout
        self.parent.main_layout = QVBoxLayout(self.parent.central_widget)

        # Set layout margins and spacing
        self.parent.main_layout.setContentsMargins(10, 10, 10, 10)
        self.parent.main_layout.setSpacing(10)
    
    def logo_setup(self):
        """Set up the logo display"""
        if getattr(sys, 'frozen', False):
            QTimer.singleShot(0, self.parent.theme._delayed_logo_setup)
        else:
            self.parent.theme._load_logo()
    
    def setup_tabs(self):
        """Set up tab styling"""
        theme_manager = ThemeManager.instance()
        
        # Create new tabs
        self.parent.tabs = QTabWidget()
        self.parent.tabs.setStyleSheet(theme_manager.get_tab_style())

        self.parent.main_layout.addWidget(self.parent.tabs)

        # Set up individual tabs
        self.parent.import_tab.setup_import_tab()
        self.parent.checks_tab.setup_checks_tab()
        self.parent.spex_tab.setup_spex_tab()