from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QPalette

import os
import sys

from ..gui.gui_theme_manager import ThemeManager, ThemeableMixin
from ..utils.config_setup import SpexConfig, ChecksConfig
from ..utils.config_manager import ConfigManager
from ..utils.log_setup import logger

from ..processing.worker_thread import ProcessingWorker
from ..processing.avspex_processor import AVSpexProcessor
from ..gui.gui_signals import ProcessingSignals

from AV_Spex import __version__
version_string = __version__

from ..gui.gui_main_window.gui_main_window_ui import MainWindowUI
from ..gui.gui_main_window.gui_main_window_signals import MainWindowSignals
from ..gui.gui_main_window.gui_main_window_processing import MainWindowProcessing
from ..gui.gui_main_window.gui_main_window_theme import MainWindowTheme

from ..gui.gui_import_tab import ImportTab
from ..gui.gui_checks_tab.gui_checks_tab import ChecksTab
from ..gui.gui_spex_tab import SpexTab

# Get configuration manager
config_mgr = ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)
spex_config = config_mgr.get_config('spex', SpexConfig)


class MainWindow(QMainWindow, ThemeableMixin):
    """Main application window with tabs for configuration and settings."""
    
    def __init__(self):
        super().__init__()
        self.signals = ProcessingSignals()
        self.worker = None
        self.processing_window = None

        # Initialize collections for theme-aware components
        self.import_tab_group_boxes = [] 
        self.spex_tab_group_boxes = []
        self.checks_tab_group_boxes = []
        
        # Initialize settings
        self.settings = QSettings('NMAAHC', 'AVSpex')
        self.selected_directories = []
        self.check_spex_clicked = False
        self.source_directories = []

        # Initialize MainWindow
        self.ui = MainWindowUI(self)
        self.signals_handler = MainWindowSignals(self)
        self.processing = MainWindowProcessing(self)
        self.theme = MainWindowTheme(self)

        #Initialize Tabs
        self.checks_tab = ChecksTab(self)
        self.spex_tab = SpexTab(self)
        self.import_tab = ImportTab(self)

        # Connect all signals
        self.signals_handler.setup_signal_connections()
        
        # Setup UI
        self.ui.setup_ui()
        
        # Setup theme handling
        self.setup_theme_handling()
    
    def closeEvent(self, event):
        """Handle application shutdown and clean up resources."""
        # Clean up any child windows first
        for child_name in ['config_widget', 'processing_window', 'new_window']:
            child = getattr(self, child_name, None)
            if child and isinstance(child, QWidget):
                # If it has theme handling, clean it up
                if hasattr(child, 'cleanup_theme_handling'):
                    child.cleanup_theme_handling()
                child.close()
        
        # Clean up tab theme connections
        for tab_name in ['import_tab', 'checks_tab', 'spex_tab']:
            tab = getattr(self, tab_name, None)
            if tab and hasattr(tab, 'cleanup_theme_handling'):
                tab.cleanup_theme_handling()
        
        # Stop worker if running
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        
        # Call quit handling method but don't call it if we're already in the closing process
        if not hasattr(self, '_is_closing') or not self._is_closing:
            self._is_closing = True
            self.signals_handler.on_quit_clicked()
        
        # Clean up main window theme connections last
        self.cleanup_theme_handling()
        
        # Call parent implementation
        super().closeEvent(event)
    