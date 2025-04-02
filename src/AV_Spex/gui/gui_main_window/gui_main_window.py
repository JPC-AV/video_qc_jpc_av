from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QPalette

import os
import sys

from ...gui.gui_theme_manager import ThemeManager, ThemeableMixin
from ...utils.config_setup import SpexConfig, ChecksConfig
from ...utils.config_manager import ConfigManager
from ...utils.log_setup import logger

from ...processing.worker_thread import ProcessingWorker
from ...processing.avspex_processor import AVSpexProcessor
from ...gui.gui_signals import ProcessingSignals

from AV_Spex import __version__
version_string = __version__

from ...gui.gui_main_window.gui_main_window_ui import MainWindowUI
from ...gui.gui_main_window.gui_main_window_signals import MainWindowSignals
from ...gui.gui_main_window.gui_main_window_processing import MainWindowProcessing
from ...gui.gui_main_window.gui_main_window_theme import MainWindowTheme
from ...gui.gui_import_tab.gui_import_tab import ImportTabSetup
from ...gui.gui_import_tab.gui_import_tab_dialog_handler import DialogHandlers
from ...gui.gui_import_tab.gui_import_tab_config_window import GuiConfigHandlers
from ...gui.gui_checks_tab.gui_checks_tab import ChecksTabSetup
from ...gui.gui_checks_tab.gui_checks_tab_profiles import ChecksProfileHandlers
from ...gui.gui_spex_tab.gui_spex_tab import SpexTabSetup
from ...gui.gui_spex_tab.gui_spex_tab_profiles import SpexProfileHandlers


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

        # Initialize configuration manager
        self.config_mgr = ConfigManager()
        self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
        self.spex_config = self.config_mgr.get_config('spex', SpexConfig)
        
        # Initialize settings
        self.settings = QSettings('NMAAHC', 'AVSpex')
        self.selected_directories = []
        self.check_spex_clicked = False
        self.source_directories = []

        # Initialize components
        self.ui = MainWindowUI(self)
        self.signals_handler = MainWindowSignals(self)
        self.processing = MainWindowProcessing(self)
        self.theme = MainWindowTheme(self)
        self.import_tab = ImportTabSetup(self)
        self.checks_tab = ChecksTabSetup(self)
        self.spex_tab = SpexTabSetup(self)
        self.guiconfig_handlers = GuiConfigHandlers(self)
        self.checks_profile_handlers = ChecksProfileHandlers(self)
        self.spex_profile_handlers = SpexProfileHandlers(self)
        self.dialog_handlers = DialogHandlers(self)

        # Connect all signals
        self.signals_handler.setup_signal_connections()
        
        # Setup UI
        self.ui.setup_ui()
        
        # Setup theme handling
        self.setup_theme_handling()
    
    def closeEvent(self, event):
        # Clean up theme connections
        self.cleanup_theme_handling()
        
        # Clean up child windows
        for child_name in ['config_widget', 'processing_window']:
            child = getattr(self, child_name, None)
            if child and hasattr(child, 'cleanup_theme_handling'):
                child.cleanup_theme_handling()
        
        # Stop worker if running
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        
        # Call quit handling method
        self.signals_handler.on_quit_clicked()
        super().closeEvent(event)