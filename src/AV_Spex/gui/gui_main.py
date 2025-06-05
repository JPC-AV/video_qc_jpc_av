from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QPalette

import os
import sys

from AV_Spex.gui.gui_theme_manager import ThemeManager, ThemeableMixin
from AV_Spex.utils.config_setup import SpexConfig, ChecksConfig
from AV_Spex.utils.config_manager import ConfigManager
from AV_Spex.utils.log_setup import logger

from AV_Spex.processing.worker_thread import ProcessingWorker
from AV_Spex.processing.avspex_processor import AVSpexProcessor
from AV_Spex.gui.gui_signals import ProcessingSignals

from AV_Spex import __version__
version_string = __version__

from AV_Spex.gui.gui_main_window.gui_main_window_ui import MainWindowUI
from AV_Spex.gui.gui_main_window.gui_main_window_signals import MainWindowSignals
from AV_Spex.gui.gui_main_window.gui_main_window_processing import MainWindowProcessing
from AV_Spex.gui.gui_main_window.gui_main_window_theme import MainWindowTheme

from AV_Spex.gui.gui_import_tab import ImportTab
from AV_Spex.gui.gui_checks_tab.gui_checks_tab import ChecksTab
from AV_Spex.gui.gui_spex_tab import SpexTab

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
        
        # Optional: Run theme diagnostic on startup (only in debug builds)
        if self._should_run_diagnostic():
            self._run_startup_diagnostic()

    def _should_run_diagnostic(self):
        """Determine if we should run theme diagnostic on startup"""
        # Only run diagnostic if:
        # 1. App is signed (potential issue environment)
        # 2. Debug flag is set
        # 3. Or environment variable is set
        
        import os
        
        # Check environment variable
        if os.environ.get('AVSPEX_DEBUG_THEME'):
            return True
        
        # Check if running as signed bundle
        try:
            import subprocess
            if getattr(sys, 'frozen', False):
                result = subprocess.run([
                    'codesign', '-dv', sys.executable
                ], capture_output=True, text=True, timeout=5)
                return result.returncode == 0
        except:
            pass
        
        return False

    def _run_startup_diagnostic(self):
        """Run theme diagnostic at startup and log results"""
        try:
            theme_manager = ThemeManager.instance()
            
            # Run quick diagnostic in a separate thread to avoid blocking UI
            from PyQt6.QtCore import QTimer
            
            # Delay diagnostic slightly to let UI fully initialize
            QTimer.singleShot(1000, lambda: theme_manager.run_theme_diagnostic())
            
        except Exception as e:
            logger.warning(f"Startup theme diagnostic failed: {e}")
    
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
        
        # Remove the recursive call to on_quit_clicked
        # We'll let the aboutToQuit signal handle saving configs
        
        # Clean up main window theme connections last
        self.cleanup_theme_handling()
        
        # Call parent implementation
        super().closeEvent(event)

    def run_theme_diagnostic_manually(self):
        """Manual theme diagnostic trigger (can be bound to menu item or shortcut)"""
        theme_manager = ThemeManager.instance()
        results = theme_manager.run_theme_diagnostic()
        
        if results:
            # Optionally show results in a dialog or console
            logger.info("Theme diagnostic completed - check logs for details")
        else:
            logger.error("Theme diagnostic failed to run")
    