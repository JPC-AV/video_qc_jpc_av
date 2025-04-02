from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QSizePolicy, QTextBrowser
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap

from ..utils.config_manager import ConfigManager
from ..utils.config_setup import SpexConfig, ChecksConfig

from AV_Spex import __version__
version_string = __version__

config_mgr = ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)
spex_config = config_mgr.get_config('spex', SpexConfig)

import os

class DialogHandlers:
    """Dialog implementations for the import tab"""
    
    def __init__(self, parent):
        self.parent = parent
    
    def show_config_info(self):
        """Define the information dialog method"""
        # Create a custom dialog
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Configuration Management Help")
        dialog.setMinimumWidth(500)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Title
        title = QLabel("<h2>Configuration Management</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Content
        content = QTextBrowser()
        content.setOpenExternalLinks(True)
        content.setHtml("""
        <p>This section allows you to save, load, or reset AV Spex configuration settings.</p>
        
        <p><b>Import Config</b><br>
        • Loads previously saved configuration settings from a JSON file<br>
        • Import file can be Checks Config, Spex Config, or All Config<br>
        • Compatible with files created using the Export feature</p>
        
        <p><b>Export Config</b><br>
        • Saves your current configuration settings to a JSON file<br>
        • Options:<br>
        - <i>Checks Config</i>: Exports only which tools run and how (fixity settings, 
            which tools run/check, etc.)<br>
        - <i>Spex Config</i>: Exports only the expected values for file validation
            (codecs, formats, naming conventions, etc.)<br>
        - <i>Complete Config</i>: Exports all settings (both Checks Config and Soex Config)</p>
        
        <p><b>Reset to Default</b><br>
        • Restores all settings to the application's built-in defaults<br>
        • Use this if settings have been changed and you want to start fresh<br>
        • Note: This action cannot be undone</p>
        """)
        
        layout.addWidget(content)
    
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        # Show dialog
        dialog.exec()
    
    def show_about_dialog(self):
        """Show the About dialog with version information and logo."""
        # Create a dialog
        about_dialog = QDialog()
        about_dialog.setWindowTitle("About AV Spex")
        about_dialog.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(about_dialog)
        
        # Add logo
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Get the logo path
        logo_path = config_mgr.get_logo_path('av_spex_the_logo.png')
        
        if logo_path and os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                # Scale pixmap to a reasonable size for the dialog
                scaled_pixmap = pixmap.scaled(QSize(300, 150), 
                                            Qt.AspectRatioMode.KeepAspectRatio, 
                                            Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(scaled_pixmap)
        
        layout.addWidget(logo_label)
        
        # Add version information
        version_label = QLabel(f"Version: {version_string}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
        layout.addWidget(version_label)
        
        # Add additional information if needed
        info_label = QLabel("AV Spex - Audio/Video Specification Checker")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        copyright_label = QLabel("GNU General Public License v3.0")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copyright_label)
        
        # Show the dialog
        about_dialog.exec()