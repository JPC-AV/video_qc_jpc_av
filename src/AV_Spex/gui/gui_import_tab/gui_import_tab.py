from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QScrollArea, 
    QPushButton, QListWidget, QFileDialog, QProgressBar, QSizePolicy, 
    QStyle, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, QDir

import os

from ...gui.gui_theme_manager import ThemeManager
from ...gui.gui_processing_window import DirectoryListWidget
from ...gui.gui_import_tab.gui_import_tab_config_window import GuiConfigHandlers
from ...gui.gui_import_tab.gui_import_tab_dialog_handler import DialogHandlers 

class ImportTabSetup:
    """Setup and handlers for the Import tab"""
    
    def __init__(self, parent):
        self.parent = parent
        self.guiconfig_handlers = GuiConfigHandlers(parent)
        self.dialog_handlers = DialogHandlers(parent)
    
    def setup_import_tab(self):
        """Set up the Import tab for directory selection"""
        # Get the theme manager instance
        theme_manager = ThemeManager.instance()
        
        # Initialize the group boxes collection for this tab
        self.parent.import_tab_group_boxes = []
        
        # Create the tab
        import_tab = QWidget()
        import_layout = QVBoxLayout(import_tab)
        self.parent.tabs.addTab(import_tab, "Import")
        
        # Main scroll area
        main_scroll_area = QScrollArea(self.parent)
        main_scroll_area.setWidgetResizable(True)
        main_widget = QWidget(self.parent)
        main_scroll_area.setWidget(main_widget)
        
        # Vertical layout for the content
        vertical_layout = QVBoxLayout(main_widget)
        
        # Import directory section
        self.import_group = QGroupBox("Import Directories")
        theme_manager.style_groupbox(self.import_group, "top center")
        self.parent.import_tab_group_boxes.append(self.import_group)
        
        import_layout_section = QVBoxLayout()
        
        # Import directory button
        import_directories_button = QPushButton("Import Directory...")
        import_directories_button.clicked.connect(self.import_directories)
        
        # Directory section
        directory_label = QLabel("Selected Directories:")
        directory_label.setStyleSheet("font-weight: bold;")
        self.parent.directory_list = DirectoryListWidget(self.parent)
        self.parent.directory_list.setStyleSheet("""
            QListWidget {
                border: 1px solid gray;
                border-radius: 3px;
            }
        """)
        
        # Delete button
        delete_button = QPushButton("Delete Selected")
        delete_button.clicked.connect(self.delete_selected_directory)
        
        # Add widgets to layout
        import_layout_section.addWidget(import_directories_button)
        import_layout_section.addWidget(directory_label)
        import_layout_section.addWidget(self.parent.directory_list)
        import_layout_section.addWidget(delete_button)
        
        self.import_group.setLayout(import_layout_section)
        vertical_layout.addWidget(self.import_group)
        
        # Style all buttons in the section
        theme_manager.style_buttons(self.import_group)

        # Config Import section
        self.config_import_group = QGroupBox("Config Import")
        theme_manager.style_groupbox(self.config_import_group, "top center")
        self.parent.import_tab_group_boxes.append(self.config_import_group)
        
        config_import_layout = QVBoxLayout()

        # Create a horizontal layout for the header row
        header_layout = QHBoxLayout()

        # Create the config info button
        info_button = QPushButton()
        info_button.setIcon(self.parent.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        info_button.setFixedSize(24, 24)
        info_button.setToolTip("Click for more info about config options")
        info_button.setFlat(True)  # Make it look like just an icon
        info_button.clicked.connect(self.dialog_handlers.show_config_info)
        header_layout.addWidget(info_button)

        # Description label
        config_desc_label = QLabel("Import, export, or reset Checks/Spex configuration:")
        config_desc_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(config_desc_label)

        # Add a stretch to push the info button to the right
        header_layout.addStretch(1)

        # Add the header layout to the main vertical layout
        config_import_layout.addLayout(header_layout)

        # Add some spacing
        config_import_layout.addSpacing(10)
        
        # Create buttons layout
        buttons_layout = QHBoxLayout()
        
        # Import Config button
        import_config_button = QPushButton("Import Config")
        import_config_button.clicked.connect(self.guiconfig_handlers.import_config)
        buttons_layout.addWidget(import_config_button)
        
        # Export Config layout
        export_button_layout = QHBoxLayout()

        # Create the dropdown for export options
        self.parent.export_config_dropdown = QComboBox()

        # Add the default placeholder option first
        self.parent.export_config_dropdown.addItem("Export Config Type...")  
        self.parent.export_config_dropdown.addItem("Export Checks Config")
        self.parent.export_config_dropdown.addItem("Export Spex Config")
        self.parent.export_config_dropdown.addItem("Export All Config")

        # Connect the combobox signal to your function
        self.parent.export_config_dropdown.currentIndexChanged.connect(self.guiconfig_handlers.export_selected_config)

        theme_manager.style_combobox(self.parent.export_config_dropdown)
        
        # Add widgets to layout
        export_button_layout.addWidget(self.parent.export_config_dropdown)
        buttons_layout.addLayout(export_button_layout)

        # Set the first item as the current item (the placeholder)
        self.parent.export_config_dropdown.setCurrentIndex(0)
        
        # Reset to Default Config button
        reset_config_button = QPushButton("Reset to Default")
        reset_config_button.clicked.connect(self.guiconfig_handlers.reset_config)
        buttons_layout.addWidget(reset_config_button)
        
        config_import_layout.addLayout(buttons_layout)
        
        self.config_import_group.setLayout(config_import_layout)
        vertical_layout.addWidget(self.config_import_group)
        
        # Style all buttons in the config section
        theme_manager.style_buttons(self.config_import_group)
        
        # Add scroll area to main layout
        import_layout.addWidget(main_scroll_area)
        
        # Bottom section with processing controls
        # Similar to what you have in checks_tab but just the processing-related buttons
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 10, 0, 10)  # Add some vertical padding
        
        # Open Processing Window button
        self.parent.open_processing_button = QPushButton("Show Processing Window")
        self.parent.open_processing_button.setStyleSheet("""
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
        self.parent.open_processing_button.clicked.connect(self.parent.signals_handler.on_open_processing_clicked)
        # Initially disable the button since no processing is running
        self.parent.open_processing_button.setEnabled(False)
        bottom_row.addWidget(self.parent.open_processing_button)
        
        # Cancel button
        self.parent.cancel_processing_button = QPushButton("Cancel Processing")
        self.parent.cancel_processing_button.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                padding: 8px 16px;
                font-size: 14px;
                background-color: #ff9999;
                color: #4d2b12;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #ff8080;
            }
            QPushButton:disabled {
                background-color: #f5e9e3; 
                color: #cd9e7f;             
                opacity: 0.8;               
            }
        """)
        self.parent.cancel_processing_button.clicked.connect(self.parent.processing.cancel_processing)
        self.parent.cancel_processing_button.setEnabled(False)
        bottom_row.addWidget(self.parent.cancel_processing_button)
        
        # create layout for current processing
        self.now_processing_layout = QVBoxLayout()
        
        # Add a status label that shows current file being processed
        self.parent.main_status_label = QLabel("Not processing")
        self.parent.main_status_label.setWordWrap(True)
        self.parent.main_status_label.setMaximumWidth(300)  # Limit width to prevent stretching
        self.parent.main_status_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)  # Minimize height
        self.parent.main_status_label.setVisible(False)  # initially hidden 
        self.now_processing_layout.addWidget(self.parent.main_status_label)
        
        # Add a small indeterminate progress bar
        self.parent.processing_indicator = QProgressBar(self.parent)
        self.parent.processing_indicator.setMaximumWidth(100)  # Make it small
        self.parent.processing_indicator.setMaximumHeight(10)  # Make it shorter
        self.parent.processing_indicator.setRange(0, 0)
        self.parent.processing_indicator.setTextVisible(False)  # No percentage text
        self.parent.processing_indicator.setStyleSheet("""
            QProgressBar {
                background-color: palette(Base);
                text-align: center;
                padding: 1px;
            }
        """)
        self.parent.processing_indicator.setVisible(False)  # Initially hidden
        self.now_processing_layout.addWidget(self.parent.processing_indicator)
        
        # Add the processing button layout to the bottom row
        # Use a stretch factor of 0 to keep it from expanding
        bottom_row.addLayout(self.now_processing_layout, 0)
        
        # Add a stretch to push the Check Spex button to the right
        bottom_row.addStretch(1)
        
        # Check Spex button
        self.parent.check_spex_button = QPushButton("Check Spex!")
        self.parent.check_spex_button.setStyleSheet("""
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
        self.parent.check_spex_button.clicked.connect(self.on_check_spex_clicked)
        bottom_row.addWidget(self.parent.check_spex_button, 0)
        
        import_layout.addLayout(bottom_row)
    
    def import_directories(self):
        """Import directories for processing."""
        # Get the last directory from settings
        last_directory = self.parent.settings.value('last_directory', '')
        
        # Use native file dialog
        file_dialog = QFileDialog(self.parent, "Select Directories")
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        
        # Set the starting directory to the parent of the last used directory
        if last_directory:
            dir_info = QDir(last_directory)
            if dir_info.cdUp():  # Move up to parent directory
                parent_dir = dir_info.absolutePath()
                file_dialog.setDirectory(parent_dir)
        
        # Try to enable multiple directory selection with the native dialog
        file_dialog.setOption(QFileDialog.Option.ReadOnly, False)

        if file_dialog.exec():
            directories = file_dialog.selectedFiles()  # Get selected directories
            
            # Save the last used directory
            if directories:
                self.parent.settings.setValue('last_directory', directories[0])
                self.parent.settings.sync()  # Ensure settings are saved
            
            for directory in directories:
                if directory not in self.parent.selected_directories:
                    self.parent.selected_directories.append(directory)
                    self.parent.directory_list.addItem(directory)
    
    def update_selected_directories(self):
        """Update source_directories from the QListWidget."""
        self.parent.source_directories = [self.parent.directory_list.item(i).text() for i in range(self.parent.directory_list.count())]

    def get_source_directories(self):
        """Return the selected directories if Check Spex was clicked."""
        return self.parent.selected_directories if self.parent.check_spex_clicked else None
    
    def delete_selected_directory(self):
        """Delete the selected directory from the list widget and the selected_directories list."""
        # Get the selected items
        selected_items = self.parent.directory_list.selectedItems()
        
        if not selected_items:
            return  # No item selected, do nothing
        
        # Remove each selected item from both the QListWidget and selected_directories list
        for item in selected_items:
            # Remove from the selected_directories list
            directory = item.text()
            if directory in self.parent.selected_directories:
                self.parent.selected_directories.remove(directory)
            
            # Remove from the QListWidget
            self.parent.directory_list.takeItem(self.parent.directory_list.row(item))

    def on_check_spex_clicked(self):
        """Handle the Check Spex button click."""
        self.update_selected_directories()
        self.parent.check_spex_clicked = True  # Mark that the button was clicked
        self.parent.config_mgr.save_last_used_config('checks')
        self.parent.config_mgr.save_last_used_config('spex')
        # Make sure the processing window is visible before starting the process
        if hasattr(self.parent, 'processing_window') and self.parent.processing_window:
            # If it exists but might be hidden, show it
            self.parent.processing_window.show()
            self.parent.processing_window.raise_()
            self.parent.processing_window.activateWindow()
        
        # Call worker thread
        self.parent.processing.call_process_directories()