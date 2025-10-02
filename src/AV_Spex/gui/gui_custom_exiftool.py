from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, 
    QScrollArea, QPushButton, QComboBox, 
    QMessageBox, QDialog, QGridLayout, QListWidget,
    QFileDialog, QInputDialog, QTextEdit
)
from PyQt6.QtCore import Qt

from AV_Spex.utils import config_edit
from AV_Spex.utils.config_setup import ExiftoolProfile
from AV_Spex.gui.gui_theme_manager import ThemeManager, ThemeableMixin
from AV_Spex.utils import exiftool_import


class CustomExiftoolDialog(QDialog, ThemeableMixin):
    def __init__(self, parent=None, edit_mode=False, profile_name=None):
        super().__init__(parent)
        self.profile = None
        self.edit_mode = edit_mode
        self.original_profile_name = profile_name
        
        if edit_mode:
            self.setWindowTitle(f"Edit Exiftool Profile: {profile_name}")
        else:
            self.setWindowTitle("Custom Exiftool Profile")
        
        self.setModal(True)
        
        # Add theme handling
        self.setup_theme_handling()
        
        # Set minimum size for the dialog
        self.setMinimumSize(600, 700)
        
        # Initialize layout
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
         # Add description
        if edit_mode:
            description = QLabel(f"Edit the exiftool profile: {profile_name}")
        else:
            description = QLabel("Define expected Exiftool values for file validation.")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Profile name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Profile Name:"))
        self.profile_name_input = QLineEdit()
        self.profile_name_input.setPlaceholderText("e.g., Custom HD Profile")
        if edit_mode:
            self.profile_name_input.setText(profile_name)
            self.profile_name_input.setEnabled(False)  # Don't allow renaming in edit mode
        name_layout.addWidget(self.profile_name_input)
        layout.addLayout(name_layout)
        
        # Import section
        import_layout = QHBoxLayout()
        import_button = QPushButton("Import from File...")
        import_button.clicked.connect(self.import_from_file)
        compare_button = QPushButton("Compare with File...")
        compare_button.clicked.connect(self.compare_with_file)
        import_layout.addWidget(import_button)
        import_layout.addWidget(compare_button)
        layout.addLayout(import_layout)
        
        # Scrollable area for fields
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.fields_layout = QGridLayout(scroll_widget)
        self.fields_layout.setSpacing(10)
        scroll.setWidget(scroll_widget)
        scroll.setMinimumHeight(400)
        
        # Create input fields
        self.field_inputs = {}
        self.create_field_inputs()
        
        # Preview section
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(QLabel("Profile Preview:"))
        self.preview_text = QLineEdit()
        self.preview_text.setReadOnly(True)
        preview_layout.addWidget(self.preview_text)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save Profile")
        save_button.clicked.connect(self.on_save_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        # Add all to main layout
        layout.addWidget(scroll)
        layout.addLayout(preview_layout)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.update_preview()
        
    def create_field_inputs(self):
        """Create input fields for all Exiftool properties"""
        # Define field configurations
        fields = [
            # File Information
            ("FileType", "File Type", "MKV", "File format (e.g., MKV, MOV, MP4)"),
            ("FileTypeExtension", "File Extension", "mkv", "File extension without dot"),
            ("MIMEType", "MIME Type", "video/x-matroska", "MIME type of the file"),
            
            # Video Properties
            ("VideoFrameRate", "Frame Rate", "29.97", "Video frame rate in fps"),
            ("ImageWidth", "Width", "720", "Video width in pixels"),
            ("ImageHeight", "Height", "486", "Video height in pixels"),
            ("VideoScanType", "Scan Type", "Interlaced", "Progressive or Interlaced"),
            
            # Display Properties
            ("DisplayWidth", "Display Width", "400", "Display width"),
            ("DisplayHeight", "Display Height", "297", "Display height"),
            ("DisplayUnit", "Display Unit", "Display Aspect Ratio", "Unit for display dimensions"),
            
            # Audio Properties
            ("AudioChannels", "Audio Channels", "2", "Number of audio channels"),
            ("AudioSampleRate", "Sample Rate", "48000", "Audio sample rate in Hz"),
            ("AudioBitsPerSample", "Bits per Sample", "24", "Audio bit depth"),
        ]
        
        row = 0
        for field_name, label_text, placeholder, tooltip in fields:
            # Label
            label = QLabel(f"{label_text}:")
            label.setToolTip(tooltip)
            self.fields_layout.addWidget(label, row, 0)
            
            # Input
            input_field = QLineEdit()
            input_field.setPlaceholderText(placeholder)
            input_field.textChanged.connect(self.update_preview)
            self.fields_layout.addWidget(input_field, row, 1)
            self.field_inputs[field_name] = input_field
            
            row += 1
        
        # Add CodecID section separately since it needs special handling
        codec_label = QLabel("Codec IDs:")
        codec_label.setToolTip("List of accepted audio codec IDs")
        self.fields_layout.addWidget(codec_label, row, 0)
        
        codec_list = QListWidget()
        codec_list.setMaximumHeight(80)
        codec_list.addItems(["A_FLAC", "A_PCM/INT/LIT"])
        
        # Buttons for adding/removing codec items
        codec_button_layout = QVBoxLayout()
        add_codec_btn = QPushButton("+")
        add_codec_btn.setMaximumWidth(30)
        add_codec_btn.clicked.connect(lambda: self.add_codec_item(codec_list))
        remove_codec_btn = QPushButton("-")
        remove_codec_btn.setMaximumWidth(30)
        remove_codec_btn.clicked.connect(lambda: self.remove_codec_item(codec_list))
        codec_button_layout.addWidget(add_codec_btn)
        codec_button_layout.addWidget(remove_codec_btn)
        
        codec_layout = QHBoxLayout()
        codec_layout.addWidget(codec_list)
        codec_layout.addLayout(codec_button_layout)
        
        codec_widget = QWidget()
        codec_widget.setLayout(codec_layout)
        self.fields_layout.addWidget(codec_widget, row, 1)
        self.field_inputs["CodecID"] = codec_list
        
    def import_from_file(self):
        """Import exiftool data from a JSON or text file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Exiftool Output File",
            "",
            "Exiftool Files (*.json *.txt *.log);;All Files (*.*)"
        )
        
        if file_path:
            try:
                # Import the file
                profile = exiftool_import.import_exiftool_file_to_profile(file_path)
                
                if profile:
                    # Load the imported data into the form
                    self.load_profile_data(profile)
                    
                    # Suggest a profile name based on the file
                    import os
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    self.profile_name_input.setText(f"Imported from {base_name}")
                    
                    QMessageBox.information(
                        self,
                        "Import Successful",
                        f"Successfully imported exiftool data from:\n{file_path}"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Import Failed",
                        f"Could not import exiftool data from:\n{file_path}\n\n"
                        "Please check the file format and content."
                    )
                    
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Import Error",
                    f"Error importing file:\n{str(e)}"
                )
    
    def compare_with_file(self):
        """Compare current profile with an exiftool output file"""
        # First check if we have valid profile data
        profile = self.get_exiftool_profile()
        if not profile:
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Exiftool Output File to Compare",
            "",
            "Exiftool Files (*.json *.txt *.log);;All Files (*.*)"
        )
        
        if file_path:
            try:
                # Validate the file against current profile
                validation = exiftool_import.validate_file_against_profile(file_path, profile)
                
                # Show results in a dialog
                self.show_comparison_results(file_path, validation)
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Comparison Error",
                    f"Error comparing file:\n{str(e)}"
                )
    
    def show_comparison_results(self, file_path, validation):
        """Show the comparison results in a dialog"""
        result_dialog = QDialog(self)
        result_dialog.setWindowTitle("Comparison Results")
        result_dialog.setModal(True)
        result_dialog.setMinimumSize(600, 500)
        
        layout = QVBoxLayout()
        
        # Summary
        import os
        summary_label = QLabel(f"<b>File:</b> {os.path.basename(file_path)}<br>"
                               f"<b>Status:</b> {'✅ VALID' if validation['valid'] else '❌ INVALID'}<br>"
                               f"<b>Matching Fields:</b> {validation['matching_fields']}/{validation['total_fields']}")
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)
        
        # Detailed results in text area
        details_text = QTextEdit()
        details_text.setReadOnly(True)
        
        details = []
        
        if validation['matches']:
            details.append("✅ MATCHING FIELDS:")
            for field, values in validation['matches'].items():
                details.append(f"  {field}: {values['actual']}")
            details.append("")
        
        if validation['mismatches']:
            details.append("❌ MISMATCHED FIELDS:")
            for field, values in validation['mismatches'].items():
                details.append(f"  {field}:")
                details.append(f"    Expected: {values['expected']}")
                details.append(f"    Actual: {values['actual']}")
            details.append("")
        
        if validation['missing']:
            details.append("⚠️ MISSING FIELDS:")
            for field, values in validation['missing'].items():
                details.append(f"  {field}: Expected {values['expected']}")
            
        details_text.setPlainText("\n".join(details))
        layout.addWidget(details_text)
        
        # Import button if there are differences
        if validation['mismatches'] or validation['missing']:
            import_btn = QPushButton("Import These Values")
            import_btn.clicked.connect(lambda: self.import_from_validation(file_path, result_dialog))
            layout.addWidget(import_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(result_dialog.accept)
        layout.addWidget(close_btn)
        
        result_dialog.setLayout(layout)
        result_dialog.exec()
    
    def import_from_validation(self, file_path, dialog):
        """Import values from a file after comparison"""
        try:
            profile = exiftool_import.import_exiftool_file_to_profile(file_path)
            if profile:
                self.load_profile_data(profile)
                dialog.accept()
                QMessageBox.information(self, "Import Successful", "Values imported from file")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Error importing: {str(e)}")
    
    def load_profile_data(self, profile_data):
        """Load profile data into the form fields"""
        # Load field values
        for field_name, input_widget in self.field_inputs.items():
            if hasattr(profile_data, field_name):
                value = getattr(profile_data, field_name)
                if field_name == "CodecID":
                    # Handle list widget
                    input_widget.clear()
                    if isinstance(value, list):
                        input_widget.addItems(value)
                else:
                    # Handle text input
                    input_widget.setText(str(value) if value else "")
                    
        self.update_preview()
    
    def add_codec_item(self, list_widget):
        """Add a new codec ID to the list"""
        from PyQt6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(self, "Add Codec ID", "Enter codec ID:")
        if ok and text:
            list_widget.addItem(text)
            self.update_preview()
            
    def remove_codec_item(self, list_widget):
        """Remove selected codec ID from the list"""
        current_item = list_widget.currentItem()
        if current_item:
            list_widget.takeItem(list_widget.row(current_item))
            self.update_preview()
            
    def update_preview(self):
        """Update the profile preview"""
        profile_name = self.profile_name_input.text() or "Unnamed Profile"
        
        # Get first few field values for preview
        file_type = self.field_inputs.get("FileType")
        if file_type and isinstance(file_type, QLineEdit):
            file_type_val = file_type.text() or "N/A"
        else:
            file_type_val = "N/A"
            
        width = self.field_inputs.get("ImageWidth")
        if width and isinstance(width, QLineEdit):
            width_val = width.text() or "N/A"
        else:
            width_val = "N/A"
            
        height = self.field_inputs.get("ImageHeight")
        if height and isinstance(height, QLineEdit):
            height_val = height.text() or "N/A"
        else:
            height_val = "N/A"
            
        preview = f"{profile_name}: {file_type_val} {width_val}x{height_val}"
        self.preview_text.setText(preview)
        
    def get_exiftool_profile(self):
        """Get the exiftool profile as an ExiftoolProfile dataclass"""
        if not self.profile_name_input.text():
            QMessageBox.warning(self, "Validation Error", "Profile name is required.")
            return None
            
        # Collect values from inputs
        profile_data = {}
        
        for field_name, input_widget in self.field_inputs.items():
            if field_name == "CodecID":
                # Handle list widget
                items = []
                for i in range(input_widget.count()):
                    items.append(input_widget.item(i).text())
                profile_data[field_name] = items
            else:
                # Handle text input
                profile_data[field_name] = input_widget.text()
                
        # Validate required fields
        required_fields = ["FileType", "FileTypeExtension", "MIMEType"]
        for field in required_fields:
            if not profile_data.get(field):
                QMessageBox.warning(self, "Validation Error", f"{field} is required.")
                return None
                
        # Create and return ExiftoolProfile instance
        return ExiftoolProfile(**profile_data)
        
    def on_save_clicked(self):
        """Handle save button click"""
        profile = self.get_exiftool_profile()
        if profile:
            try:
                    # Store the profile with the name
                profile_name = self.original_profile_name if self.edit_mode else self.profile_name_input.text()
                self.profile = {
                    'name': profile_name,
                    'data': profile,
                    'is_edit': self.edit_mode
                }
                self.accept()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save profile: {str(e)}")
                
    def get_profile(self):
        """Return the stored profile"""
        return self.profile
        
    def load_existing_profile(self, profile_name, profile_data):
        """Load an existing profile into the dialog"""
        self.profile_name_input.setText(profile_name)
        
        # Load field values
        for field_name, input_widget in self.field_inputs.items():
            if hasattr(profile_data, field_name):
                value = getattr(profile_data, field_name)
                if field_name == "CodecID":
                    # Handle list widget
                    input_widget.clear()
                    if isinstance(value, list):
                        input_widget.addItems(value)
                else:
                    # Handle text input
                    input_widget.setText(str(value) if value else "")
                    
        self.update_preview()
        
    def on_theme_changed(self, palette):
        """Apply theme changes to this dialog"""
        self.setPalette(palette)
        
    def closeEvent(self, event):
        """Clean up theme connections before closing"""
        self.cleanup_theme_handling()
        super().closeEvent(event)