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
        self.setMinimumSize(700, 800)
        
        # Initialize layout
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Add description
        if edit_mode:
            description = QLabel(f"Edit the exiftool profile: {profile_name}")
        else:
            description = QLabel("Define expected Exiftool values for file validation. Each field can have multiple values.")
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
        scroll_widget.setAutoFillBackground(False)
        scroll_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        scroll_widget.setStyleSheet("QWidget { background-color: transparent; }")
        self.fields_layout = QGridLayout(scroll_widget)
        self.fields_layout.setSpacing(5)
        self.fields_layout.setContentsMargins(5, 5, 5, 5)
        scroll.setWidget(scroll_widget)
        scroll.setMinimumHeight(500)
        
        # Create input fields - now stores list of QLineEdit widgets per field
        self.field_inputs = {}
        self.field_containers = {}  # Store the container widgets for each field
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
        """Create input fields for all Exiftool properties - all support multiple values"""
        # Define field configurations
        fields = [
            # File Information
            ("FileType", "File Type", ["MKV"], "File format (e.g., MKV, MOV, MP4)"),
            ("FileTypeExtension", "File Extension", ["mkv"], "File extension without dot"),
            ("MIMEType", "MIME Type", ["video/x-matroska"], "MIME type of the file"),
            
            # Video Properties
            ("VideoFrameRate", "Frame Rate", ["29.97"], "Video frame rate in fps"),
            ("ImageWidth", "Width", ["720"], "Video width in pixels"),
            ("ImageHeight", "Height", ["486"], "Video height in pixels"),
            ("VideoScanType", "Scan Type", ["Interlaced"], "Progressive or Interlaced"),
            
            # Display Properties
            ("DisplayWidth", "Display Width", ["400"], "Display width"),
            ("DisplayHeight", "Display Height", ["297"], "Display height"),
            ("DisplayUnit", "Display Unit", ["Display Aspect Ratio"], "Unit for display dimensions"),
            
            # Audio Properties
            ("AudioChannels", "Audio Channels", ["2"], "Number of audio channels"),
            ("AudioSampleRate", "Sample Rate", ["48000"], "Audio sample rate in Hz"),
            ("AudioBitsPerSample", "Bits per Sample", ["24"], "Audio bit depth"),
            
            # Codec IDs
            ("CodecID", "Codec IDs", ["A_FLAC", "A_PCM/INT/LIT"], "List of accepted audio codec IDs"),
        ]
        
        row = 0
        for field_name, label_text, default_values, tooltip in fields:
            # Create horizontal layout for the entire field (label + inputs + buttons)
            field_layout = QHBoxLayout()
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(5)
            
            # Label - now part of the field_layout
            label = QLabel(f"{label_text}:")
            label.setToolTip(tooltip)
            label.setMinimumWidth(120)  # Optional: set a minimum width for alignment
            field_layout.addWidget(label)
            
            # Create vertical layout for MULTIPLE line edits
            inputs_layout = QVBoxLayout()
            inputs_layout.setContentsMargins(0, 0, 0, 0)
            inputs_layout.setSpacing(5)
            
            # Store list of line edits for this field
            self.field_inputs[field_name] = []
            self.field_containers[field_name] = inputs_layout
            
            # Add first line edit
            first_value = default_values[0] if default_values else ""
            line_edit = QLineEdit()
            line_edit.setText(first_value)
            line_edit.setPlaceholderText(f"Enter {field_name} value...")
            line_edit.textChanged.connect(self.update_preview)
            inputs_layout.addWidget(line_edit)
            self.field_inputs[field_name].append(line_edit)
            
            # Add inputs layout to field layout
            field_layout.addLayout(inputs_layout, 1)  # Stretch factor 1
            
            # Buttons for adding/removing items
            add_btn = QPushButton("+")
            add_btn.setMaximumWidth(30)
            add_btn.setMaximumHeight(25)
            add_btn.setToolTip(f"Add {label_text}")
            add_btn.setStyleSheet("QPushButton { background-color: transparent; }")
            add_btn.clicked.connect(lambda checked, fn=field_name: self.add_textbox_row(fn))
            
            remove_btn = QPushButton("-")
            remove_btn.setMaximumWidth(30)
            remove_btn.setMaximumHeight(25)
            remove_btn.setToolTip(f"Remove last {label_text}")
            remove_btn.setStyleSheet("QPushButton { background-color: transparent; }")
            remove_btn.clicked.connect(lambda checked, fn=field_name: self.remove_textbox_row(fn))
            
            field_layout.addWidget(add_btn)
            field_layout.addWidget(remove_btn)
            
            # Create widget to hold this layout - CRITICAL: no background
            field_widget = QWidget()
            field_widget.setLayout(field_layout)
            field_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
            field_widget.setAutoFillBackground(False)
            
            # Add to grid layout, spanning both columns
            self.fields_layout.addWidget(field_widget, row, 0, 1, 2)
            
            row += 1

    def add_textbox_row(self, field_name, value="", container_layout=None):
        """Add a new text box row for a field"""
        if container_layout is None:
            container_layout = self.field_containers[field_name]
        
        line_edit = QLineEdit()
        line_edit.setText(value)
        line_edit.setPlaceholderText(f"Enter {field_name} value...")
        line_edit.textChanged.connect(self.update_preview)
        
        container_layout.addWidget(line_edit)
        self.field_inputs[field_name].append(line_edit)
        
        # Update preview if it exists (during init it doesn't exist yet)
        if hasattr(self, 'preview_text'):
            self.update_preview()

    def remove_textbox_row(self, field_name):
        """Remove the last text box row for a field"""
        if len(self.field_inputs[field_name]) > 1:  # Keep at least one text box
            line_edit = self.field_inputs[field_name].pop()
            line_edit.deleteLater()
            # Update preview if it exists
            if hasattr(self, 'preview_text'):
                self.update_preview()
    
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
        # Clear existing text boxes and add new ones based on profile data
        for field_name, line_edits in self.field_inputs.items():
            container_layout = self.field_containers[field_name]
            
            # Remove all widgets from the layout
            while container_layout.count():
                item = container_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            self.field_inputs[field_name].clear()
            
            # Get the value from profile
            if hasattr(profile_data, field_name):
                value = getattr(profile_data, field_name)
                
                # Convert to list if not already
                if isinstance(value, list):
                    values = value
                elif value:
                    values = [value]
                else:
                    values = [""]
                
                # Add text boxes for each value
                for val in values:
                    line_edit = QLineEdit()
                    line_edit.setText(str(val))
                    line_edit.setPlaceholderText(f"Enter {field_name} value...")
                    line_edit.textChanged.connect(self.update_preview)
                    container_layout.addWidget(line_edit)
                    self.field_inputs[field_name].append(line_edit)
            else:
                # Field not in profile, add one empty text box
                line_edit = QLineEdit()
                line_edit.setText("")
                line_edit.setPlaceholderText(f"Enter {field_name} value...")
                line_edit.textChanged.connect(self.update_preview)
                container_layout.addWidget(line_edit)
                self.field_inputs[field_name].append(line_edit)
        
        # Update preview if it exists
        if hasattr(self, 'preview_text'):
            self.update_preview()
    
    def update_preview(self):
        """Update the profile preview"""
        profile_name = self.profile_name_input.text() or "Unnamed Profile"
        
        # Get first field value for preview
        file_type_val = "N/A"
        if "FileType" in self.field_inputs and self.field_inputs["FileType"]:
            text = self.field_inputs["FileType"][0].text()
            if text:
                file_type_val = text
            
        width_val = "N/A"
        if "ImageWidth" in self.field_inputs and self.field_inputs["ImageWidth"]:
            text = self.field_inputs["ImageWidth"][0].text()
            if text:
                width_val = text
            
        height_val = "N/A"
        if "ImageHeight" in self.field_inputs and self.field_inputs["ImageHeight"]:
            text = self.field_inputs["ImageHeight"][0].text()
            if text:
                height_val = text
            
        preview = f"{profile_name}: {file_type_val} {width_val}x{height_val}"
        self.preview_text.setText(preview)
        
    def get_exiftool_profile(self):
        """Get the exiftool profile as an ExiftoolProfile dataclass"""
        if not self.profile_name_input.text():
            QMessageBox.warning(self, "Validation Error", "Profile name is required.")
            return None
            
        # Collect values from text boxes
        profile_data = {}
        
        for field_name, line_edits in self.field_inputs.items():
            values = []
            for line_edit in line_edits:
                text = line_edit.text().strip()
                if text:  # Only add non-empty values
                    values.append(text)
            
            # Store as list if multiple values, single value if one, empty string if none
            if len(values) > 1:
                profile_data[field_name] = values
            elif len(values) == 1:
                profile_data[field_name] = values[0]
            else:
                profile_data[field_name] = ""
                
        # Validate required fields
        required_fields = ["FileType", "FileTypeExtension", "MIMEType"]
        for field in required_fields:
            value = profile_data.get(field)
            if not value or (isinstance(value, list) and not value):
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
        self.load_profile_data(profile_data)
        
    def on_theme_changed(self, palette):
        """Apply theme changes to this dialog"""
        self.setPalette(palette)
        
    def closeEvent(self, event):
        """Clean up theme connections before closing"""
        self.cleanup_theme_handling()
        super().closeEvent(event)