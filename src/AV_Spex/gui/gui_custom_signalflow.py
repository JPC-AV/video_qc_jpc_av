from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, 
    QScrollArea, QPushButton, QComboBox, 
    QMessageBox, QDialog, QCheckBox, QGridLayout
)

from ..utils import config_edit
from ..utils.config_setup import SpexConfig, EncoderSettings

from ..gui.gui_theme_manager import ThemeManager, ThemeableMixin

class CustomSignalflowDialog(QDialog, ThemeableMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.profile = None
        self.setWindowTitle("Custom Signal Flow Profile")
        self.setModal(True)

        # Add theme handling
        self.setup_theme_handling()

        # Set minimum size for the dialog
        self.setMinimumSize(550, 650)  # Width: 550px, Height: 650px
        
        # Initialize layout
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Add description
        description = QLabel("Define your expected signal flow profile by specifying the video equipment chain.")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Scrollable area for configuration
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.config_layout = QVBoxLayout(scroll_widget)
        self.config_layout.setSpacing(10)
        self.config_layout.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(scroll_widget)
        scroll.setMinimumHeight(400)
        
        # Setup configuration sections
        self.setup_source_vtr_section()
        self.setup_tbc_section()
        self.setup_adc_section()
        self.setup_capture_device_section()
        self.setup_computer_section()
        
        # Preview section
        preview_group = QWidget()
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.addWidget(QLabel("<b>Profile Preview:</b>"))
        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_label)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        self.profile_name_label = QLabel("Profile Name:")
        self.profile_name_input = QLineEdit()
        self.profile_name_input.setPlaceholderText("Enter a name for this profile")
        
        save_button = QPushButton("Save Profile")
        save_button.clicked.connect(self.on_save_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(self.profile_name_label)
        name_layout.addWidget(self.profile_name_input)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        # Add all layouts to main layout
        layout.addWidget(scroll)
        layout.addWidget(preview_group)
        layout.addLayout(name_layout)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.update_preview()
        
        # Style with current theme
        theme_manager = ThemeManager.instance()
        theme_manager.style_buttons(self)
        
    def setup_source_vtr_section(self):
        """Setup the Source VTR section"""
        section_label = QLabel("<b>Source VTR:</b>")
        self.config_layout.addWidget(section_label)
        
        # Grid layout for VTR fields
        vtr_grid = QGridLayout()
        vtr_grid.setColumnStretch(1, 1)
        
        # VTR Model
        row = 0
        vtr_grid.addWidget(QLabel("VTR Model:"), row, 0)
        self.vtr_model_combo = QComboBox()
        self.vtr_model_combo.addItems([
            "SVO5800 (S-VHS VCR)",
            "BVH3100 (1\" Type C)",
            "DVW-A500 (Digital Betacam)",
            "UVW-1800 (Betacam SP)",
            "BR-S525U (S-VHS)",
            "PVW-2800 (Betacam SP)",
            "Other"
        ])
        self.vtr_model_combo.currentIndexChanged.connect(self.update_preview)
        vtr_grid.addWidget(self.vtr_model_combo, row, 1)
        
        # Custom VTR model input (enabled when "Other" is selected)
        row += 1
        vtr_grid.addWidget(QLabel("Custom VTR Model:"), row, 0)
        self.custom_vtr_input = QLineEdit()
        self.custom_vtr_input.setPlaceholderText("Enter custom VTR model")
        self.custom_vtr_input.setEnabled(False)
        self.custom_vtr_input.textChanged.connect(self.update_preview)
        vtr_grid.addWidget(self.custom_vtr_input, row, 1)
        
        # Enable custom input only when "Other" is selected
        self.vtr_model_combo.currentIndexChanged.connect(
            lambda idx: self.custom_vtr_input.setEnabled(idx == self.vtr_model_combo.count() - 1)
        )
        
        # Serial Number
        row += 1
        vtr_grid.addWidget(QLabel("Serial Number:"), row, 0)
        self.vtr_sn_input = QLineEdit()
        self.vtr_sn_input.setPlaceholderText("SN 12345")
        self.vtr_sn_input.textChanged.connect(self.update_preview)
        vtr_grid.addWidget(self.vtr_sn_input, row, 1)
        
        # Connection Type
        row += 1
        vtr_grid.addWidget(QLabel("Connection Type:"), row, 0)
        self.vtr_connection_combo = QComboBox()
        self.vtr_connection_combo.addItems([
            "composite", 
            "component", 
            "S-Video", 
            "SDI"
        ])
        self.vtr_connection_combo.currentIndexChanged.connect(self.update_preview)
        vtr_grid.addWidget(self.vtr_connection_combo, row, 1)
        
        # Audio Type
        row += 1
        vtr_grid.addWidget(QLabel("Audio Type:"), row, 0)
        self.vtr_audio_combo = QComboBox()
        self.vtr_audio_combo.addItems([
            "analog balanced", 
            "analog unbalanced", 
            "digital embedded", 
            "digital AES/EBU"
        ])
        self.vtr_audio_combo.currentIndexChanged.connect(self.update_preview)
        vtr_grid.addWidget(self.vtr_audio_combo, row, 1)
        
        self.config_layout.addLayout(vtr_grid)
        
    def setup_tbc_section(self):
        """Setup the TBC section"""
        section_label = QLabel("<b>Time Base Corrector/Frame Sync:</b>")
        self.config_layout.addWidget(section_label)
        
        # Grid layout for TBC fields
        tbc_grid = QGridLayout()
        tbc_grid.setColumnStretch(1, 1)
        
        # TBC Model
        row = 0
        tbc_grid.addWidget(QLabel("TBC Model:"), row, 0)
        self.tbc_model_combo = QComboBox()
        self.tbc_model_combo.addItems([
            "Internal TBC (same as VTR)",
            "DPS575 with flash firmware",
            "AVT-8710 (Digital TBC)",
            "FA-300 (Digital TBC)",
            "FOR-A FA-310 (Digital TBC)",
            "None",
            "Other"
        ])
        self.tbc_model_combo.currentIndexChanged.connect(self.update_preview)
        tbc_grid.addWidget(self.tbc_model_combo, row, 1)
        
        # Custom TBC model input (enabled when "Other" is selected)
        row += 1
        tbc_grid.addWidget(QLabel("Custom TBC Model:"), row, 0)
        self.custom_tbc_input = QLineEdit()
        self.custom_tbc_input.setPlaceholderText("Enter custom TBC model")
        self.custom_tbc_input.setEnabled(False)
        self.custom_tbc_input.textChanged.connect(self.update_preview)
        tbc_grid.addWidget(self.custom_tbc_input, row, 1)
        
        # Enable custom input only when "Other" is selected
        self.tbc_model_combo.currentIndexChanged.connect(
            lambda idx: self.custom_tbc_input.setEnabled(idx == self.tbc_model_combo.count() - 1)
        )
        
        # Firmware Version
        row += 1
        tbc_grid.addWidget(QLabel("Firmware Version:"), row, 0)
        self.tbc_firmware_input = QLineEdit()
        self.tbc_firmware_input.setPlaceholderText("h2.16")
        self.tbc_firmware_input.textChanged.connect(self.update_preview)
        tbc_grid.addWidget(self.tbc_firmware_input, row, 1)
        
        # Serial Number
        row += 1
        tbc_grid.addWidget(QLabel("Serial Number:"), row, 0)
        self.tbc_sn_input = QLineEdit()
        self.tbc_sn_input.setPlaceholderText("SN 12345")
        self.tbc_sn_input.textChanged.connect(self.update_preview)
        tbc_grid.addWidget(self.tbc_sn_input, row, 1)
        
        # Output Connection
        row += 1
        tbc_grid.addWidget(QLabel("Output Connection:"), row, 0)
        self.tbc_connection_combo = QComboBox()
        self.tbc_connection_combo.addItems([
            "SDI", 
            "composite", 
            "component", 
            "S-Video"
        ])
        self.tbc_connection_combo.currentIndexChanged.connect(self.update_preview)
        tbc_grid.addWidget(self.tbc_connection_combo, row, 1)
        
        # Audio Output
        row += 1
        tbc_grid.addWidget(QLabel("Audio Output:"), row, 0)
        self.tbc_audio_combo = QComboBox()
        self.tbc_audio_combo.addItems([
            "audio embedded", 
            "audio pass-through", 
            "analog balanced",
            "analog unbalanced"
        ])
        self.tbc_audio_combo.currentIndexChanged.connect(self.update_preview)
        tbc_grid.addWidget(self.tbc_audio_combo, row, 1)
        
        self.config_layout.addLayout(tbc_grid)
        
    def setup_adc_section(self):
        """Setup the A/D Converter section"""
        section_label = QLabel("<b>A/D Converter (if separate from TBC):</b>")
        self.config_layout.addWidget(section_label)
        
        # Checkbox for separate ADC
        self.separate_adc_check = QCheckBox("Use separate A/D Converter")
        self.separate_adc_check.stateChanged.connect(self.toggle_adc_section)
        self.separate_adc_check.stateChanged.connect(self.update_preview)
        self.config_layout.addWidget(self.separate_adc_check)
        
        # Grid layout for ADC fields
        adc_widget = QWidget()
        self.adc_grid = QGridLayout(adc_widget)
        self.adc_grid.setColumnStretch(1, 1)
        
        # ADC Model
        row = 0
        self.adc_grid.addWidget(QLabel("ADC Model:"), row, 0)
        self.adc_model_combo = QComboBox()
        self.adc_model_combo.addItems([
            "Leitch DPS575",
            "Blackmagic Design Mini Converter", 
            "AJA HD10C2", 
            "Decimator Design MD-LX",
            "Other"
        ])
        self.adc_model_combo.currentIndexChanged.connect(self.update_preview)
        self.adc_grid.addWidget(self.adc_model_combo, row, 1)
        
        # Custom ADC model input
        row += 1
        self.adc_grid.addWidget(QLabel("Custom ADC Model:"), row, 0)
        self.custom_adc_input = QLineEdit()
        self.custom_adc_input.setPlaceholderText("Enter custom ADC model")
        self.custom_adc_input.setEnabled(False)
        self.custom_adc_input.textChanged.connect(self.update_preview)
        self.adc_grid.addWidget(self.custom_adc_input, row, 1)
        
        # Enable custom input only when "Other" is selected
        self.adc_model_combo.currentIndexChanged.connect(
            lambda idx: self.custom_adc_input.setEnabled(idx == self.adc_model_combo.count() - 1)
        )
        
        # Serial Number
        row += 1
        self.adc_grid.addWidget(QLabel("Serial Number:"), row, 0)
        self.adc_sn_input = QLineEdit()
        self.adc_sn_input.setPlaceholderText("SN 12345")
        self.adc_sn_input.textChanged.connect(self.update_preview)
        self.adc_grid.addWidget(self.adc_sn_input, row, 1)
        
        # Output Connection
        row += 1
        self.adc_grid.addWidget(QLabel("Output Connection:"), row, 0)
        self.adc_connection_combo = QComboBox()
        self.adc_connection_combo.addItems([
            "SDI", 
            "HDMI", 
            "component"
        ])
        self.adc_connection_combo.currentIndexChanged.connect(self.update_preview)
        self.adc_grid.addWidget(self.adc_connection_combo, row, 1)
        
        # By default, disable the ADC section
        adc_widget.setEnabled(False)
        self.adc_widget = adc_widget
        self.config_layout.addWidget(adc_widget)
        
    def toggle_adc_section(self, state):
        """Enable or disable the ADC section based on checkbox state"""
        self.adc_widget.setEnabled(state)
        
    def setup_capture_device_section(self):
        """Setup the Capture Device section"""
        section_label = QLabel("<b>Capture Device:</b>")
        self.config_layout.addWidget(section_label)
        
        # Grid layout for Capture Device fields
        capture_grid = QGridLayout()
        capture_grid.setColumnStretch(1, 1)
        
        # Capture Device Model
        row = 0
        capture_grid.addWidget(QLabel("Capture Device:"), row, 0)
        self.capture_model_combo = QComboBox()
        self.capture_model_combo.addItems([
            "Blackmagic Design UltraStudio",
            "Blackmagic Intensity Pro",
            "AJA Kona LHe Plus", 
            "Magewell USB Capture",
            "Blackmagic DeckLink",
            "AJA U-TAP",
            "Other"
        ])
        self.capture_model_combo.currentIndexChanged.connect(self.update_preview)
        capture_grid.addWidget(self.capture_model_combo, row, 1)
        
        # Custom Capture Device model input
        row += 1
        capture_grid.addWidget(QLabel("Custom Device:"), row, 0)
        self.custom_capture_input = QLineEdit()
        self.custom_capture_input.setPlaceholderText("Enter custom device model")
        self.custom_capture_input.setEnabled(False)
        self.custom_capture_input.textChanged.connect(self.update_preview)
        capture_grid.addWidget(self.custom_capture_input, row, 1)
        
        # Enable custom input only when "Other" is selected
        self.capture_model_combo.currentIndexChanged.connect(
            lambda idx: self.custom_capture_input.setEnabled(idx == self.capture_model_combo.count() - 1)
        )
        
        # Serial Number
        row += 1
        capture_grid.addWidget(QLabel("Serial Number:"), row, 0)
        self.capture_sn_input = QLineEdit()
        self.capture_sn_input.setPlaceholderText("SN 12345")
        self.capture_sn_input.textChanged.connect(self.update_preview)
        capture_grid.addWidget(self.capture_sn_input, row, 1)
        
        # Interface Type
        row += 1
        capture_grid.addWidget(QLabel("Interface:"), row, 0)
        self.capture_interface_combo = QComboBox()
        self.capture_interface_combo.addItems([
            "Thunderbolt", 
            "PCIe", 
            "USB 3.0", 
            "USB-C"
        ])
        self.capture_interface_combo.currentIndexChanged.connect(self.update_preview)
        capture_grid.addWidget(self.capture_interface_combo, row, 1)
        
        self.config_layout.addLayout(capture_grid)
        
    def setup_computer_section(self):
        """Setup the Computer section"""
        section_label = QLabel("<b>Computer:</b>")
        self.config_layout.addWidget(section_label)
        
        # Grid layout for Computer fields
        computer_grid = QGridLayout()
        computer_grid.setColumnStretch(1, 1)
        
        # Computer Model
        row = 0
        computer_grid.addWidget(QLabel("Computer Model:"), row, 0)
        self.computer_model_combo = QComboBox()
        self.computer_model_combo.addItems([
            "Mac Mini", 
            "Mac Studio", 
            "Mac Pro", 
            "Custom PC",
            "Other"
        ])
        self.computer_model_combo.currentIndexChanged.connect(self.update_preview)
        computer_grid.addWidget(self.computer_model_combo, row, 1)
        
        # Custom Computer model input
        row += 1
        computer_grid.addWidget(QLabel("Custom Computer:"), row, 0)
        self.custom_computer_input = QLineEdit()
        self.custom_computer_input.setPlaceholderText("Enter custom computer model")
        self.custom_computer_input.setEnabled(False)
        self.custom_computer_input.textChanged.connect(self.update_preview)
        computer_grid.addWidget(self.custom_computer_input, row, 1)
        
        # Enable custom input only when "Other" is selected
        self.computer_model_combo.currentIndexChanged.connect(
            lambda idx: self.custom_computer_input.setEnabled(idx == self.computer_model_combo.count() - 1)
        )
        
        # Computer Specs
        row += 1
        computer_grid.addWidget(QLabel("Processor:"), row, 0)
        self.computer_cpu_input = QLineEdit()
        self.computer_cpu_input.setPlaceholderText("e.g., Apple M2 Pro chip")
        self.computer_cpu_input.textChanged.connect(self.update_preview)
        computer_grid.addWidget(self.computer_cpu_input, row, 1)
        
        # Serial Number
        row += 1
        computer_grid.addWidget(QLabel("Serial Number:"), row, 0)
        self.computer_sn_input = QLineEdit()
        self.computer_sn_input.setPlaceholderText("SN 12345")
        self.computer_sn_input.textChanged.connect(self.update_preview)
        computer_grid.addWidget(self.computer_sn_input, row, 1)
        
        # OS Version
        row += 1
        computer_grid.addWidget(QLabel("OS Version:"), row, 0)
        self.computer_os_input = QLineEdit()
        self.computer_os_input.setPlaceholderText("e.g., macOS 14.5")
        self.computer_os_input.textChanged.connect(self.update_preview)
        computer_grid.addWidget(self.computer_os_input, row, 1)
        
        # Software
        row += 1
        computer_grid.addWidget(QLabel("Software:"), row, 0)
        self.computer_software_input = QLineEdit()
        self.computer_software_input.setPlaceholderText("e.g., vrecord v2023-08-07, ffmpeg")
        self.computer_software_input.textChanged.connect(self.update_preview)
        computer_grid.addWidget(self.computer_software_input, row, 1)
        
        self.config_layout.addLayout(computer_grid)
        
    def update_preview(self):
        """Update the preview based on current settings"""
        # Build preview text for Source VTR
        preview_parts = []
        
        # Source VTR
        vtr_model = self.vtr_model_combo.currentText().split(" (")[0] if "(" in self.vtr_model_combo.currentText() else self.vtr_model_combo.currentText()
        if vtr_model == "Other" and self.custom_vtr_input.text():
            vtr_model = self.custom_vtr_input.text()
            
        vtr_parts = [vtr_model]
        if self.vtr_sn_input.text():
            vtr_parts.append(f"SN {self.vtr_sn_input.text()}")
        vtr_parts.append(self.vtr_connection_combo.currentText())
        vtr_parts.append(self.vtr_audio_combo.currentText())
        
        vtr_text = f"Source_VTR: {', '.join(vtr_parts)}"
        preview_parts.append(vtr_text)
        
        # TBC/Framesync
        tbc_model = self.tbc_model_combo.currentText()
        if tbc_model == "Internal TBC (same as VTR)":
            tbc_model = vtr_model
        elif tbc_model == "Other" and self.custom_tbc_input.text():
            tbc_model = self.custom_tbc_input.text()
        elif "(" in tbc_model:
            tbc_model = tbc_model.split(" (")[0]
            
        tbc_parts = []
        if tbc_model != "None":
            tbc_parts = [tbc_model]
            if self.tbc_firmware_input.text():
                tbc_parts[0] += f" with flash firmware {self.tbc_firmware_input.text()}"
            if self.tbc_sn_input.text():
                tbc_parts.append(f"SN {self.tbc_sn_input.text()}")
            tbc_parts.append(self.tbc_connection_combo.currentText())
            tbc_parts.append(self.tbc_audio_combo.currentText())
            
            tbc_text = f"TBC_Framesync: {', '.join(tbc_parts)}"
            preview_parts.append(tbc_text)
        
        # ADC (if separate checkbox is checked or TBC used as ADC)
        if self.separate_adc_check.isChecked():
            adc_model = self.adc_model_combo.currentText()
            if adc_model == "Other" and self.custom_adc_input.text():
                adc_model = self.custom_adc_input.text()
                
            adc_parts = [adc_model]
            if self.adc_sn_input.text():
                adc_parts.append(f"SN {self.adc_sn_input.text()}")
            adc_parts.append(self.adc_connection_combo.currentText())
            
            adc_text = f"ADC: {', '.join(adc_parts)}"
            preview_parts.append(adc_text)
        elif tbc_model != "None":
            # If separate ADC not checked but TBC exists, show TBC as ADC in preview
            adc_text = f"ADC: {', '.join(tbc_parts)} (same as TBC)"
            preview_parts.append(adc_text)
        
        # Capture Device
        capture_model = self.capture_model_combo.currentText()
        if capture_model == "Other" and self.custom_capture_input.text():
            capture_model = self.custom_capture_input.text()
            
        capture_parts = [capture_model]
        if self.capture_sn_input.text():
            capture_parts.append(f"SN {self.capture_sn_input.text()}")
        capture_parts.append(self.capture_interface_combo.currentText())
        
        capture_text = f"Capture_Device: {', '.join(capture_parts)}"
        preview_parts.append(capture_text)
        
        # Computer
        computer_model = self.computer_model_combo.currentText()
        if computer_model == "Other" and self.custom_computer_input.text():
            computer_model = self.custom_computer_input.text()
            
        computer_parts = [f"{computer_model}"]
        if self.computer_cpu_input.text():
            computer_parts.append(self.computer_cpu_input.text())
        if self.computer_sn_input.text():
            computer_parts.append(f"SN {self.computer_sn_input.text()}")
        if self.computer_os_input.text():
            computer_parts.append(f"OS {self.computer_os_input.text()}")
        if self.computer_software_input.text():
            computer_parts.append(self.computer_software_input.text())
        
        computer_text = f"Computer: {', '.join(computer_parts)}"
        preview_parts.append(computer_text)
        
        # Join all parts with line breaks
        preview_text = "\n".join(preview_parts)
        self.preview_label.setText(preview_text)
        
    def get_profile(self):
        """Get the signal flow profile as a dictionary structure"""
        if not self.profile_name_input.text():
            QMessageBox.warning(self, "Validation Error", "Profile name is required.")
            return None
            
        # Create the profile dictionary
        profile = {
            "name": self.profile_name_input.text()
        }
        
        # Source VTR
        vtr_model = self.vtr_model_combo.currentText().split(" (")[0] if "(" in self.vtr_model_combo.currentText() else self.vtr_model_combo.currentText()
        if vtr_model == "Other" and self.custom_vtr_input.text():
            vtr_model = self.custom_vtr_input.text()
            
        vtr_parts = [vtr_model]
        if self.vtr_sn_input.text():
            vtr_parts.append(f"SN {self.vtr_sn_input.text()}")
        vtr_parts.append(self.vtr_connection_combo.currentText())
        vtr_parts.append(self.vtr_audio_combo.currentText())
        
        profile["Source_VTR"] = vtr_parts
        
        # TBC/Framesync
        tbc_model = self.tbc_model_combo.currentText()
        if tbc_model == "Internal TBC (same as VTR)":
            tbc_model = vtr_model
        elif tbc_model == "Other" and self.custom_tbc_input.text():
            tbc_model = self.custom_tbc_input.text()
        elif "(" in tbc_model:
            tbc_model = tbc_model.split(" (")[0]
            
        if tbc_model != "None":
            tbc_parts = [tbc_model]
            if self.tbc_firmware_input.text():
                tbc_parts[0] += f" with flash firmware {self.tbc_firmware_input.text()}"
            if self.tbc_sn_input.text():
                tbc_parts.append(f"SN {self.tbc_sn_input.text()}")
            tbc_parts.append(self.tbc_connection_combo.currentText())
            tbc_parts.append(self.tbc_audio_combo.currentText())
            
            profile["TBC_Framesync"] = tbc_parts
        
        # ADC (if separate checkbox is checked, use user-specified ADC, 
        # otherwise use TBC info as ADC if TBC exists)
        if self.separate_adc_check.isChecked():
            adc_model = self.adc_model_combo.currentText()
            if adc_model == "Other" and self.custom_adc_input.text():
                adc_model = self.custom_adc_input.text()
                
            adc_parts = [adc_model]
            if self.adc_sn_input.text():
                adc_parts.append(f"SN {self.adc_sn_input.text()}")
            adc_parts.append(self.adc_connection_combo.currentText())
            
            profile["ADC"] = adc_parts
        elif "TBC_Framesync" in profile and tbc_model != "None":
            # If separate ADC not checked but TBC exists, use TBC as ADC
            adc_parts = profile["TBC_Framesync"].copy()
            profile["ADC"] = adc_parts
        
        # Capture Device
        capture_model = self.capture_model_combo.currentText()
        if capture_model == "Other" and self.custom_capture_input.text():
            capture_model = self.custom_capture_input.text()
            
        capture_parts = [capture_model]
        if self.capture_sn_input.text():
            capture_parts.append(f"SN {self.capture_sn_input.text()}")
        capture_parts.append(self.capture_interface_combo.currentText())
        
        profile["Capture_Device"] = capture_parts
        
        # Computer
        computer_model = self.computer_model_combo.currentText()
        if computer_model == "Other" and self.custom_computer_input.text():
            computer_model = self.custom_computer_input.text()
            
        computer_parts = [f"{computer_model}"]
        if self.computer_cpu_input.text():
            computer_parts.append(self.computer_cpu_input.text())
        if self.computer_sn_input.text():
            computer_parts.append(f"SN {self.computer_sn_input.text()}")
        if self.computer_os_input.text():
            computer_parts.append(f"OS {self.computer_os_input.text()}")
        if self.computer_software_input.text():
            computer_parts.append(self.computer_software_input.text())
        
        profile["Computer"] = computer_parts
        
        # Store the complete profile
        self.profile = profile
        return profile
        
    def on_save_clicked(self):
        """Handle the save button click event"""
        # Validate and get the profile data
        profile = self.get_profile()
        if profile is None:
            return  # Validation failed
        
        try:
            # Save the profile using config_edit.add_signalflow_profile utility
            config_edit.apply_signalflow_profile(profile)
            
            # Store the profile and close the dialog with success
            self.profile = profile
            self.accept()  # This will trigger QDialog.accepted
            
        except Exception as e:
            # Show error message if save fails
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to save profile: {str(e)}"
            )
    
    def get_result(self):
        """Return the created/edited profile"""
        return self.profile