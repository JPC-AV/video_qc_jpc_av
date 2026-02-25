from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QLineEdit,
    QLabel, QComboBox, QPushButton, QScrollArea
)
from PyQt6.QtCore import Qt

from AV_Spex.gui.gui_theme_manager import ThemeManager, ThemeableMixin
from AV_Spex.utils.config_setup import ChecksConfig
from AV_Spex.utils.config_manager import ConfigManager

config_mgr = ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)

class ComplexWindow(QWidget, ThemeableMixin):
    """Configuration window for complex analysis settings."""
    
    def __init__(self):
        super().__init__()
        self.is_loading = False

        # Initialize themed_group_boxes before setup_theme_handling
        self.themed_group_boxes = {}
        
        # Setup theme handling
        self.setup_theme_handling()
        
        # Setup UI and load config
        self.setup_ui()
        self.load_config_values()

    def setup_ui(self):
        """Create fixed layout structure"""
        main_layout = QVBoxLayout(self)

        self.setup_qctools_section(main_layout)
        self.setup_qct_parse_section(main_layout)
        self.setup_frame_analysis_sections(main_layout)
        self.connect_signals()

    def setup_qctools_section(self, main_layout):
        """Set up the QCTools section"""
        theme_manager = ThemeManager.instance()
        
        # QCTools section
        self.qctools_group = QGroupBox("QCTools")
        theme_manager.style_groupbox(self.qctools_group, "top center")
        self.themed_group_boxes['qctools'] = self.qctools_group

        qctools_layout = QVBoxLayout()

        # Run Tool checkbox
        self.run_qctools_cb = QCheckBox("Run Tool")
        self.run_qctools_cb.setStyleSheet("font-weight: bold;")
        run_qctools_desc = QLabel("Run QCTools on input video file")
        run_qctools_desc.setIndent(20)

        # File Extension dropdown
        qctools_ext_label = QLabel("QCTools File Extension")
        qctools_ext_label.setStyleSheet("font-weight: bold;")
        qctools_ext_desc = QLabel("Set the extension for QCTools output files")
        qctools_ext_desc.setIndent(20)
        self.qctools_ext_combo = QComboBox()
        self.qctools_ext_combo.addItems(["qctools.xml.gz", "qctools.mkv"])
        self.qctools_ext_combo.setMinimumWidth(160)
        
        # Create a horizontal layout for the extension input
        qctools_ext_layout = QHBoxLayout()
        qctools_ext_layout.addWidget(qctools_ext_label)
        qctools_ext_layout.addWidget(self.qctools_ext_combo)
        qctools_ext_layout.addStretch()

        # Add all widgets to the qctools layout
        qctools_layout.addWidget(self.run_qctools_cb)
        qctools_layout.addWidget(run_qctools_desc)
        qctools_layout.addSpacing(10)  # Add some space between sections
        qctools_layout.addLayout(qctools_ext_layout)
        qctools_layout.addWidget(qctools_ext_desc)
        
        self.qctools_group.setLayout(qctools_layout)
        main_layout.addWidget(self.qctools_group)
        
    # QCT Parse Section
    def setup_qct_parse_section(self, main_layout):
        """Set up the qct-parse section with palette-aware styling"""
        theme_manager = ThemeManager.instance()
        
        # QCT Parse section
        self.qct_group = QGroupBox("qct-parse")
        theme_manager.style_groupbox(self.qct_group, "top center")
        self.themed_group_boxes['qct'] = self.qct_group

        qct_layout = QVBoxLayout()

        # Checkboxes with descriptions on second line
        self.run_qctparse_cb = QCheckBox("Run Tool")
        self.run_qctparse_cb.setStyleSheet("font-weight: bold;")
        run_qctparse_desc = QLabel("Run qct-parse tool on input video file")
        run_qctparse_desc.setIndent(20)

        self.bars_detection_cb = QCheckBox("Detect Color Bars")
        self.bars_detection_cb.setStyleSheet("font-weight: bold;")
        bars_detection_desc = QLabel("Detect color bars in the video content")
        bars_detection_desc.setIndent(20)

        self.evaluate_bars_cb = QCheckBox("Evaluate Color Bars")
        self.evaluate_bars_cb.setStyleSheet("font-weight: bold;")
        evaluate_bars_desc = QLabel("Compare content to color bars for validation")
        evaluate_bars_desc.setIndent(20)

        self.thumb_export_cb = QCheckBox("Thumbnail Export")
        self.thumb_export_cb.setStyleSheet("font-weight: bold;")
        thumb_export_desc = QLabel("Export thumbnails of failed frames for review")
        thumb_export_desc.setIndent(20)

        # Add all widgets to the qct layout
        qct_layout.addWidget(self.run_qctparse_cb)
        qct_layout.addWidget(run_qctparse_desc)
        qct_layout.addWidget(self.bars_detection_cb)
        qct_layout.addWidget(bars_detection_desc)
        qct_layout.addWidget(self.evaluate_bars_cb)
        qct_layout.addWidget(evaluate_bars_desc)
        qct_layout.addWidget(self.thumb_export_cb)
        qct_layout.addWidget(thumb_export_desc)
        
        self.qct_group.setLayout(qct_layout)
        main_layout.addWidget(self.qct_group)

    # Frame Analysis Sections (restructured)
    def setup_frame_analysis_sections(self, main_layout):
        """Set up the individual frame analysis sections"""
        self.setup_analysis_periods_section(main_layout)
        self.setup_border_detection_section(main_layout)
        self.setup_brng_analysis_section(main_layout)
        self.setup_signalstats_section(main_layout)

    def setup_analysis_periods_section(self, main_layout):
        """Set up the analysis periods section (shared by signalstats and BRNG analysis)"""
        theme_manager = ThemeManager.instance()

        self.analysis_periods_group = QGroupBox("Frame Analysis Periods")
        theme_manager.style_groupbox(self.analysis_periods_group, "top left")
        self.themed_group_boxes['analysis_periods'] = self.analysis_periods_group

        periods_main_layout = QVBoxLayout()

        periods_overview = QLabel("Controls how many time windows are sampled across the video for signalstats and BRNG analysis")
        periods_overview.setWordWrap(True)
        periods_overview.setIndent(5)
        periods_main_layout.addWidget(periods_overview)
        periods_main_layout.addSpacing(10)

        # Number of periods
        count_layout = QHBoxLayout()
        count_label = QLabel("Number of Periods:")
        count_label.setStyleSheet("font-weight: bold;")
        self.analysis_period_count_input = QLineEdit("3")
        self.analysis_period_count_input.setMaximumWidth(60)
        count_layout.addWidget(count_label)
        count_layout.addWidget(self.analysis_period_count_input)
        count_layout.addStretch()

        count_desc = QLabel("Number of analysis periods to spread across video")
        count_desc.setIndent(20)

        # Period duration
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Period Duration (s):")
        duration_label.setStyleSheet("font-weight: bold;")
        self.analysis_period_duration_input = QLineEdit("60")
        self.analysis_period_duration_input.setMaximumWidth(60)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.analysis_period_duration_input)
        duration_layout.addStretch()

        duration_desc = QLabel("Length of each analysis period in seconds")
        duration_desc.setIndent(20)

        periods_main_layout.addLayout(count_layout)
        periods_main_layout.addWidget(count_desc)
        periods_main_layout.addLayout(duration_layout)
        periods_main_layout.addWidget(duration_desc)

        self.analysis_periods_group.setLayout(periods_main_layout)
        main_layout.addWidget(self.analysis_periods_group)

    def setup_border_detection_section(self, main_layout):
        """Set up the border detection section with enable checkbox"""
        theme_manager = ThemeManager.instance()
        
        # Border Detection Group Box
        self.border_detection_group = QGroupBox("Border Detection Settings")
        theme_manager.style_groupbox(self.border_detection_group, "top left")
        self.themed_group_boxes['border_detection'] = self.border_detection_group
        
        border_detection_layout = QVBoxLayout()
        
        # Enable Border Detection checkbox (moved from substeps)
        self.enable_border_detection_cb = QCheckBox("Enable Border Detection")
        self.enable_border_detection_cb.setStyleSheet("font-weight: bold;")
        border_det_desc = QLabel("Detect and crop blanking borders from the video")
        border_det_desc.setIndent(20)
        
        border_detection_layout.addWidget(self.enable_border_detection_cb)
        border_detection_layout.addWidget(border_det_desc)
        border_detection_layout.addSpacing(10)
        
        # Border Detection Mode selector
        border_mode_widget = QWidget()
        border_mode_layout = QHBoxLayout(border_mode_widget)
        border_mode_layout.setContentsMargins(0, 0, 0, 0)
        
        border_mode_label = QLabel("Detection Mode:")
        border_mode_label.setStyleSheet("font-weight: bold;")
        self.border_mode_combo = QComboBox()
        self.border_mode_combo.addItem("Simple", "simple")
        self.border_mode_combo.addItem("Sophisticated", "sophisticated")
        
        border_mode_layout.addWidget(border_mode_label)
        border_mode_layout.addWidget(self.border_mode_combo)
        border_mode_layout.addStretch()
        
        border_detection_layout.addWidget(border_mode_widget)
        
        # Simple Border Parameters
        self.simple_params_widget = QWidget()
        simple_params_layout = QVBoxLayout(self.simple_params_widget)
        simple_params_layout.setContentsMargins(0, 0, 0, 0)
        
        simple_border_layout = QHBoxLayout()
        simple_border_label = QLabel("Border Pixels:")
        simple_border_label.setStyleSheet("font-weight: bold;")
        self.simple_border_pixels_input = QLineEdit()
        self.simple_border_pixels_input.setMaximumWidth(60)
        self.simple_border_pixels_input.setText("25")
        simple_border_layout.addWidget(simple_border_label)
        simple_border_layout.addWidget(self.simple_border_pixels_input)
        simple_border_layout.addStretch()
        
        simple_params_layout.addLayout(simple_border_layout)
        simple_params_desc = QLabel("Fixed number of pixels to crop from each edge")
        simple_params_desc.setIndent(20)
        simple_params_layout.addWidget(simple_params_desc)
        
        # Sophisticated Border Parameters
        self.sophisticated_params_widget = QWidget()
        sophisticated_params_layout = QVBoxLayout(self.sophisticated_params_widget)
        sophisticated_params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Detection Parameters Group
        detection_params_group = QGroupBox("Detection Parameters")
        theme_manager.style_groupbox(detection_params_group, "top center")
        self.themed_group_boxes['detection_params'] = detection_params_group
        
        detection_params_layout = QVBoxLayout()
        
        # Border Brightness Threshold
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Brightness Threshold:")
        threshold_label.setStyleSheet("font-weight: bold;")
        self.soph_threshold_input = QLineEdit("10")
        self.soph_threshold_input.setMaximumWidth(60)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.soph_threshold_input)
        threshold_layout.addStretch()
        
        threshold_desc = QLabel("0 = pure black, 255 = pure white")
        threshold_desc.setIndent(20)
        
        # Edge Sample Width
        edge_width_layout = QHBoxLayout()
        edge_width_label = QLabel("Edge Sample Width:")
        edge_width_label.setStyleSheet("font-weight: bold;")
        self.soph_edge_width_input = QLineEdit("100")
        self.soph_edge_width_input.setMaximumWidth(60)
        edge_width_layout.addWidget(edge_width_label)
        edge_width_layout.addWidget(self.soph_edge_width_input)
        edge_width_layout.addStretch()
        
        edge_width_desc = QLabel("Pixels to examine from each edge")
        edge_width_desc.setIndent(20)
        
        # Sample Frames
        sample_frames_layout = QHBoxLayout()
        sample_frames_label = QLabel("Sample Frames:")
        sample_frames_label.setStyleSheet("font-weight: bold;")
        self.soph_sample_frames_input = QLineEdit("30")
        self.soph_sample_frames_input.setMaximumWidth(60)
        sample_frames_layout.addWidget(sample_frames_label)
        sample_frames_layout.addWidget(self.soph_sample_frames_input)
        sample_frames_layout.addStretch()
        
        sample_frames_desc = QLabel("Number of frames to sample across the video")
        sample_frames_desc.setIndent(20)
        
        # Padding
        padding_layout = QHBoxLayout()
        padding_label = QLabel("Padding:")
        padding_label.setStyleSheet("font-weight: bold;")
        self.soph_padding_input = QLineEdit("5")
        self.soph_padding_input.setMaximumWidth(60)
        padding_layout.addWidget(padding_label)
        padding_layout.addWidget(self.soph_padding_input)
        padding_layout.addStretch()
        
        padding_desc = QLabel("Extra margin around detected borders")
        padding_desc.setIndent(20)
        
        detection_params_layout.addLayout(threshold_layout)
        detection_params_layout.addWidget(threshold_desc)
        detection_params_layout.addLayout(edge_width_layout)
        detection_params_layout.addWidget(edge_width_desc)
        detection_params_layout.addLayout(sample_frames_layout)
        detection_params_layout.addWidget(sample_frames_desc)
        detection_params_layout.addLayout(padding_layout)
        detection_params_layout.addWidget(padding_desc)
        
        detection_params_group.setLayout(detection_params_layout)
        
        # Auto Retry checkbox
        self.auto_retry_borders_cb = QCheckBox("Auto-retry border detection if BRNG detects edge artifacts")
        self.auto_retry_borders_cb.setStyleSheet("font-weight: bold;")
        auto_retry_desc = QLabel("Automatically adjusts borders if edge artifacts are found")
        auto_retry_desc.setIndent(20)
        
        # Max Border Retries
        max_retries_layout = QHBoxLayout()
        max_retries_label = QLabel("Max Retries:")
        max_retries_label.setStyleSheet("font-weight: bold;")
        self.max_border_retries_input = QLineEdit("5")
        self.max_border_retries_input.setMaximumWidth(60)
        max_retries_layout.addWidget(max_retries_label)
        max_retries_layout.addWidget(self.max_border_retries_input)
        max_retries_layout.addStretch()
        
        max_retries_desc = QLabel("Maximum number of border adjustment attempts")
        max_retries_desc.setIndent(20)
        
        # Add sophisticated mode components
        sophisticated_params_layout.addWidget(detection_params_group)
        sophisticated_params_layout.addWidget(self.auto_retry_borders_cb)
        sophisticated_params_layout.addWidget(auto_retry_desc)
        sophisticated_params_layout.addLayout(max_retries_layout)
        sophisticated_params_layout.addWidget(max_retries_desc)
        
        # Add both parameter widgets to border detection layout
        border_detection_layout.addWidget(self.simple_params_widget)
        border_detection_layout.addWidget(self.sophisticated_params_widget)
        
        # Initially hide sophisticated params
        self.sophisticated_params_widget.setVisible(False)
        
        self.border_detection_group.setLayout(border_detection_layout)
        main_layout.addWidget(self.border_detection_group)
        
        # Connect enable/disable logic
        self.enable_border_detection_cb.stateChanged.connect(self.update_border_detection_visibility)
        self.border_mode_combo.currentIndexChanged.connect(self.update_border_detection_visibility)

    def setup_brng_analysis_section(self, main_layout):
        """Set up the BRNG analysis section with enable checkbox"""
        theme_manager = ThemeManager.instance()
        
        # BRNG Analysis Group Box
        self.brng_group = QGroupBox("BRNG Analysis Settings")
        theme_manager.style_groupbox(self.brng_group, "top left")
        self.themed_group_boxes['brng_analysis'] = self.brng_group
        
        brng_layout = QVBoxLayout()
        
        # Enable BRNG Analysis checkbox (moved from substeps)
        self.enable_brng_analysis_cb = QCheckBox("Enable BRNG Analysis")
        self.enable_brng_analysis_cb.setStyleSheet("font-weight: bold;")
        brng_desc = QLabel("Analyze broadcast range violations in the active area")
        brng_desc.setIndent(20)
        
        brng_layout.addWidget(self.enable_brng_analysis_cb)
        brng_layout.addWidget(brng_desc)
        brng_layout.addSpacing(10)
        
        # Duration Limit
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Duration Limit (s):")
        duration_label.setStyleSheet("font-weight: bold;")
        self.brng_duration_input = QLineEdit("300")
        self.brng_duration_input.setMaximumWidth(60)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.brng_duration_input)
        duration_layout.addStretch()
        
        duration_desc = QLabel("Maximum duration to analyze for BRNG violations")
        duration_desc.setIndent(20)
        
        # Skip Color Bars
        self.brng_skip_colorbars_cb = QCheckBox("Skip Color Bars")
        self.brng_skip_colorbars_cb.setStyleSheet("font-weight: bold;")
        skip_bars_desc = QLabel("Exclude color bar sections from BRNG analysis")
        skip_bars_desc.setIndent(20)
        
        brng_layout.addLayout(duration_layout)
        brng_layout.addWidget(duration_desc)
        brng_layout.addWidget(self.brng_skip_colorbars_cb)
        brng_layout.addWidget(skip_bars_desc)
        
        self.brng_group.setLayout(brng_layout)
        main_layout.addWidget(self.brng_group)
        
        # Connect enable/disable logic
        self.enable_brng_analysis_cb.stateChanged.connect(self.update_brng_analysis_visibility)

    def setup_signalstats_section(self, main_layout):
        """Set up the signalstats section with enable checkbox"""
        theme_manager = ThemeManager.instance()
        
        # Signalstats Group Box
        self.signalstats_group = QGroupBox("Signalstats Settings")
        theme_manager.style_groupbox(self.signalstats_group, "top left")
        self.themed_group_boxes['signalstats'] = self.signalstats_group
        
        signalstats_layout = QVBoxLayout()
        
        # Enable Signalstats checkbox (moved from substeps)
        self.enable_signalstats_cb = QCheckBox("Enable Signalstats Analysis")
        self.enable_signalstats_cb.setStyleSheet("font-weight: bold;")
        signalstats_desc = QLabel("Enhanced FFprobe signalstats")
        signalstats_desc.setIndent(20)
        
        signalstats_layout.addWidget(self.enable_signalstats_cb)
        signalstats_layout.addWidget(signalstats_desc)
        
        self.signalstats_group.setLayout(signalstats_layout)
        main_layout.addWidget(self.signalstats_group)
        
        # Connect enable/disable logic
        self.enable_signalstats_cb.stateChanged.connect(self.update_signalstats_visibility)

    def update_border_detection_visibility(self):
        """Update visibility of border detection components"""
        enabled = self.enable_border_detection_cb.isChecked()
        is_sophisticated = self.border_mode_combo.currentData() == "sophisticated"
        
        # Show/hide mode-specific parameters
        self.simple_params_widget.setVisible(enabled and not is_sophisticated)
        self.sophisticated_params_widget.setVisible(enabled and is_sophisticated)
        
        # Update signalstats availability
        self.update_signalstats_dependency()

    def update_brng_analysis_visibility(self):
        """Update visibility of BRNG analysis components based on checkbox"""
        # BRNG section is always visible, but could be used for future logic
        pass

    def update_signalstats_visibility(self):
        """Update visibility of signalstats components based on dependencies"""
        self.update_signalstats_dependency()

    def update_signalstats_dependency(self):
        """Update signalstats checkbox tooltip based on dependencies"""
        border_enabled = self.enable_border_detection_cb.isChecked()
        
        if not border_enabled:
            self.enable_signalstats_cb.setToolTip("Requires border detection to be enabled")
            self.enable_signalstats_cb.setEnabled(False)
        else:
            # Signalstats now works with both simple and sophisticated modes
            self.enable_signalstats_cb.setToolTip("")
            self.enable_signalstats_cb.setEnabled(True)

    def on_theme_changed(self, palette):
        """Handle theme changes for ComplexWindow"""
        # Apply the palette directly
        self.setPalette(palette)
        
        # Get the theme manager
        theme_manager = ThemeManager.instance()
        
        # Update all tracked group boxes with their specific title positions
        for key, group_box in self.themed_group_boxes.items():
            # Preserve the title position if set
            position = group_box.property("title_position") or "top left"
            theme_manager.style_groupbox(group_box, position)
        
        # Style all buttons
        theme_manager.style_buttons(self)
            
        # Force repaint
        self.update()

    def connect_signals(self):
        """Connect all widget signals to their handlers"""
        # QCTools section
        self.run_qctools_cb.stateChanged.connect(
            lambda state: self.on_boolean_changed(state, ['tools', 'qctools', 'run_tool'])
        )
        self.qctools_ext_combo.currentTextChanged.connect(self.on_qctools_ext_changed)

        # Sub-step enable checkboxes
        self.enable_border_detection_cb.stateChanged.connect(
            lambda state: self.on_boolean_changed(state, ['outputs', 'frame_analysis', 'enable_border_detection'])
        )
        self.enable_brng_analysis_cb.stateChanged.connect(
            lambda state: self.on_boolean_changed(state, ['outputs', 'frame_analysis', 'enable_brng_analysis'])
        )
        self.enable_signalstats_cb.stateChanged.connect(
            lambda state: self.on_boolean_changed(state, ['outputs', 'frame_analysis', 'enable_signalstats'])
        )
        
        # Border mode combo
        self.border_mode_combo.currentIndexChanged.connect(self.on_frame_analysis_mode_changed)

        # Simple parameters
        self.simple_border_pixels_input.textChanged.connect(
            lambda text: self.on_frame_analysis_param_changed('simple_border_pixels', text)
        )

        # Sophisticated parameters
        self.soph_threshold_input.textChanged.connect(
            lambda text: self.on_frame_analysis_param_changed('sophisticated_threshold', text)
        )
        self.soph_edge_width_input.textChanged.connect(
            lambda text: self.on_frame_analysis_param_changed('sophisticated_edge_sample_width', text)
        )
        self.soph_sample_frames_input.textChanged.connect(
            lambda text: self.on_frame_analysis_param_changed('sophisticated_sample_frames', text)
        )
        self.soph_padding_input.textChanged.connect(
            lambda text: self.on_frame_analysis_param_changed('sophisticated_padding', text)
        )
        self.auto_retry_borders_cb.stateChanged.connect(
            lambda state: self.on_boolean_changed(state, ['outputs', 'frame_analysis', 'auto_retry_borders'])
        )

        # BRNG parameters
        self.brng_duration_input.textChanged.connect(
            lambda text: self.on_frame_analysis_param_changed('brng_duration_limit', text)
        )
        self.brng_skip_colorbars_cb.stateChanged.connect(
            lambda state: self.on_boolean_changed(state, ['outputs', 'frame_analysis', 'brng_skip_color_bars'])
        )
        self.max_border_retries_input.textChanged.connect(
            lambda text: self.on_frame_analysis_param_changed('max_border_retries', text)
        )

        # Analysis period parameters
        self.analysis_period_duration_input.textChanged.connect(
            lambda text: self.on_frame_analysis_param_changed('analysis_period_duration', text)
        )
        self.analysis_period_count_input.textChanged.connect(
            lambda text: self.on_frame_analysis_param_changed('analysis_period_count', text)
        )
        
        # QCT Parse - Run Tool turns all dependent checks on/off
        self.run_qctparse_cb.stateChanged.connect(self.on_run_qctparse_changed)
        self.bars_detection_cb.stateChanged.connect(self.on_bars_detection_changed)
        self.evaluate_bars_cb.stateChanged.connect(self.on_evaluate_bars_changed)
        self.thumb_export_cb.stateChanged.connect(
            lambda state: self.on_boolean_changed(state, ['tools', 'qct_parse', 'thumbExport'])
        )

    def load_config_values(self):
        """Load current config values into UI elements"""
        # Set loading flag to True
        self.is_loading = True

        checks_config = config_mgr.get_config('checks', ChecksConfig)

        # QCTools
        qctools = checks_config.tools.qctools
        self.run_qctools_cb.setChecked(bool(qctools.run_tool))
        self.qctools_ext_combo.blockSignals(True)
        qctools_ext = getattr(checks_config.outputs, 'qctools_ext', 'qctools.xml.gz')
        ext_index = self.qctools_ext_combo.findText(qctools_ext)
        if ext_index >= 0:
            self.qctools_ext_combo.setCurrentIndex(ext_index)
        else:
            self.qctools_ext_combo.setCurrentText('qctools.xml.gz')
        self.qctools_ext_combo.blockSignals(False)
        
        # Frame Analysis
        if hasattr(checks_config.outputs, 'frame_analysis'):
            frame_config = checks_config.outputs.frame_analysis
            
            # Load sub-step enable states
            self.enable_border_detection_cb.setChecked(bool(frame_config.enable_border_detection))
            self.enable_brng_analysis_cb.setChecked(bool(frame_config.enable_brng_analysis))
            self.enable_signalstats_cb.setChecked(bool(frame_config.enable_signalstats))
            
            # Set border detection mode
            mode_index = self.border_mode_combo.findData(frame_config.border_detection_mode)
            if mode_index >= 0:
                self.border_mode_combo.setCurrentIndex(mode_index)
            
            # Load parameter values
            self.simple_border_pixels_input.setText(str(frame_config.simple_border_pixels))
            self.soph_threshold_input.setText(str(frame_config.sophisticated_threshold))
            self.soph_edge_width_input.setText(str(frame_config.sophisticated_edge_sample_width))
            self.soph_sample_frames_input.setText(str(frame_config.sophisticated_sample_frames))
            self.soph_padding_input.setText(str(frame_config.sophisticated_padding))
            self.auto_retry_borders_cb.setChecked(bool(frame_config.auto_retry_borders))
            self.brng_duration_input.setText(str(frame_config.brng_duration_limit))
            self.brng_skip_colorbars_cb.setChecked(bool(frame_config.brng_skip_color_bars))
            self.max_border_retries_input.setText(str(getattr(frame_config, 'max_border_retries', 3)))
            self.analysis_period_duration_input.setText(str(frame_config.analysis_period_duration))
            self.analysis_period_count_input.setText(str(getattr(frame_config, 'analysis_period_count', 3)))
            
            # Update visibility based on loaded state
            self.update_border_detection_visibility()
            self.update_brng_analysis_visibility()
            self.update_signalstats_visibility()
        
        # QCT Parse
        qct = checks_config.tools.qct_parse
        self.run_qctparse_cb.setChecked(bool(qct.run_tool))
        self.bars_detection_cb.setChecked(qct.barsDetection)
        self.evaluate_bars_cb.setChecked(qct.evaluateBars)
        self.thumb_export_cb.setChecked(qct.thumbExport)

        # Set initial enabled state for QCT Parse dependent checkboxes
        dependent_checkboxes = [
            self.bars_detection_cb,
            self.evaluate_bars_cb, 
            self.thumb_export_cb
        ]

        # Enable/disable dependent checkboxes based on run_tool state
        for checkbox in dependent_checkboxes:
            checkbox.setEnabled(qct.run_tool)

        # Additional logic for thumbnail dependency
        if qct.run_tool:
            # Thumbnail is only enabled if at least one of bars detection or evaluate bars is checked
            thumbnail_should_be_enabled = qct.barsDetection or qct.evaluateBars
            self.thumb_export_cb.setEnabled(thumbnail_should_be_enabled)

        # Set loading flag back to False after everything is loaded
        self.is_loading = False

    def on_checkbox_changed(self, state, path):
        """Handle changes in yes/no checkboxes"""
        # Skip updates while loading
        if self.is_loading:
            return

        new_value = 'yes' if Qt.CheckState(state) == Qt.CheckState.Checked else 'no'
        
        if path[0] == "tools" and len(path) > 2:
            tool_name = path[1]
            field = path[2]
            updates = {'tools': {tool_name: {field: new_value}}}
        elif len(path) == 3:  # Handle nested structures like outputs.frame_analysis.field
            section = path[0]
            subsection = path[1]
            field = path[2]
            updates = {section: {subsection: {field: new_value}}}
        else:
            section = path[0]
            field = path[1]
            updates = {section: {field: new_value}}
            
        config_mgr.update_config('checks', updates)

    def on_boolean_changed(self, state, path):
        """Handle changes in boolean checkboxes"""
        # Skip updates while loading
        if self.is_loading:
            return
        
        new_value = Qt.CheckState(state) == Qt.CheckState.Checked
        
        if path[0] == "tools" and path[1] == "qct_parse":
            updates = {'tools': {'qct_parse': {path[2]: new_value}}}
            config_mgr.update_config('checks', updates)
        elif path[0] == "tools" and path[1] == "qctools":
            updates = {'tools': {'qctools': {path[2]: new_value}}}
            config_mgr.update_config('checks', updates)
        elif path[0] == "outputs" and path[1] == "frame_analysis":
            updates = {'outputs': {'frame_analysis': {path[2]: new_value}}}
            config_mgr.update_config('checks', updates)

    def on_frame_analysis_mode_changed(self, index):
        """Handle border detection mode changes"""
        if self.is_loading:
            return
        
        mode = self.border_mode_combo.itemData(index)
        
        # Update visibility
        self.update_border_detection_visibility()
        
        # Update config
        updates = {'outputs': {'frame_analysis': {'border_detection_mode': mode}}}
        config_mgr.update_config('checks', updates)

    def on_frame_analysis_param_changed(self, param_name, value):
        """Handle frame analysis parameter changes"""
        if self.is_loading:
            return
        
        # Convert to appropriate type
        if param_name in ['simple_border_pixels', 'sophisticated_threshold', 'sophisticated_edge_sample_width',
                        'sophisticated_sample_frames', 'sophisticated_padding', 'sophisticated_viz_time',
                        'sophisticated_search_window', 'brng_duration_limit',
                        'analysis_period_duration', 'analysis_period_count', 'max_border_retries']:
            try:
                # Handle empty string case
                value = int(value) if value.strip() else 0
            except ValueError:
                return  # Don't update config if conversion fails
        
        updates = {'outputs': {'frame_analysis': {param_name: value}}}
        config_mgr.update_config('checks', updates)

    def on_qct_combo_changed(self, value, field):
        """Handle changes in QCT Parse combo boxes"""
        # Skip updates while loading
        if self.is_loading:
            return

        values = [value] if value is not None else []
        updates = {'tools': {'qct_parse': {field: values}}}
        config_mgr.update_config('checks', updates)

    def on_qctools_ext_changed(self, ext):
        """Handle changes in QCTools file extension dropdown"""
        if self.is_loading:
            return
        
        updates = {'outputs': {'qctools_ext': ext}}
        config_mgr.update_config('checks', updates)

    def on_run_qctparse_changed(self, state):
        """Handle changes in run qct-parse checkbox with dependency logic"""
        # Skip updates while loading to avoid issues during initialization
        if self.is_loading:
            return
        
        # Handle the normal config update for run_tool (using boolean)
        new_value = Qt.CheckState(state) == Qt.CheckState.Checked
        updates = {'tools': {'qct_parse': {'run_tool': new_value}}}
        config_mgr.update_config('checks', updates)
        
        # Get list of dependent checkboxes
        dependent_checkboxes = [
            self.bars_detection_cb,
            self.evaluate_bars_cb, 
            self.thumb_export_cb
        ]
        
        if Qt.CheckState(state) == Qt.CheckState.Checked:
            # If run_tool is checked, enable and check all dependent checkboxes
            self.bars_detection_cb.setEnabled(True)
            self.bars_detection_cb.blockSignals(True)
            self.bars_detection_cb.setChecked(True)
            self.bars_detection_cb.blockSignals(False)
            
            self.evaluate_bars_cb.setEnabled(True)
            self.evaluate_bars_cb.blockSignals(True)
            self.evaluate_bars_cb.setChecked(True)
            self.evaluate_bars_cb.blockSignals(False)
            
            self.thumb_export_cb.setEnabled(True)
            self.thumb_export_cb.blockSignals(True)
            self.thumb_export_cb.setChecked(True)
            self.thumb_export_cb.blockSignals(False)
            
            # Update the config for all dependent options
            dependent_updates = {
                'tools': {
                    'qct_parse': {
                        'barsDetection': True,
                        'evaluateBars': True,
                        'thumbExport': True
                    }
                }
            }
            config_mgr.update_config('checks', dependent_updates)
        else:
            # If run_tool is unchecked, disable (grey out) all dependent checkboxes
            for checkbox in dependent_checkboxes:
                checkbox.setEnabled(False)

    def on_bars_detection_changed(self, state):
        """Handle changes in bars detection checkbox with dependency logic"""
        # Skip updates while loading
        if self.is_loading:
            return
        
        # Update the config normally
        new_value = Qt.CheckState(state) == Qt.CheckState.Checked
        updates = {'tools': {'qct_parse': {'barsDetection': new_value}}}
        config_mgr.update_config('checks', updates)
        
        if Qt.CheckState(state) == Qt.CheckState.Checked:
            # If bars detection is checked, enable evaluate bars and thumbnail export
            self.evaluate_bars_cb.setEnabled(True)
            self.thumb_export_cb.setEnabled(True)
        else:
            # If bars detection is unchecked, disable and uncheck evaluate bars and thumbnail export
            self.evaluate_bars_cb.blockSignals(True)
            self.evaluate_bars_cb.setChecked(False)
            self.evaluate_bars_cb.setEnabled(False)
            self.evaluate_bars_cb.blockSignals(False)
            
            self.thumb_export_cb.blockSignals(True)
            self.thumb_export_cb.setChecked(False)
            self.thumb_export_cb.setEnabled(False)
            self.thumb_export_cb.blockSignals(False)
            
            # Update config for the dependent options that got unchecked
            dependent_updates = {
                'tools': {
                    'qct_parse': {
                        'evaluateBars': False,
                        'thumbExport': False
                    }
                }
            }
            config_mgr.update_config('checks', dependent_updates)
        
        # Check overall dependencies
        self.check_qct_dependencies()

    def on_evaluate_bars_changed(self, state):
        """Handle changes in evaluate bars checkbox with dependency logic"""
        # Skip updates while loading
        if self.is_loading:
            return
        
        # Update the config normally
        new_value = Qt.CheckState(state) == Qt.CheckState.Checked
        updates = {'tools': {'qct_parse': {'evaluateBars': new_value}}}
        config_mgr.update_config('checks', updates)
        
        # Check overall dependencies
        self.check_qct_dependencies()

    def check_qct_dependencies(self):
        """Check and enforce QCT Parse dependencies"""
        # If bars detection is off, then evaluate bars and thumbnail should already be disabled
        # This check is for the case where both bars detection and evaluate bars are unchecked
        if not self.bars_detection_cb.isChecked() and not self.evaluate_bars_cb.isChecked():
            # Uncheck run tool since no detection methods are active
            self.run_qctparse_cb.blockSignals(True)
            self.run_qctparse_cb.setChecked(False)
            self.run_qctparse_cb.blockSignals(False)
            
            # Disable all dependent options since run tool is now off
            self.bars_detection_cb.setEnabled(False)
            self.evaluate_bars_cb.setEnabled(False)
            self.thumb_export_cb.setEnabled(False)
            
            # Update config for run_tool change
            run_tool_updates = {
                'tools': {
                    'qct_parse': {
                        'run_tool': False
                    }
                }
            }
            config_mgr.update_config('checks', run_tool_updates)