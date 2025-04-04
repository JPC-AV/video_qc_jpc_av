from PyQt6.QtWidgets import QApplication, QGroupBox, QPushButton
from PyQt6.QtGui import QPalette, QFont
from PyQt6.QtCore import QObject, pyqtSignal, Qt

class ThemeManager(QObject):
    """
    Singleton manager for handling theme changes in a PyQt6 application.
    """
    
    # Signal emitted when theme changes
    themeChanged = pyqtSignal(QPalette)
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def instance(cls):
        """Get the ThemeManager singleton instance"""
        if cls._instance is None:
            cls._instance = ThemeManager()
        return cls._instance
    
    def __new__(cls, *args, **kwargs):
        """Create singleton instance"""
        if cls._instance is None:
            cls._instance = super(ThemeManager, cls).__new__(cls)
            # Initialize the QObject part here
            QObject.__init__(cls._instance)
            # Set initialized flag
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the ThemeManager."""
        # Only initialize once for the singleton
        if not self._initialized:
            # Note: We don't call super().__init__() here because it's
            # already called in __new__
            self._initialized = True
            self.app = QApplication.instance()
            
            # Connect to application's palette change signal if app exists
            if self.app:
                self.app.paletteChanged.connect(self._on_palette_changed)
    
    def _on_palette_changed(self, palette):
        """Handle system palette changes and propagate to connected widgets."""
        self.themeChanged.emit(palette)
    
    def style_groupbox(self, group_box, title_position=None):
        """
        Apply consistent styling to a group box based on current theme.
        
        Args:
            group_box: The QGroupBox to style
            title_position: Position of the title ("top left", "top center", etc.)
                            If None, maintains the group box's current title position
        """
        if not isinstance(group_box, QGroupBox) or not self.app:
            return
            
        # Get the current palette
        palette = self.app.palette()
        midlight_color = palette.color(palette.ColorRole.Midlight).name()
        text_color = palette.color(palette.ColorRole.Text).name()
        
         # If title_position is None, use a simpler approach
        if title_position is None:
            # Store the group title position in the widget property if not already set
            title_pos = group_box.property("title_position")
            if title_pos:
                title_position = title_pos
            else:
                title_position = "top left"  # Default
        else:
            # Store position for future reference
            group_box.setProperty("title_position", title_position)
        
        # Apply style based on current palette with specified or preserved title position
        group_box.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 14px;
                color: {text_color};
                border: 2px solid gray;
                border-radius: 5px;
                margin-top: 10px;
                background-color: {midlight_color};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: {title_position};
                padding: 0 10px;
                color: {text_color};
            }}
        """)

    def style_buttons(self, parent_widget):
        """Apply consistent styling to all buttons under a parent widget."""
        if not self.app:
            return
        
        palette = self.app.palette()
        highlight_color = palette.color(palette.ColorRole.Highlight).name()
        highlight_text_color = palette.color(palette.ColorRole.HighlightedText).name()
        button_color = palette.color(palette.ColorRole.Button).name()
        button_text_color = palette.color(palette.ColorRole.ButtonText).name()
        
        button_style = f"""
            QPushButton {{
                font-weight: bold;
                padding: 8px;
                border: 1px solid gray;
                border-radius: 4px;
                background-color: {button_color};
                color: {button_text_color};
            }}
            QPushButton:hover {{
                background-color: {highlight_color};
                color: {highlight_text_color};
            }}
        """
        
        # Apply to all buttons in the widget
        buttons = parent_widget.findChildren(QPushButton)
        for button in buttons:
            button.setStyleSheet(button_style)

    def style_combobox(self, combo_box):
        """
        Apply consistent styling to a combo box based on current theme.
        
        Args:
            combo_box: The QComboBox to style
        """
        
        # Get the current palette
        palette = self.app.palette()
        
        # Get colors from palette using ColorRoles
        highlight_color = palette.color(palette.ColorRole.Highlight).name()
        highlight_text_color = palette.color(palette.ColorRole.HighlightedText).name()
        dropdown_bg_color = palette.color(palette.ColorRole.AlternateBase).name()
        button_color = palette.color(palette.ColorRole.Button).name()
        button_text_color = palette.color(palette.ColorRole.ButtonText).name()
        
        # Border color from shadow or mid
        border_color = palette.color(palette.ColorRole.Mid).name()
        
        # Apply style based on current palette
        combo_box.setStyleSheet(f"""
            QComboBox {{
                font-weight: bold;
                padding: 8px;
                border: 1px solid gray;
                border-radius: 4px;
                background-color: {button_color};
                color: {button_text_color};
            }}
            QComboBox:hover {{
                background-color: {highlight_color};
                color: {highlight_text_color};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: right;
                width: 20px;
                border-left: 1px solid {border_color};
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }}
            QComboBox QAbstractItemView, QComboBox QListView {{
                background-color: {dropdown_bg_color};
            }}
            QComboBox::item:selected {{
                background-color: {highlight_color};
                color: #4CAF50;
            }}
        """)
        

    def style_comboboxes(self, parent_widget):
        """Apply consistent styling to all comboboxes under a parent widget."""
        if not self.app:
            return
        
        # Get QComboBox class
        from PyQt6.QtWidgets import QComboBox
        
        # Apply to all comboboxes in the widget
        comboboxes = parent_widget.findChildren(QComboBox)
        for combobox in comboboxes:
            self.style_combobox(combobox)

    def get_tab_style(self):
        """
        Generate style for tab widgets based on current palette.
        
        Returns:
            str: CSS stylesheet for QTabWidget and QTabBar
        """
        if not self.app:
            return ""
            
        palette = self.app.palette()
        highlight_color = palette.color(palette.ColorRole.Highlight).name()
        highlight_text_color = palette.color(palette.ColorRole.HighlightedText).name()
        dark_color = palette.color(palette.ColorRole.Mid).name()
        
        return f"""
            QTabBar::tab {{
                padding: 8px 12px;
                margin-right: 2px;
                font-size: 14px;
                font-weight: bold;
                background-color: {dark_color};
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            
            QTabBar::tab:selected, QTabBar::tab:hover {{
                background-color: {highlight_color};
                color: {highlight_text_color};
            }}
            QTabBar::tab:selected {{
                border-bottom: 2px solid #0066cc;
            }}
                                
            QTabWidget::pane {{
                border: 1px solid lightgray;
                background-color: none;
            }}
        """
    
    # Add method to style console text edit
    def style_console_text(self, text_edit):
        """
        Apply consistent styling to a ConsoleTextEdit based on current theme.
        
        Args:
            text_edit: The ConsoleTextEdit to style
        """
            
        # Get the current palette
        palette = self.app.palette()
        base_color = palette.color(QPalette.ColorRole.Base).name()
        text_color = palette.color(QPalette.ColorRole.Text).name()
        
        # Define darker/lighter background based on theme
        is_dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
        if is_dark:
            # For dark themes, use slightly lighter than base
            bg_color = f"rgba({min(palette.color(QPalette.ColorRole.Base).red() + 15, 255)}, "\
                      f"{min(palette.color(QPalette.ColorRole.Base).green() + 15, 255)}, "\
                      f"{min(palette.color(QPalette.ColorRole.Base).blue() + 15, 255)}, 255)"
            # Border for dark theme
            border_color = palette.color(QPalette.ColorRole.Mid).name()
        else:
            # For light themes, use slightly darker than base
            bg_color = f"rgba({max(palette.color(QPalette.ColorRole.Base).red() - 15, 0)}, "\
                      f"{max(palette.color(QPalette.ColorRole.Base).green() - 15, 0)}, "\
                      f"{max(palette.color(QPalette.ColorRole.Base).blue() - 15, 0)}, 255)"
            # Border for light theme
            border_color = palette.color(QPalette.ColorRole.Mid).name()
            
        # Create console-like style
        text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 5px;
                padding: 5px;
                selection-background-color: {palette.color(palette.ColorRole.Highlight).name()};
                selection-color: {palette.color(palette.ColorRole.HighlightedText).name()};
            }}
            QScrollBar:vertical {{
                background: {bg_color};
                width: 14px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {palette.color(palette.ColorRole.Mid).name()};
                min-height: 20px;
                border-radius: 7px;
            }}
            QScrollBar:horizontal {{
                background: {bg_color};
                height: 14px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {palette.color(palette.ColorRole.Mid).name()};
                min-width: 20px;
                border-radius: 7px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                height: 0px;
                width: 0px;
            }}
        """)
        
        # Clear format cache so they'll be recreated with new theme colors
        if hasattr(text_edit, 'clear_formats'):
            text_edit.clear_formats()

    def get_theme_appropriate_logo(self, light_logo_path, dark_logo_path):
        """
        Returns the appropriate logo path based on the current theme.
        
        Args:
            light_logo_path: Path to the logo for light theme
            dark_logo_path: Path to the logo for dark theme
            
        Returns:
            str: Path to the appropriate logo for current theme
        """
        if not self.app:
            return light_logo_path  # Default to light theme if no app
            
        # Determine if we're in dark mode
        palette = self.app.palette()
        is_dark = palette.color(palette.ColorRole.Window).lightness() < 128
        
        # Return appropriate logo path
        return dark_logo_path if is_dark else light_logo_path
    
    def get_current_palette(self):
        """Return the current application palette"""
        return QApplication.palette()
    

class ThemeableMixin:
    """Mixin class for objects that need theme support"""
    
    def setup_theme_handling(self):
        """Connect to theme change notifications"""
        theme_manager = ThemeManager.instance()
        
        # Connect to the theme changed signal
        if hasattr(theme_manager, 'themeChanged'):
            theme_manager.themeChanged.connect(self.on_theme_changed)
        
        # Apply current theme immediately using QApplication's palette
        self.on_theme_changed(QApplication.palette())
        
    def cleanup_theme_handling(self):
        """Disconnect from theme change notifications"""
        theme_manager = ThemeManager.instance()
        try:
            # Disconnect from whichever signal exists
            if hasattr(theme_manager, 'themeChanged'):
                theme_manager.themeChanged.disconnect(self.on_theme_changed)
        except TypeError:
            # Already disconnected or never connected
            pass
            
    def on_theme_changed(self, palette):
        """Override this method to handle theme changes"""
        # Apply the palette to this widget
        if hasattr(self, 'setPalette'):
            self.setPalette(palette)
        
        # Propagate theme change to child components that have their own handlers
        for attr_name in dir(self):
            # Skip special methods and avoid potential recursive calls
            if attr_name.startswith('__') or attr_name == 'on_theme_changed':
                continue
                
            try:
                attr = getattr(self, attr_name)
                # Check if this attribute has an on_theme_changed method
                if hasattr(attr, 'on_theme_changed') and callable(attr.on_theme_changed):
                    # Call the method with the palette argument
                    attr.on_theme_changed(palette)
            except (AttributeError, TypeError) as e:
                # Safely handle any errors in propagation
                print(f"Error propagating theme to {attr_name}: {e}")