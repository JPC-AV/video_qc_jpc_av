from PyQt6.QtWidgets import QApplication, QGroupBox, QPushButton, QComboBox, QTextEdit
from PyQt6.QtGui import QPalette, QFont, QPixmap
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QSize

import os
import sys
import platform

from AV_Spex.utils.log_setup import logger

class ThemeManager(QObject):
    """
    Singleton manager for handling theme changes in a PyQt6 application.
    Provides centralized styling for all UI components.
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
    
    # === GROUPBOX STYLING ===
    
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

    # === BUTTON STYLING ===
    
    def style_button(self, button, special_style=None):
        """
        Apply styling to a single button.
        
        Args:
            button: The QPushButton to style
            special_style: Optional special style identifier for custom buttons
        """
        if not isinstance(button, QPushButton) or not self.app:
            return
            
        # Get palette colors
        palette = self.app.palette()
        highlight_color = palette.color(palette.ColorRole.Highlight).name()
        highlight_text_color = palette.color(palette.ColorRole.HighlightedText).name()
        button_color = palette.color(palette.ColorRole.Button).name()
        button_text_color = palette.color(palette.ColorRole.ButtonText).name()
        
        # Apply special styling if requested
        if special_style == "check_spex":
            button.setStyleSheet("""
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
        elif special_style == "processing_window":
            button.setStyleSheet("""
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
        elif special_style == "cancel_processing":
            button.setStyleSheet("""
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
        else:
            # Apply standard button styling
            button.setStyleSheet(f"""
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
            """)
    
    def style_buttons(self, parent_widget):
        """Apply consistent styling to all buttons under a parent widget."""
        if not self.app:
            return
        
        # Apply to all buttons in the widget
        buttons = parent_widget.findChildren(QPushButton)
        for button in buttons:
            self.style_button(button)
    
    # === COMBOBOX STYLING ===
    
    def style_combobox(self, combo_box):
        """
        Apply consistent styling to a combo box based on current theme.
        
        Args:
            combo_box: The QComboBox to style
        """
        if not isinstance(combo_box, QComboBox) or not self.app:
            return
        
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
        
        # Apply to all comboboxes in the widget
        comboboxes = parent_widget.findChildren(QComboBox)
        for combobox in comboboxes:
            self.style_combobox(combobox)
    
    # === TAB STYLING ===
    
    def style_tabs(self, tab_widget):
        """
        Apply styling to a tab widget.
        
        Args:
            tab_widget: The QTabWidget to style
        """
        if not self.app or not tab_widget:
            return
            
        tab_widget.setStyleSheet(self.get_tab_style())
    
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
    
    # === TEXT STYLING ===
    
    def style_console_text(self, text_edit):
        """
        Apply consistent styling to a ConsoleTextEdit based on current theme.
        
        Args:
            text_edit: The ConsoleTextEdit or QTextEdit to style
        """
        if not isinstance(text_edit, QTextEdit) or not self.app:
            return
            
        # Get the current palette
        palette = self.app.palette()
        base_color = palette.color(QPalette.ColorRole.Base).name()
        text_color = palette.color(QPalette.ColorRole.Text).name()
        
        # Define darker/lighter background based on theme
        theme = self.detect_system_theme()
        is_dark = theme == 'Dark'

        # Define background based on theme
        if is_dark:
            # For dark themes, use slightly lighter than base
            try:
                bg_color = f"rgba({min(palette.color(QPalette.ColorRole.Base).red() + 15, 255)}, "\
                        f"{min(palette.color(QPalette.ColorRole.Base).green() + 15, 255)}, "\
                        f"{min(palette.color(QPalette.ColorRole.Base).blue() + 15, 255)}, 255)"
            except:
                bg_color = "#2a2a2a"  # Fallback dark color
            # Border for dark theme
            try:
                border_color = palette.color(QPalette.ColorRole.Mid).name()
            except:
                border_color = "#555555"  # Fallback border
        else:
            # For light themes, use slightly darker than base
            try:
                bg_color = f"rgba({max(palette.color(QPalette.ColorRole.Base).red() - 15, 0)}, "\
                        f"{max(palette.color(QPalette.ColorRole.Base).green() - 15, 0)}, "\
                        f"{max(palette.color(QPalette.ColorRole.Base).blue() - 15, 0)}, 255)"
            except:
                bg_color = "#f0f0f0"  # Fallback light color
            # Border for light theme
            try:
                border_color = palette.color(QPalette.ColorRole.Mid).name()
            except:
                border_color = "#cccccc"  # Fallback border
        
        # Get highlight colors with fallbacks
        try:
            highlight_color = palette.color(palette.ColorRole.Highlight).name()
            highlight_text_color = palette.color(palette.ColorRole.HighlightedText).name()
            mid_color = palette.color(palette.ColorRole.Mid).name()
        except:
            if is_dark:
                highlight_color = "#3b82f6"
                highlight_text_color = "#ffffff"
                mid_color = "#666666"
            else:
                highlight_color = "#0066cc"
                highlight_text_color = "#ffffff"
                mid_color = "#999999"
            
        # Create console-like style with fallback-safe colors
        text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 5px;
                padding: 5px;
                selection-background-color: {highlight_color};
                selection-color: {highlight_text_color};
            }}
            QScrollBar:vertical {{
                background: {bg_color};
                width: 14px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {mid_color};
                min-height: 20px;
                border-radius: 7px;
            }}
            QScrollBar:horizontal {{
                background: {bg_color};
                height: 14px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {mid_color};
                min-width: 20px;
                border-radius: 7px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                height: 0px;
                width: 0px;
            }}
        """)
        
        # Clear format cache if the method exists
        if hasattr(text_edit, 'clear_formats'):
            text_edit.clear_formats()
    
    # === PROGRESS BAR STYLING ===
    
    def style_progress_bar(self, progress_bar):
        """
        Apply styling to a progress bar.
        
        Args:
            progress_bar: The QProgressBar to style
        """
        if not self.app or not progress_bar:
            return
            
        progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: palette(Base);
                text-align: center;
                padding: 1px;
            }
            QProgressBar::chunk {
                background-color: palette(Highlight);
            }
        """)
    
    # === LOGO & BRANDING ===
    
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
            
        # Use robust theme detection
        theme = self.detect_system_theme()
        is_dark = theme == 'Dark'
        
        # Return appropriate logo path
        return dark_logo_path if is_dark else light_logo_path
    
    def load_logo(self, label, logo_path, width=None, height=None):
        """
        Load a logo image into a QLabel.
        
        Args:
            label: QLabel to place the logo into
            logo_path: Path to the logo image file
            width: Optional width to scale to (maintains aspect ratio if height is None)
            height: Optional height to scale to (maintains aspect ratio if width is None)
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(logo_path):
            return False
            
        pixmap = QPixmap(logo_path)
        if pixmap.isNull():
            return False
            
        # Scale the pixmap if requested
        if width is not None and height is not None:
            scaled_pixmap = pixmap.scaled(QSize(width, height), 
                                         Qt.AspectRatioMode.KeepAspectRatio, 
                                         Qt.TransformationMode.SmoothTransformation)
        elif width is not None:
            scaled_pixmap = pixmap.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
        elif height is not None:
            scaled_pixmap = pixmap.scaledToHeight(height, Qt.TransformationMode.SmoothTransformation)
        else:
            scaled_pixmap = pixmap
            
        label.setPixmap(scaled_pixmap)
        return True
    
    # === UTILITY METHODS ===
    
    def get_current_palette(self):
        """Return the current application palette"""
        return self.app.palette() if self.app else None
        
    def apply_theme_to_all(self, widget):
        """
        Apply theming to all appropriate widgets under a parent widget.
        This is a utility method that combines multiple styling methods.
        
        Args:
            widget: Parent widget containing elements to style
        """
        if not widget:
            return
            
        # Style groups
        for group in widget.findChildren(QGroupBox):
            self.style_groupbox(group)
        
        # Style buttons
        self.style_buttons(widget)
        
        # Style comboboxes
        self.style_comboboxes(widget)
        
        # Find and style text edits
        for text_edit in widget.findChildren(QTextEdit):
            self.style_console_text(text_edit)

    def detect_system_theme(self):
        """
        Robust theme detection that works in packaged apps
        """
        # Method 1: PyQt6 palette detection (most reliable for packaged apps)
        try:
            if self.app:
                palette = self.app.palette()
                window_color = palette.color(palette.ColorRole.Window)
                
                # Get RGB values
                r, g, b = window_color.red(), window_color.green(), window_color.blue()
                
                # Calculate luminance using the relative luminance formula
                # This is more accurate than just using lightness()
                luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                
                # Debug the actual color values
                logger.debug(f"Window color RGB: ({r}, {g}, {b}), Luminance: {luminance:.3f}")
                
                # Threshold for dark mode detection
                # Lower threshold = more likely to detect as dark
                is_dark = luminance < 0.5  # Adjust this threshold if needed
                
                theme = 'Dark' if is_dark else 'Light'
                logger.info(f"PyQt6 palette detection: {theme} (luminance: {luminance:.3f})")
                return theme
                
        except Exception as e:
            logger.warning(f"PyQt6 theme detection failed: {e}")
        
        # Method 2: Check if we're on macOS and try Cocoa directly
        try:
            if platform.system() == 'Darwin':
                # Try using PyObjC to access macOS appearance directly
                try:
                    from Foundation import NSUserDefaults
                    
                    defaults = NSUserDefaults.standardUserDefaults()
                    appearance_name = defaults.stringForKey_('AppleInterfaceStyle')
                    
                    if appearance_name:
                        theme = 'Dark' if 'Dark' in appearance_name else 'Light'
                        logger.info(f"macOS NSUserDefaults detection: {theme}")
                        return theme
                    else:
                        # No dark mode key means light mode
                        logger.info("macOS NSUserDefaults detection: Light (no dark mode key)")
                        return 'Light'
                        
                except ImportError:
                    logger.debug("PyObjC not available, skipping NSUserDefaults method")
                    
        except Exception as e:
            logger.debug(f"macOS detection failed: {e}")
        
        # Method 3: Check for packaged app environment
        try:
            # Check if we're in a PyInstaller packaged app
            if hasattr(sys, '_MEIPASS'):
                # In packaged app, check for environment variable or config
                packaged_theme = os.environ.get('AVSPEX_THEME', None)
                if packaged_theme in ['Dark', 'Light']:
                    logger.info(f"Packaged app theme override: {packaged_theme}")
                    return packaged_theme
                    
        except Exception as e:
            logger.debug(f"Packaged app check failed: {e}")
        
        # Method 4: Environment-based detection for CI/CD
        try:
            # Check for CI environment variables that might indicate theme
            if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
                # In CI, we might want to default to a specific theme
                # or check for a theme preference environment variable
                ci_theme = os.environ.get('PREFERRED_THEME', 'Light')
                logger.info(f"CI environment detected, using theme: {ci_theme}")
                return ci_theme
                
        except Exception as e:
            logger.debug(f"Environment check failed: {e}")
        
        # Method 5: Subprocess as last resort (won't work in some packaged apps)
        if platform.system() == 'Darwin':
            try:
                import subprocess
                result = subprocess.run(
                    ['defaults', 'read', '-g', 'AppleInterfaceStyle'],
                    capture_output=True,
                    text=True,
                    timeout=1  # Shorter timeout for packaged apps
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    theme = 'Dark'
                    logger.info(f"Subprocess detection: {theme}")
                    return theme
                else:
                    theme = 'Light'
                    logger.info(f"Subprocess detection: {theme} (no dark mode)")
                    return theme
                    
            except Exception as e:
                logger.debug(f"Subprocess failed: {e}")
        
        # Method 6: Time-based heuristic as creative fallback
        try:
            from datetime import datetime
            current_hour = datetime.now().hour
            
            # Assume dark mode during evening/night hours
            if 18 <= current_hour or current_hour < 6:
                theme = 'Dark'
                logger.info(f"Time-based fallback: {theme} (hour: {current_hour})")
            else:
                theme = 'Light'
                logger.info(f"Time-based fallback: {theme} (hour: {current_hour})")
                
            return theme
            
        except Exception as e:
            logger.debug(f"Time-based detection failed: {e}")
        
        # Ultimate fallback
        default_theme = 'Light'
        logger.warning(f"All theme detection methods failed, using default: {default_theme}")
        return default_theme
    
    def get_theme_with_fallback(self):
        """
        Get current theme with fallback detection methods.
        Returns 'Dark' or 'Light'
        
        This is an alias for detect_system_theme() for backward compatibility.
        """
        return self.detect_system_theme()



class ThemeableMixin:
    """
    Mixin class for objects that need theme support.
    Provides automatic connection to theme change signals.
    """
    
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
        """
        Override this method to handle theme changes.
        
        This default implementation will:
        1. Apply palette to self if it has setPalette method
        2. Propagate theme change to child components with their own handlers
        3. Apply standard theming to known components
        """
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
                pass
        
        # Apply any common styling for known component types
        # Call theme-specific styling implementations if defined
        theme_specific_methods = [
            '_style_groupboxes', 
            '_style_buttons',
            '_style_comboboxes',
            '_style_tabs', 
            '_style_text_edits',
            '_style_progress_bars',
            '_refresh_logo'
        ]
        
        for method_name in theme_specific_methods:
            method = getattr(self, method_name, None)
            if callable(method):
                method()