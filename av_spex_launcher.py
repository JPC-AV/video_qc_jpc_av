#!/usr/bin/env python3
"""
Main launcher for AV-Spex application.
This script serves as the entry point for both development and the packaged application.
"""
import sys
import os
import platform

# Add the src directory to the Python path if needed
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# More comprehensive macOS specific handling
if platform.system() == 'Darwin':
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    os.environ['PYTHONHOME'] = os.path.dirname(os.path.abspath(__file__))
    
    # Force the application to be recognized as the frontmost app
    try:
        from AppKit import NSApplication
        NSApplication.sharedApplication()
    except ImportError:
        # If AppKit isn't available, try with PyQt directly
        pass

# The import needs to happen after the environment setup
from AV_Spex.av_spex_the_file import main_gui

# For macOS, we need to update the LazyGUILoader to properly handle the menu bar
# This can be done by modifying the get_application method in the LazyGUILoader class
# in av_spex_the_file.py, but a simpler approach is to use a wrapper function here:

def main():
    # Create the application properly for macOS
    if platform.system() == 'Darwin':
        # Try to set up explicit menu bar handling
        try:
            # Import needed PyQt classes
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import QCoreApplication
            
            # Set organization and application name
            QCoreApplication.setOrganizationName("NMAAHC")
            QCoreApplication.setApplicationName("AV-Spex")
            
            # Run the main function which will create and show the window
            main_gui()
        except Exception as e:
            print(f"Error setting up macOS application: {e}")
            # Fall back to standard mode
            main_gui()
    else:
        # For non-macOS, just run normally
        main_gui()

if __name__ == "__main__":
    main()