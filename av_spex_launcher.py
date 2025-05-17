#!/usr/bin/env python3
"""
Main launcher for AV-Spex application.
This script serves as the entry point for both development and the packaged application.
"""
import sys
import os

# Add the src directory to the Python path if needed
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import and run the main function
from AV_Spex.av_spex_the_file import main_gui

if __name__ == "__main__":
    main_gui()