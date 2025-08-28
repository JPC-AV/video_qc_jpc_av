#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
import colorlog
from datetime import datetime
from colorlog import ColoredFormatter
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from AV_Spex.gui.gui_processing_window_console import MessageType

class QtLogHandler(logging.Handler, QObject):
    """
    A custom logging handler that emits Qt signals for log messages.
    Bridges Python's logging system with Qt's signal/slot mechanism.
    """
    
    # Signal to emit log messages: (message, level)
    log_message = pyqtSignal(str, object)
    
    def __init__(self, level=logging.NOTSET):
        # Initialize both parent classes
        logging.Handler.__init__(self, level)
        QObject.__init__(self)
        
        # Create a formatter - use the same format as the console
        self.setFormatter(logging.Formatter('%(message)s'))
        
    def emit(self, record):
        """Process a log record and emit it as a Qt signal"""
        # Format the record according to our formatter
        msg = self.format(record)
        
        # Map logging levels to MessageType
        msg_type = MessageType.NORMAL
        if record.levelno >= logging.CRITICAL:
            msg_type = MessageType.ERROR  # Critical as error but could be a distinct type
        elif record.levelno >= logging.ERROR:
            msg_type = MessageType.ERROR
        elif record.levelno >= logging.WARNING:
            msg_type = MessageType.WARNING
        elif record.levelno >= logging.INFO:
            msg_type = MessageType.INFO
        elif record.levelno == logging.DEBUG:
            msg_type = MessageType.COMMAND  # Use command style for debug messages
            
        # Emit the signal with message and type
        self.log_message.emit(msg, msg_type)


def get_log_directory():
    """Determine the appropriate log directory based on how the app is running"""
    if getattr(sys, 'frozen', False):
        # If running as packaged app
        log_dir = os.path.join(str(Path.home()), 'Library', 'Logs', 'AVSpex')
    else:
        # If running from source
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        log_dir = os.path.join(root_dir, 'logs')
    
    # Add date subdirectory
    log_dir = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d'))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# Much of this script is taken from the AMIA open source project loglog. More information here: https://github.com/amiaopensource/loglog

def setup_logger(): 
    # Suppress matplotlib verbose output before any matplotlib imports
    # This prevents matplotlib configuration messages from appearing
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.colorbar').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)  # Also suppress PIL/Pillow messages
    
    # Assigns getLogger function from imported module, creates logger 
    logger = logging.getLogger()
    # Sets 'lowest' log level
    logger.setLevel(logging.DEBUG)

    # Establishes path to 'logs' directory
    log_dir = get_log_directory()
    log_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_AVSpex"
    log_path = os.path.join(log_dir, f"{log_name}.log")

    # Define log formats which will be used in 'formatter' for the 2 log handlers
    LOG_FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
    STDOUT_FORMAT = '%(message)s' 

    ## This project uses 2 log handlers, one for the log file 'file_handler', and one for the terminal output 'console_handler' 
    # define file handler and set formatter
    file_handler = logging.FileHandler(log_path)
    formatter    = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    # set log level
    file_handler.setLevel(logging.DEBUG)
    # add file handler to logger
    logger.addHandler(file_handler)

    # define console_handler and set format for terminal output
    console_handler = colorlog.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s' + STDOUT_FORMAT,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    # add console handler to logger
    logger.addHandler(console_handler)

    return logger

# Initialize logger once on module import
logger = setup_logger() 

def connect_logger_to_ui(ui_component):
    """
    Connect the existing logger to a UI component without recreating the logger.
    Only adds a QtLogHandler to the existing logger.
    
    Args:
        ui_component: The UI component with update_status method
    
    Returns:
        The logger instance with the added Qt handler
    """
    if ui_component is not None and hasattr(ui_component, 'update_status'):
        # Check if a Qt handler is already connected to prevent duplicates
        for handler in logger.handlers:
            if isinstance(handler, QtLogHandler):
                # If there's already a Qt handler, disconnect old signals and connect new one
                handler.log_message.disconnect()
                handler.log_message.connect(ui_component.update_status)
                return logger
                
        # If no Qt handler exists, create and add a new one
        qt_handler = QtLogHandler()
        qt_handler.log_message.connect(ui_component.update_status)
        # Set log level - can adjust this to control what appears in the UI
        qt_handler.setLevel(logging.DEBUG)  
        # Add Qt handler to logger
        logger.addHandler(qt_handler)
    

# Example logs (only execute if this file is run directly, not imported)
if __name__ == "__main__":
    logger.debug('A debug message')
    logger.info('An info message')
    logger.warning('Something is not right.')
    logger.error('A Major error has happened.')
    logger.critical('Fatal error. Cannot continue')