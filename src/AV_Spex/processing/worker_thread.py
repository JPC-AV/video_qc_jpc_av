from PyQt6.QtCore import QThread, pyqtSignal

from AV_Spex.processing.avspex_processor import AVSpexProcessor
from AV_Spex.utils.log_setup import logger


class ProcessingWorker(QThread):
    """Worker thread for processing directories."""
    
    # Thread-specific signals
    started_processing = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(str)
    processing_time = pyqtSignal(str)
    
    def __init__(self, source_directories, signals, parent=None):
        super().__init__(parent)
        self.source_directories = source_directories
        self.signals = signals

        self.processor = AVSpexProcessor(signals=signals)
        self.user_cancelled = False
        
    def run(self):
        """Worker thread run method with pause handling"""
        try:
            if self.signals:
                self.signals.started.emit("Processing started")

            
            # Process directories
            result = self.processor.process_directories(self.source_directories)
            
            if result == "paused":
                print("DEBUG: Worker detected pause, not emitting completion")
                # Don't emit any completion signals for pause
                return
            elif result and not self.processor._cancelled:  # ADD THIS CHECK
                # Only emit completion if not cancelled
                self.processing_time.emit(result)
                if self.signals:
                    self.signals.completed.emit("Processing completed successfully!")
            elif self.processor._cancelled:
                print("DEBUG: Processing was cancelled, not emitting completion")
                # Don't emit completion signal for cancellation
                return
            else:
                # Failure case
                if self.signals:
                    self.signals.error.emit("Processing failed")
                    
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            print(f"DEBUG: Worker exception: {error_msg}")
            if self.signals:
                self.signals.error.emit(error_msg)
    
    def cancel(self):
        """Cancel the processing."""
        self.user_cancelled = True  # Set the flag
        if self.processor:
            self.processor.cancel()