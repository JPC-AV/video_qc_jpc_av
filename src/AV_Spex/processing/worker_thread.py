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
        """Run the worker thread."""
        try:
            # Emit that we started
            self.started_processing.emit()
            
            # Process the directories
            processing_time = self.processor.process_directories(self.source_directories)
            
            if processing_time:
                self.processing_time.emit(processing_time)
            
            self.finished.emit()
        
        except Exception as e:
            logger.exception("Error in processing worker")
            self.error.emit(f"Processing error: {str(e)}")
            # Commenting this out to keep the window open after an error
            # self.finished.emit()
    
    def cancel(self):
        """Cancel the processing."""
        self.user_cancelled = True  # Set the flag
        if self.processor:
            self.processor.cancel()