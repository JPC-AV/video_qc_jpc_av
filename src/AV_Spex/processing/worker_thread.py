from PyQt6.QtCore import QThread, pyqtSignal

from AV_Spex.processing.avspex_processor import AVSpexProcessor
from AV_Spex.processing.dry_run_analyzer import DryRunAnalyzer
from AV_Spex.utils.log_setup import logger


class ProcessingWorker(QThread):
    """Worker thread for processing directories."""
    
    # Thread-specific signals
    started_processing = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(str)
    processing_time = pyqtSignal(str)
    dry_run_finished = pyqtSignal(dict)  # Signal for dry run results
    
    def __init__(self, source_directories, signals, dry_run=False, parent=None):
        super().__init__(parent)
        self.source_directories = source_directories
        self.signals = signals
        self.dry_run = dry_run
        
        # Create the appropriate processor
        if dry_run:
            self.processor = DryRunAnalyzer(signals=signals)
        else:
            self.processor = AVSpexProcessor(signals=signals)

        self.user_cancelled = False
        
    def run(self):
        """Run the worker thread."""
        try:
            # Emit that we started
            self.started_processing.emit()
            
            if self.dry_run:
                # Run dry run analysis
                results = self.processor.analyze_directories(self.source_directories)
                if not self.user_cancelled:
                    self.dry_run_finished.emit(results)
            else:
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