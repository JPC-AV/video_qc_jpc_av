from PyQt6.QtCore import QObject, pyqtSignal

class ProcessingSignals(QObject):
    started = pyqtSignal(str)  # Signal for when processing starts
    completed = pyqtSignal(str)  # Signal for when processing completes
    error = pyqtSignal(str)  # Signal for when an error occurs
    cancelled = pyqtSignal() # For cancel button

    # New pause/resume signals
    pause_requested = pyqtSignal()  # Signal to request pause
    resume_requested = pyqtSignal()  # Signal to request resume
    paused = pyqtSignal()  # Signal when actually paused
    resumed = pyqtSignal()  # Signal when resumed
    
    status_update = pyqtSignal(str)  # Signal for general status updates
    progress = pyqtSignal(int, int)  # Signal for numerical progress (current, total)

    file_started = pyqtSignal(str, int, int)  # Signal for when processing of a specific file starts
    tool_started = pyqtSignal(str)  # Signal for when a specific tool starts
    tool_completed = pyqtSignal(str)  # Signal for when a tool completes
    step_completed = pyqtSignal(str)  # Signal when a specific step is completed
    
    fixity_progress = pyqtSignal(str)  # For detailed fixity status
    mediaconch_progress = pyqtSignal(str)  # For detailed mediaconch status
    metadata_progress = pyqtSignal(str)  # For detailed metadata status
    output_progress = pyqtSignal(str)  # For detailed output creation status

    stream_hash_progress = pyqtSignal(int)  # Signal for stream hash progress percentage
    md5_progress = pyqtSignal(int)          # Signal for MD5 calculation progress percentage
    access_file_progress = pyqtSignal(int)  # Signal for access file creation progress percentage
    qctools_progress = pyqtSignal(int)       # Signal for qctools xml creation progress percentage