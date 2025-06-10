# AV Spex GUI Developer Documentation

## Overview

This document describes the code that defines the GUI mode of the AV Spex application.

The GUI is created using the PyQt6, a set of Python bindings for Qt6 library.   

PyQt6's import structure is organized into several modules, the main ones are:

 - `PyQt6.QtWidgets`
    Contains all UI components (widgets) for building interfaces:
    ```python
    from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
    ```
- `PyQt6.QtCore`
    Contains core non-GUI functionality:
    ```python
    from PyQt6.QtCore import Qt, QSize, QTimer, QRect, pyqtSignal
    ```
- `PyQt6.QtGui`
    Contains GUI-related classes that aren't widgets:
    ```python
    from PyQt6.QtGui import QIcon, QFont, QColor, QPainter, QPixmap
    ```

The AV Spex GUI code is divided across several scripts:
- gui_checks_window.py
- gui_console_textbox.py
- gui_main_window.py
- gui_processing_gui.py
- gui_signals.py
- gui_theme_manager.py

## Entry Point: 
The gui is launched from the main_gui() function, which triggers the Loader:

```python
def main_gui():
    args = parse_arguments()
    
    # Get application 
    app = LazyGUILoader.get_application()
    
    # Get main window 
    window = LazyGUILoader.get_main_window()
    window.show()
    
    return app.exec()

class LazyGUILoader:
    _app = None
    _ChecksWindow = None
    _MainWindow = None
    _QApplication = None
    
    @classmethod
    def load_gui_components(cls):
        if cls._QApplication is None:
            from PyQt6.QtWidgets import QApplication
            # Update imports to use the new UI modules
            from .gui.gui_checks_window import ChecksWindow
            from .gui.gui_main_window import MainWindow
            cls._QApplication = QApplication
            cls._ChecksWindow = ChecksWindow
            cls._MainWindow = MainWindow
            
    @classmethod
    def get_application(cls):
        cls.load_gui_components()
        if cls._app is None:
            cls._app = cls._QApplication(sys.argv)
        return cls._app
    
    @classmethod
    def get_main_window(cls):
        cls.load_gui_components()
        return cls._MainWindow()
```

## Main Window

The `gui_main_window.py` contains the `MainWindow` class, which serves as the central UI component for the AV Spex application. This class creates a tabbed interface, manages processing workflows, and coordinates theme handling throughout the application.

### Core Methods:
- `setup_ui()`: Initializes the main UI components and layout
- `setup_tabs()`: Creates the tabbed interface for configuration
    -  `setup_ui()` establishes the window structure
    - Calls `setup_main_layout()` to create the core layout
    - Calls `logo_setup()` to load application branding
    - Calls `setup_tabs()` to create the tabbed interface, which then calls:
        - `setup_checks_tab()` to build the Checks configuration UI
        - `setup_spex_tab()` to build the Spex configuration UI
- `call_process_directories()`: creates worker thread and connects worker-specific signals
    - Calls `initialize_processing_window()` if needed
        - `initialize_processing_window()` creates and configures the [processing window](#processing-window-and-console-text-box)
    - Signal connection: Connections between the GUI and Processor to report progress throughout the process (Described in detail in [AV Spex GUI Processing Signals Flow section](#AVSpexGUIProcessingSignalsFlow))

## Checks Window

The `gui_checks_window.py` contains the `ChecksWindow` class, which provides an interface for displaying and editing the `ChecksConfig`. The window is added as a widget to the `MainWindow` in `setup_checks_tab()` function.

### Core Methods:
- `setup_ui()`: Initializes the Checks interface
  - Creates the main layout structure
  - Calls section-specific setup methods in sequence (outputs section, fixity section, tools section)
  - Connects all UI signals to their handlers
- `setup_outputs_section(main_layout)`
- `setup_fixity_section(main_layout)`
- `setup_tools_section(main_layout)`

Each section utilizes the ThemeManager to maintain consistent styling, using Qt GroupBoxes and the ThemeManager's `style_groupbox()` function (more in the [Theme Manager section](#theme-manager-documentation)). Signal connections trigger updates through handler methods that work with the `ConfigManager` to persist changes to `ChecksConfig`.

As described in the [ConfigManager documentation](https://github.com/JPC-AV/JPC_AV_videoQC/blob/main/developer_docs/config-manager-documentation.md), the updates to the `ChecksConfig` from the `ChecksWindow` primarily make use of the `ConfigManager`'s `update_config()` function, such as in the `on_checkbox_changed` function:

```python
 def on_checkbox_changed(self, state, path):
        """Handle changes in yes/no checkboxes"""
        new_value = 'yes' if Qt.CheckState(state) == Qt.CheckState.Checked else 'no'
        
        if path[0] == "tools" and len(path) > 2:
            tool_name = path[1]
            field = path[2]
            updates = {'tools': {tool_name: {field: new_value}}}
        else:
            section = path[0]
            field = path[1]
            updates = {section: {field: new_value}}
            
        self.config_mgr.update_config('checks', updates)
```

## Processing Window and Console Text Box

The `gui_processing_gui.py` contains the `ProcessingWindow` class, which provides real-time visualization of processing operations.

### Status Display Methods:

#### file_status_label
- `update_file_status(filename, current_index=None, total_files=None)`: Updates the main file processing label with current file and progress count

#### progress_bar
- `update_file_status(filename, current_index=None, total_files=None)`: Updates main progress bar based on file count
- `update_progress(current, total)`: Sets progress bar value and maximum 

#### steps_list
- `populate_steps_list()`: Builds the processing checklist based on `ChecksConfig`
- `mark_step_complete(step_name)`: Marks a step as completed with checkmark and bold formatting
- `reset_steps_list()`: Resets all steps to initial state when processing a new file
- `_add_step_item(step_name)`: Helper method that adds unchecked steps to the list

#### details_text
- `update_status(message, msg_type=None)`: Adds color-coded messages to the console output
  - Automatically detects message type (error, warning, command, success, info)

The `details_text` component is an instance of `ConsoleTextEdit`, a customized `QTextEdit` widget defined in `gui_console_textbox.py`.

The ProcessingWindow's `update_status()` function is imported into the logger (`AV_Spex.utils.log_setup`) to push logging messages to the text box:

```python
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
```

#### detailed_status
- `update_detailed_status(message)`: Updates the detailed status message below the console
- `update_detail_progress(percentage)`: Updates the detailed progress bar with percentage
  - Sets progress value and updates overlay percentage text
  - Resets progress when starting new operations
  - Uses theme-aware colors for bar and text

The progress tracking uses signals defined in the signals system documented in the [Processing Signals Flow section](#av-spex-gui-processing-signals-flow).

## AV Spex GUI Processing Signals Flow

The AV Spex GUI uses PyQt's signal-slot mechanism to handle processing events and update the user interface asynchronously. This document outlines the signal flow architecture between the main components of the application to show how processing events are communicated throughout the system.

### 1. Signal Definition (`ProcessingSignals` Class)

The `ProcessingSignals` class (in `gui_signals.py`) defines all the custom signals used throughout the application:

```python
class ProcessingSignals(QObject):
    started = pyqtSignal(str)           # Processing started
    completed = pyqtSignal(str)         # Processing completed
    error = pyqtSignal(str)             # Error occurred
    cancelled = pyqtSignal()            # Processing cancelled
    
    status_update = pyqtSignal(str)     # General status updates
    progress = pyqtSignal(int, int)     # Numerical progress (current, total)

    file_started = pyqtSignal(str)      # File processing started
    tool_started = pyqtSignal(str)      # Tool processing started
    tool_completed = pyqtSignal(str)    # Tool processing completed
    step_completed = pyqtSignal(str)    # Processing step completed
    
    fixity_progress = pyqtSignal(str)   # Fixity status updates
    mediaconch_progress = pyqtSignal(str) # MediaConch status updates
    metadata_progress = pyqtSignal(str) # Metadata status updates
    output_progress = pyqtSignal(str)   # Output creation status updates

    stream_hash_progress = pyqtSignal(int)  # Signal for stream hash progress percentage
    md5_progress = pyqtSignal(int)          # Signal for MD5 calculation progress percentage
    access_file_progress = pyqtSignal(int)  # Signal for access file creation progress percentage
```

### 2. Main Window (`MainWindow` Class)

The `MainWindow` class:
- Instantiates the `ProcessingSignals` object
- Connects signals to appropriate handler methods
- Creates and manages the `ProcessingWindow`
- Creates and manages the worker thread

### 3. Processing Window (`ProcessingWindow` Class)

The `ProcessingWindow` class:
- Displays status messages, progress bar, and step completion status
- Updates UI elements based on signals
- Contains the cancel button to terminate processing

### 4. Worker Thread (`ProcessingWorker` Class)

The `ProcessingWorker` class:
- Runs the actual processing in a separate thread
- Emits worker-specific signals such as `started_processing`, `finished`, etc.
- Forwards signals from the processor

### 5. Processor (`AVSpexProcessor` Class)

The `AVSpexProcessor` class:
- Performs the actual media file processing
- Emits signals about the processing status

## Signal Flow Sequence

```
┌─────────────┐    1. Creates    ┌─────────────┐
│  MainWindow │────────────────▶ │ ProcessingWindow │
└─────┬───────┘                 └───────┬─────┘
      │                                 │
      │ 2. Creates                      │
      ▼                                 │
┌─────────────┐    3. Runs     ┌─────────────┐
│ Worker Thread│────────────────▶ │ AVSpexProcessor │
└─────┬───────┘                 └───────┬─────┘
      │                                 │
      │ 4. Emits worker signals         │ 5. Emits processing signals
      │                                 │
      │                                 │
      ▼                                 ▼
┌─────────────────────────────────────────────┐
│               Signal Bus                     │
└─────────────────┬───────────────────────────┘
                  │
                  │ 6. Signals routed to handlers
                  │
                  ▼
┌─────────────────────────────────────────────┐
│       MainWindow signal handler methods      │
└─────────────────┬───────────────────────────┘
                  │
                  │ 7. Update UI
                  │
                  ▼
┌─────────────────────────────────────────────┐
│              ProcessingWindow                │
└─────────────────────────────────────────────┘
```

## Signal Connection Setup

In the `MainWindow` class, the `setup_signal_connections` method connects signals to their respective handler methods:

```python
def setup_signal_connections(self):
     # Processing window signals
    self.signals.started.connect(self.on_processing_started)
    self.signals.completed.connect(self.on_processing_completed)
    self.signals.error.connect(self.on_error)
    self.signals.cancelled.connect(self.on_processing_cancelled)

    # Connect file_started signal to update main status label
    self.signals.file_started.connect(self.update_main_status_label)
    
    # Tool-specific signals
    self.signals.tool_started.connect(self.on_tool_started)
    self.signals.tool_completed.connect(self.on_tool_completed)
    self.signals.fixity_progress.connect(self.on_fixity_progress)
    self.signals.mediaconch_progress.connect(self.on_mediaconch_progress)
    self.signals.metadata_progress.connect(self.on_metadata_progress)
    self.signals.output_progress.connect(self.on_output_progress)
```

## Signal Handler Methods

The signal handler methods in `MainWindow` update the `ProcessingWindow` UI:

```python
def on_tool_started(self, tool_name):
    """Handle tool processing start"""
    if self.processing_window:
        self.processing_window.update_status(f"Starting {tool_name}")

def on_tool_completed(self, message):
    """Handle tool processing completion"""
    if self.processing_window:
        self.processing_window.update_status(message)
        # Let UI update
        QApplication.processEvents()
```

## Worker Thread Signal Connections

When initializing the worker thread, specific worker signals are connected:

```python
def call_process_directories(self):
    # ...
    self.worker = ProcessingWorker(self.source_directories, self.signals)
    
    # Connect worker-specific signals
    self.worker.started_processing.connect(self.on_processing_started)
    self.worker.finished.connect(self.on_worker_finished)
    self.worker.error.connect(self.on_error)
    self.worker.processing_time.connect(self.on_processing_time)
    
    # Start the worker thread
    self.worker.start()
```

## Signal Emission Points

### In the Worker Thread

```python
def run(self):
    try:
        # Signal that processing has started
        self.started_processing.emit()
        
        # Process directories...
        
        # Signal completion with timing information
        self.processing_time.emit(processing_time)
        self.finished.emit()
    
    except Exception as e:
        self.error.emit(f"Processing error: {str(e)}")
```

### In the Processor

```python
def process_single_directory(self, source_directory):
    if self.check_cancelled():
        return False

    # Process fixity...
    if fixity_enabled:
        if self.signals:
            self.signals.tool_started.emit("Fixity...")
        
        # Processing...
        
        if self.signals:
            self.signals.tool_completed.emit("Fixity processing complete")
```


# Theme Manager Documentation

## Overview

The Theme Manager system in our PyQt6 application provides a centralized mechanism for applying consistent theming across the application, especially when switching between light and dark modes. It allows all UI components to respond to system palette changes and maintain visual consistency throughout the application.

## Architecture

The theme management system consists of two core components:

1. **ThemeManager**: A singleton class responsible for detecting palette changes and providing styling methods
2. **ThemeableMixin**: A mixin class that can be added to any widget that needs to respond to theme changes

### Class Diagram

```
┌───────────────────┐     Theme Change     ┌──────────────────┐
│                   │       Events         │                  │
│   QApplication    │───────────────────▶ │   ThemeManager   │
│                   │                      │                  │
└───────────────────┘                      └──────────────────┘
                                                    │
                                                    │ Notifies
                                                    ▼
┌───────────────────┐     Implements     ┌──────────────────┐
│                   │                    │                  │
│    MainWindow     │──────────────────▶│  ThemeableMixin  │
│  ProcessingWindow │                    │                  │
│   ConfigWindow    │                    │                  │
└───────────────────┘                    └──────────────────┘
```

## Key Components

### ThemeManager Class

The `ThemeManager` class is implemented as a singleton to ensure that only one instance exists in the application. It monitors the system palette changes and provides styling methods for different UI components.

#### Key Features:

- **Singleton Pattern**: Only one instance of ThemeManager exists at any time
- **System Palette Monitoring**: Connects to QApplication's paletteChanged signal
- **Theme Change Notifications**: Emits signals when theme changes are detected
- **Styling Utilities**: Provides methods for consistent styling of UI components

### ThemeableMixin Class

The `ThemeableMixin` class is a mixin that can be applied to any QWidget-derived class that needs to respond to theme changes.

#### Key Features:

- **Theme Change Handling**: Automatically responds to theme changes
- **Simple Integration**: Easy to add to any PyQt widget class
- **Customizable Response**: Can be overridden to customize theme behavior
- **Proper Cleanup**: Includes methods to disconnect from theme signals

## Signal Flow

When the system palette changes, the following sequence occurs:

1. QApplication emits a `paletteChanged` signal with the new palette
2. ThemeManager receives this signal and emits its own `themeChanged` signal
3. All widgets that implement ThemeableMixin receive the `themeChanged` signal
4. Each widget's `on_theme_changed` method is called to apply the new styling

## Integration with UI Components

### MainWindow

The main application window integrates with the theme system as follows:

1. Inherits from `ThemeableMixin` to receive theme updates
2. Calls `setup_theme_handling()` during initialization
3. Implements `on_theme_changed(palette)` to handle theme updates 
4. Updates all child widgets when the theme changes
5. Calls `cleanup_theme_handling()` during window closure

### Child Windows (ProcessingWindow, ConfigWindow)

Child windows follow a similar pattern:

1. Inherit from `ThemeableMixin`
2. Call `setup_theme_handling()` during initialization
3. Implement custom `on_theme_changed(palette)` methods
4. Clean up theme connections on window closure

## Styling Methods

The ThemeManager provides several styling methods that can be used to maintain consistent appearance:

### style_groupbox(group_box, title_position)

Applies consistent styling to QGroupBox widgets:
- Sets border, background color, and text color based on current palette
- Positions the title according to the specified position (or preserves existing position)
- Ensures consistent appearance across light and dark themes

### style_buttons(parent_widget)

Applies consistent styling to all QPushButton widgets within a parent widget:
- Sets background color, text color, borders, and hover effects
- Uses palette colors to ensure consistency with system theme
- Automatically finds and styles all buttons within the parent widget

### get_tab_style()

Returns a stylesheet for QTabWidget and QTabBar components:
- Creates tab styling that adapts to the current palette colors
- Provides consistent styling for tabs across the application
- Includes hover and selected state styling

## Usage Examples

### Adding Theme Support to a New Window

```python
from PyQt6.QtWidgets import QMainWindow
from ..utils.theme_manager import ThemeableMixin

class MyNewWindow(QMainWindow, ThemeableMixin):
    def __init__(self):
        super().__init__()
        
        # Set up UI
        # ...
        
        # Set up theme handling
        self.setup_theme_handling()
    
    def on_theme_changed(self, palette):
        # Apply palette to this window
        self.setPalette(palette)
        
        # Get theme manager for styling
        theme_manager = ThemeManager.instance()
        
        # Style components
        theme_manager.style_groupbox(self.my_group_box, "top center")
        theme_manager.style_buttons(self)
        
        # Update the window
        self.update()
    
    def closeEvent(self, event):
        # Clean up theme connections
        self.cleanup_theme_handling()
        super().closeEvent(event)
```

### Applying Styling to UI Components

```python
# In a UI setup method
def setup_ui(self):
    # Create a group box
    self.settings_group = QGroupBox("Settings")
    
    # Apply theme-aware styling
    theme_manager = ThemeManager.instance()
    theme_manager.style_groupbox(self.settings_group, "top center")
    
    # Later, when adding buttons
    self.apply_button = QPushButton("Apply")
    self.cancel_button = QPushButton("Cancel")
    
    # Style all buttons at once
    theme_manager.style_buttons(self)
```

# Dependency Manager

The Dependency Manager system provides a robust mechanism for checking external CLI tool dependencies required by the AV Spex application. It offers both GUI and CLI checking modes, with detailed feedback about missing dependencies and installation guidance for users.

## Architecture

The dependency management system consists of three core components:

1. **DependencyInfo**: A dataclass that encapsulates information about each required dependency
2. **DependencyCheckWorker**: A QThread worker that performs non-blocking dependency checks in the GUI
3. **DependencyManager**: A static class that orchestrates dependency checking for both GUI and CLI modes

### Class Diagram

```
┌───────────────────┐     Creates     ┌──────────────────┐
│                   │                 │                  │
│ DependencyManager │────────────────▶│ DependencyInfo   │
│                   │                 │                  │
└─────────┬─────────┘                 └──────────────────┘
          │
          │ Creates (GUI mode)
          ▼
┌───────────────────┐    Runs checks   ┌──────────────────┐
│ DependencyCheck   │─────────────────▶│ DependencyCheck  │
│     Dialog        │                  │     Worker       │
└───────────────────┘                  └──────────────────┘
```

## Key Components

### DependencyInfo Class

The `DependencyInfo` dataclass encapsulates all information about a single dependency:

```python
@dataclass
class DependencyInfo:
    name: str                           # Display name (e.g., "FFmpeg")
    command: str                        # CLI command (e.g., "ffmpeg")
    version_command: Optional[str]      # Version check command
    min_version: Optional[str]          # Minimum required version
    description: str                    # User-friendly description
    install_hint: str                   # Installation instructions
    status: DependencyStatus            # Current check status
    version_found: Optional[str]        # Detected version
    error_message: Optional[str]        # Error details if check failed
```

### DependencyCheckWorker Class

The worker thread performs dependency checks without blocking the GUI:

- **Non-blocking Checks**: Runs in separate thread to maintain UI responsiveness
- **Real-time Updates**: Emits signals as each dependency is checked
- **Version Validation**: Checks both existence and version requirements when specified
- **Cancellation Support**: Can be interrupted if user cancels the operation

### DependencyCheckDialog Class

The GUI dialog provides these options dependency checking:

- **Progress Tracking**: Shows real-time progress bar and status updates
- **Color-coded Results**: Uses visual indicators (✅❌⚠️) for dependency status
- **Installation Guidance**: Displays installation hints for missing dependencies
- **User Choice**: Continue anyway or exit to install missing tools

## Signal Flow Sequence

```
┌─────────────┐    1. Creates    ┌─────────────┐
│ Application │────────────────▶ │ DependencyCheck │
│  Startup    │                 │    Dialog       │
└─────────────┘                 └───────┬─────────┘
                                        │
                                        │ 2. Creates
                                        ▼
                                ┌─────────────┐
                                │ DependencyCheck │
                                │    Worker       │
                                └───────┬─────────┘
                                        │
                                        │ 3. Emits check results
                                        │
                                        ▼
                                ┌─────────────────────┐
                                │    Signal Bus       │
                                └─────────┬───────────┘
                                          │
                                          │ 4. Updates UI
                                          │
                                          ▼
                                ┌─────────────────────┐
                                │  Dialog UI Updates  │
                                └─────────────────────┘
```

## Required Dependencies

The system currently checks for five external CLI tools:

1. **FFmpeg** (`ffmpeg`) - Video/audio processing
2. **MediaInfo** (`mediainfo`) - Media metadata extraction  
3. **ExifTool** (`exiftool`) - Metadata extraction
4. **MediaConch** (`mediaconch`) - Media conformance checking
5. **QCTools** (`qcli`) - Quality control analysis and video QC metrics

