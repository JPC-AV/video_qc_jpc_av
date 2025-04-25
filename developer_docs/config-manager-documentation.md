# ConfigManager Documentation

## Instantiation

The ConfigManager class uses the __new__ initialization method with the super() function to ensure only one instance is ever created:

```python
class ConfigManager:
    _instance = None  # Class-level variable to hold single instance
    _configs: Dict[str, Any] = {}  # Shared configuration cache dictionary
    
    # The __new__(cls) insures only one instance is ever created
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            # One-time initialization of paths and directories
```

`super()` is used to call methods from a parent/base class. In Python, all classes ultimately inherit from `object`, which provides the basic `__new__()` implementation.

Here's the sequence:

When `super().__new__(cls)` is called:

1. `super()` gets the parent class (which is `object` in this case)
2. It calls the parent's `__new__()` method
3. It passes `cls` (`ConfigManager`) as the argument
4. The parent `object.__new__(cls)` creates a bare instance of `ConfigManager`

## The Merge Process

The ConfigManager class uses a deep merge strategy when loading the Checks config and the Spex config to combine default settings with user-specific last-used settings.

### Loading Sequence

1. When `get_config()` is called for a config that isn't in cache:
   - Loads the JSON configuration using `_load_json_config()`
   - Creates a dataclass instance from the result using `_deserialize_dataclass()`

### Deep Merge (Recursive) 


The `_update_dict_recursively()` method implements a recursive dictionary merging strategy:

```python
def _update_dict_recursively(self, target: dict, source: dict) -> None:
    """Update dictionary recursively."""
    for key, value in source.items():
        if key in target:
            if isinstance(value, dict) and isinstance(target[key], dict):
                self._update_dict_recursively(target[key], value)
            else:
                target[key] = value
```

This function:
- Traverses both dictionaries recursively
- For nested dictionaries, continues merging deeper
- For non-dictionary values, the source value overwrites the target value


## Config Setup

As described in the previous section, any time `config_mgr = ConfigManager()` is called would instantiate the config, but the `_instance` clause of the `ConfigManager` class prevents more than one instance of the config existing at once.

After this initial instantiation, all other `ConfigManager` instantiations will receive the same singleton instance.

The first call of `get_config()` similarly creates an instance of the Checks config or the Spex config if one is not already in the cache dictionary `_configs`.  

```python
config_mgr = ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)
spex_config = config_mgr.get_config('spex', SpexConfig)
```

This calls the `ConfigManager`'s `get_config()` function, with the dataclass defined in `config_setup.py` passed as an argument:

```python
def get_config(self, config_name: str, config_class: Type[T], 
               use_last_used: bool = True) -> T:
    """
    Get configuration as a dataclass instance.
    
    Args:
        config_name: Name of the configuration
        config_class: Dataclass type to instantiate
        use_last_used: Whether to use the last used config if available
        
    Returns:
        An instance of the config_class
    """
    # Return cached config if available
    if config_name in self._configs:
        return self._configs[config_name]
        
    # Load config data from file
    config_data = self._load_json_config(config_name, use_last_used=use_last_used)
    
    # Convert to dataclass and cache
    config = self._deserialize_dataclass(config_class, config_data)
    self._configs[config_name] = config
    
    return config
```

The `_deserialize_dataclass()` helper function performs a recursive conversion of the JSON data into a proper dataclass instance, handling nested dataclasses, lists, dictionaries, and union types:

```python
def _deserialize_dataclass(self, cls: Type[T], data: Dict[str, Any]) -> T:
    """Convert a dictionary to a dataclass instance."""
    if data is None:
        return None
    
    # Get type hints for the class
    type_hints = get_type_hints(cls)
    
    # Initialize kwargs for class constructor
    kwargs = {}
    
    for field_name, field_value in data.items():
        if field_name not in type_hints:
            # Skip fields not in dataclass
            continue
        
        field_type = type_hints[field_name]
        kwargs[field_name] = self._process_field(field_type, field_value)
    
    # Create and return the dataclass instance
    return cls(**kwargs)
```

The `_process_field()` function and its helper methods handle the specific conversion of different field types:

```python
def _process_field(self, field_type: Type, field_value: Any) -> Any:
    """Process a field value based on its type."""
    if field_value is None:
        return None
    
    # Get origin and args for complex types
    origin = get_origin(field_type)
    args = get_args(field_type)
    
    # Handle based on type
    if origin is list or origin is List:
        return self._handle_list(args[0], field_value)
    elif origin is dict or origin is Dict:
        return self._handle_dict(args[0], args[1], field_value)
    elif origin is Union:
        return self._handle_union(args, field_value)
    elif hasattr(field_type, '__dataclass_fields__'):  # Is a dataclass
        return self._deserialize_dataclass(field_type, field_value)
    else:
        # Simple type, use as is
        return field_value
```

The dataclasses defined in `config_setup.py` are passed to the `_deserialize_dataclass()` helper function as an argument (initially passed to `get_config()`).  

### Type Handling

The approach implements systematic type handling through specialized helper methods:

```python
def _handle_list(self, item_type: Type, values: List) -> List:
    """Handle list type conversion."""
    if not isinstance(values, list):
        return values
    
    if hasattr(item_type, '__dataclass_fields__'):
        return [self._deserialize_dataclass(item_type, item) 
               for item in values if isinstance(item, dict)]
    return values

def _handle_dict(self, key_type: Type, value_type: Type, data: Dict) -> Dict:
    """Handle dictionary type conversion."""
    if not isinstance(data, dict):
        return data
    
    if hasattr(value_type, '__dataclass_fields__'):
        return {k: self._deserialize_dataclass(value_type, v) 
               for k, v in data.items() if isinstance(v, dict)}
    return data

def _handle_union(self, union_types: tuple, value: Any) -> Any:
    """Handle Union types (including Optional)."""
    if value is None:
        return None
    
    # Try each type in the union
    for union_type in union_types:
        if union_type is type(None):
            continue
            
        if hasattr(union_type, '__dataclass_fields__') and isinstance(value, dict):
            return self._deserialize_dataclass(union_type, value)
        elif isinstance(value, union_type):
            return value
    
    # If no specific handling, return as is
    return value
```

This approach:
- Handles nested dataclasses recursively through `_deserialize_dataclass()`
- Processes lists and dictionaries with the appropriate handling for their element types
- Supports Union types (including Optional fields) through `_handle_union()`
- Preserves basic types without modification

For each field, based on its type:

- For simple types (int, str, bool): retains the value directly
- For lists: uses `_handle_list()` to process each element according to the list's item type
- For dictionaries: uses `_handle_dict()` to process key-value pairs based on their types
- For Union types: uses `_handle_union()` to attempt matching against each possible type
- For nested dataclasses: recursively applies `_deserialize_dataclass()`

Here is an example from the Checks config JSON:

```json
"qct_parse": {
    "run_tool": "yes",        // string
    "barsDetection": true,    // boolean
    "evaluateBars": true,     // boolean
    "contentFilter": [],      // List[str]
    "profile": [],           // List[str]
    "tagname": null,         // Optional[str]
    "thumbExport": true      // boolean
}
```

The conversion process:

```python
# JSON input
data = {
    "run_tool": "yes",        # String - passed through unchanged
    "barsDetection": true,    # Boolean - passed through unchanged
    "contentFilter": [],      # Empty list - passed through unchanged
    "tagname": null,         # null becomes None
}

# Each field is processed based on its expected type:
1. For run_tool (string): Value remains "yes"
2. For barsDetection (boolean): Value remains True
3. For contentFilter (List[str]): Empty list remains []
4. For tagname (Optional[str]): null becomes None

# For nested dataclasses:
1. Get expected type from type hints
2. Process each field recursively using _process_field
3. Instantiate the dataclass with processed values
```

### Entry Points for Configuration Creation
The first instance of the ChecksConfig and SpexConfig are loaded at different point of the process in the CLI and GUI modes.  

**GUI Mode**   
In `gui_main_window.py`, the configuration objects are first created when the main window is initialized:
```python
self.config_mgr = ConfigManager()
self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
self.spex_config = self.config_mgr.get_config('spex', SpexConfig)
```

**CLI Mode**    
In `av_spex_the_file.py`, the global config_mgr is created, but the first actual retrieval of configurations happens during command processing, in the AVSpexProcessor:   

```python
# In AVSpexProcessor.__init__
self.config_mgr = ConfigManager()
self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
self.spex_config = self.config_mgr.get_config('spex', SpexConfig)
```

## The Refresh Challenge

When working with the singleton pattern across multiple modules in Python, each module that imports and instantiates the `ConfigManager` at import time will get its own cached copy of the configuration data. This can lead to a critical synchronization issue:

```python
# Module-level instantiation pattern (GUI modules)
config_mgr = ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)
```

While the singleton ensures there's only one `ConfigManager` instance per module, configuration changes made in the GUI aren't automatically reflected in the processing modules that load their configurations at different times.

## The Solution: Explicit Configuration Refresh

To address this issue, there is a dedicated method to explicitly refresh the configuration cache:

```python
def refresh_configs(self):
    """Force a reload of all configs from disk."""
    # Clear the cached configurations to force a reload from disk
    self._configs = {}
    logger.debug("Config cache cleared, configs will be reloaded from disk")
```

## Different Instantiation Patterns

The application uses two distinct patterns for configuration management:

### 1. GUI Components (Module-level Instantiation)
```python
# At module level (outside of class)
config_mgr = ConfigManager()
checks_config = config_mgr.get_config('checks', ChecksConfig)
spex_config = config_mgr.get_config('spex', SpexConfig)

class SomeGuiComponent:
    # Access configurations directly
```

### 2. Processing Classes (Instance-level Refresh)
```python
class AVSpexProcessor:
    def __init__(self, signals=None):
        self.signals = signals
        self._cancelled = False
        self._cancel_emitted = False 
        
        # Force a reload of the config from disk
        self.config_mgr = ConfigManager()
        self.config_mgr.refresh_configs()
        self.checks_config = self.config_mgr.get_config('checks', ChecksConfig)
        self.spex_config = self.config_mgr.get_config('spex', SpexConfig)
```

This pattern ensures that processing components always have the most current configuration data at the time processing begins, regardless of when configuration changes were made in the GUI.

## Best Practices

1. **GUI Components**: Use module-level instantiation for consistent configuration access across UI components
2. **Processing Classes**: Refresh configurations at initialization time to ensure the latest settings are used
3. **Cache Control**: Call `refresh_configs()` whenever a fresh copy of the configuration is needed
4. **Configuration Updates**: Continue using `update_config()` for making changes that will be reflected across the application

This pattern balances consistency and freshness, ensuring that all components work with appropriately timed configuration data.


## Editing Configs

The ConfigManager provides mechanisms for editing the Checks and Spex configs through the CLI or UI. These include targeted updates to specific fields, applying predefined profiles, and saving the updated configurations to persist changes between sessions.

### Updating Individual Settings

The `update_config()` method enables precise updates to configuration values:

```python
def update_config(self, config_name: str, updates: dict) -> None:
    """
    Update specific fields in a config.
    
    Args:
        config_name: Name of the config to update
        updates: Dictionary of updates to apply
    """
    config = self._configs.get(config_name)
    if not config:
        logger.error(f"No config found for {config_name}, cannot update")
        return
        
    # Convert current config to dict
    config_dict = asdict(config)
    
    # Apply updates recursively
    self._update_dict_recursively(config_dict, updates)
    
    # Convert back to dataclass
    self._configs[config_name] = self._deserialize_dataclass(type(config), config_dict)
    
    # Save as last used
    self.save_config(config_name, is_last_used=True)
```

This method:
- Converts the current dataclass instance to a dictionary
- Applies updates recursively using `_update_dict_recursively()`
- Converts the updated dictionary back to a dataclass using `_deserialize_dataclass()`
- Saves the updated configuration as the last used config


#### GUI Integration

The GUI uses this mechanism to handle checkbox state changes:

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

This allows each checkbox in the interface to directly modify its corresponding field in the configuration, with changes immediately reflected in the in-memory configuration.

### Applying Predefined Profiles

The system supports applying predefined configuration profiles that modify multiple settings at once through the `apply_profile()` function:

```python
def apply_profile(selected_profile):
    """Apply profile changes to checks_config.
    
    Args:
        selected_profile (dict): The profile configuration to apply
    """
    # Prepare the updates dictionary with the structure matching the dataclass
    updates = {}
    
    # Handle outputs section
    if 'outputs' in selected_profile:
        updates['outputs'] = selected_profile['outputs']
    
    # Handle fixity section
    if 'fixity' in selected_profile:
        updates['fixity'] = selected_profile['fixity']
    
    # Handle tools section with special cases
    if 'tools' in selected_profile:
        tools_updates = {}
        
        for tool_name, tool_updates in selected_profile['tools'].items():
            # No need for special cases - the update_config method will handle it
            tools_updates[tool_name] = tool_updates
        
        updates['tools'] = tools_updates
    
    # Apply all updates at once using the new update_config method
    if updates:
        config_mgr.update_config('checks', updates)
```

Predefined profiles are stored as dictionaries in `config_edit.py`:

```python
profile_step1 = {
    'tools': {
        'mediainfo': {'check_tool': 'yes', 'run_tool': 'yes'},
        'ffprobe': {'check_tool': 'yes', 'run_tool': 'yes'},
        'mediaconch': {'run_mediaconch': 'yes'},
        'qctools': {'run_tool': 'yes'},
        'qct_parse': {'run_tool': 'yes'},
        'exiftool': {'check_tool': 'yes', 'run_tool': 'yes'}
    },
    'fixity': {'check_fixity': 'yes'}
}
```

This approach allows for quick application of common configuration groups, such as turning on a set of tools for common workflow steps.

### Specialized Configuration Updates

For domain-specific configuration areas like signal flow profiles, specialized update functions are provided:

```python
def apply_signalflow_profile(selected_profile: dict):
    """
    Apply signalflow profile changes to spex_config.
    
    Updates encoder settings in both mediatrace and ffmpeg configurations
    with values from the provided profile.
    
    Args:
        selected_profile (dict): The signalflow profile to apply (encoder settings)
    """
    # First refresh configs to ensure we're working with the latest data
    config_mgr.refresh_configs()
    
    # Get the current spex config
    spex_config = config_mgr.get_config('spex', SpexConfig)
    
    # Validate input
    if not isinstance(selected_profile, dict):
        logger.critical(f"Invalid signalflow settings: {selected_profile}")
        return
    
    # Update mediatrace_values.ENCODER_SETTINGS
    # Each key in selected_profile should be a field in ENCODER_SETTINGS (like Source_VTR)
    for key, value in selected_profile.items():
        if hasattr(spex_config.mediatrace_values.ENCODER_SETTINGS, key):
            setattr(spex_config.mediatrace_values.ENCODER_SETTINGS, key, value)
    
    # Now update ffmpeg_values.format.tags.ENCODER_SETTINGS if it exists
    if (hasattr(spex_config, 'ffmpeg_values') and 
        'format' in spex_config.ffmpeg_values and 
        'tags' in spex_config.ffmpeg_values['format']):
        
        # Initialize ENCODER_SETTINGS as a dict if needed
        if 'ENCODER_SETTINGS' not in spex_config.ffmpeg_values['format']['tags'] or \
           spex_config.ffmpeg_values['format']['tags']['ENCODER_SETTINGS'] is None:
            spex_config.ffmpeg_values['format']['tags']['ENCODER_SETTINGS'] = {}
            
        # Update the settings
        for key, value in selected_profile.items():
            spex_config.ffmpeg_values['format']['tags']['ENCODER_SETTINGS'][key] = value
    
    # Update the cached config directly
    config_mgr._configs['spex'] = spex_config
    
    # Save the updated config to disk
    config_mgr.save_config('spex', is_last_used=True)
    
    logger.debug(f"Applied signalflow profile to configuration")
```

This function handles the specific complexities of updating the signal flow configuration, ensuring that settings are applied correctly to all relevant parts of the nested structure.

### CLI Configuration Management

The command-line interface provides options for targeted configuration changes through the `--on` and `--off` flags, which are processed by the `toggle_on()` and `toggle_off()` functions:

```python
def update_tool_setting(tool_names: List[str], value: str):
    """
    Update specific tool settings using config_mgr.update_config
    Args:
        tool_names: List of strings in format 'tool.field'
        value: 'yes' or 'no' (or True/False for qct_parse)
    """
    updates = {'tools': {}}
    
    for tool_spec in tool_names:
        try:
            tool_name, field = tool_spec.split('.')
            
            # Special handling for qct_parse which uses booleans instead of yes/no
            if tool_name == 'qct_parse':
                if value.lower() not in ('yes', 'no'):
                    logger.warning(f"Invalid value '{value}' for qct_parse. Must be 'yes' or 'no'")
                    continue
                bool_value = True if value.lower() == 'yes' else False
                updates['tools'][tool_name] = {field: bool_value}
                
            # Special handling for mediaconch which has different field names
            elif tool_name == 'mediaconch':
                if field not in ('run_mediaconch'):
                    logger.warning(f"Invalid field '{field}' for mediaconch. To turn mediaconch on/off use 'mediaconch.run_mediaconch'.")
                    continue
                updates['tools'][tool_name] = {field: value}

            elif tool_name == 'fixity':
                updates['fixity'] = {}
                if field not in ('check_fixity','validate_stream_fixity','embed_stream_fixity','output_fixity','overwrite_stream_fixity'):
                    logger.warning(f"Invalid field '{field}' for fixity settings")
                    continue
                updates['fixity'][field] = value
                
            # Standard tools with check_tool/run_tool fields
            else:
                if field not in ('check_tool', 'run_tool'):
                    logger.warning(f"Invalid field '{field}' for {tool_name}. Must be 'check_tool' or 'run_tool'")
                    continue
                updates['tools'][tool_name] = {field: value}
                
            logger.debug(f"{tool_name}.{field} will be set to '{value}'")
            
        except ValueError:
            logger.warning(f"Invalid format '{tool_spec}'. Expected format: tool.field")
    
    if updates:  # Only update if we have changes
        config_mgr.update_config('checks', updates)

def toggle_on(tool_names: List[str]):
    update_tool_setting(tool_names, 'yes')

def toggle_off(tool_names: List[str]):
    update_tool_setting(tool_names, 'no')
```

This implementation allows for precise control through the command line:

```bash
# Turn on mediainfo tool
python av_spex_the_file.py --on mediainfo.run_tool

# Turn off exiftool and fixed
python av_spex_the_file.py --off exiftool.run_tool --off fixity.check_fixity
```


### Persisting Configuration Changes

After modifying a configuration, changes can be persisted to disk using the `save_config()` method:

```python
def save_config(self, config_name: str, is_last_used: bool = False) -> None:
    """
    Save current config to a file.
    
    Args:
        config_name: Name of the config to save
        is_last_used: If True, save as last_used config, otherwise as regular config
    """
    config = self._configs.get(config_name)
    if not config:
        logger.error(f"No config found for {config_name}, cannot save")
        return
        
    filename = f"{'last_used_' if is_last_used else ''}{config_name}_config.json"
    save_path = os.path.join(self._user_config_dir, filename)
    
    try:
        with open(save_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        logger.info(f"Saved config to {save_path}")
    except Exception as e:
        logger.error(f"Error saving config to {save_path}: {str(e)}")
```

This method:
1. Retrieves the configuration from the cache
2. Converts the dataclass instance to a dictionary using `asdict()`
3. Writes the configuration to a JSON file, using the `is_last_used` flag to determine the filename

### Resetting Configurations

The `reset_config()` method provides a way to restore a configuration to its default values:

```python
def reset_config(self, config_name: str, config_class: Type[T]) -> T:
    """
    Reset config to default values.
    
    Args:
        config_name: Name of the config to reset
        config_class: Type of the config class
        
    Returns:
        The reset config instance
    """
    # Remove from cache
    if config_name in self._configs:
        del self._configs[config_name]
        
    # Remove last used config file if it exists
    last_used_path = os.path.join(self._user_config_dir, f"last_used_{config_name}_config.json")
    if os.path.exists(last_used_path):
        try:
            os.remove(last_used_path)
            logger.info(f"Removed last used config file {last_used_path}")
        except Exception as e:
            logger.error(f"Failed to remove {last_used_path}: {str(e)}")
                
    # Load fresh config
    return self.get_config(config_name, config_class, use_last_used=False)
```

This method:
1. Removes the configuration from the in-memory cache
2. Deletes any last-used configuration file for this configuration
3. Loads a fresh configuration from the default config file
4. Returns the reset configuration instance

The `refresh_configs()` method provides a more general way to clear the entire configuration cache:

```python
def refresh_configs(self):
    """Force reload of all configs from disk"""
    self._configs = {}
    logger.debug(f"Config cache cleared, configs will be reloaded from disk\n")
```

This method simply clears the configuration cache, ensuring that the next time a configuration is requested, it will be reloaded from disk.