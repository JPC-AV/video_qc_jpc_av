from dataclasses import asdict
from typing import Optional, TypeVar, Type, Dict, Any, get_type_hints
import json
import os
from pathlib import Path
import appdirs
import sys

from ..utils.log_setup import logger

T = TypeVar('T')

class ConfigManager:
    _instance = None  # Class-level variable to hold single instance
    _configs: Dict[str, Any] = {}  # Shared configuration cache
    
    # The __new__(cls) insures only one instance is ever created
    def __new__(cls):
        if cls._instance is None:
            # If no config class instance exists, create a instance of ConfigManager.
            # super() calls the parent class to get ConfigManager's __new__ from within ConfigManager
            cls._instance = super(ConfigManager, cls).__new__(cls)
            
            # One-time initialization of paths and directories
            if getattr(sys, 'frozen', False):
                cls._instance._bundle_dir = os.path.join(sys._MEIPASS, 'AV_Spex')
            else:
                cls._instance._bundle_dir = os.path.dirname(os.path.dirname(__file__))
            
            # Set up user config directory using appdirs
            cls._instance._user_config_dir = appdirs.user_config_dir(
                appname="AVSpex",
                appauthor="NMAAHC"
            )
            
            # Set up logo files directory path
            cls._instance._logo_files_dir = os.path.join(cls._instance._bundle_dir, 'logo_image_files')
            
            # Set up policies directory paths
            cls._instance._bundled_policies_dir = os.path.join(cls._instance._bundle_dir, 'config', 'mediaconch_policies')
            cls._instance._user_policies_dir = os.path.join(cls._instance._user_config_dir, 'mediaconch_policies')
            
            # Ensure directories exist
            os.makedirs(cls._instance._user_config_dir, exist_ok=True)
            os.makedirs(cls._instance._user_policies_dir, exist_ok=True)
            
            # Verify bundled configs exist
            bundle_config_dir = os.path.join(cls._instance._bundle_dir, 'config')
            if not os.path.exists(bundle_config_dir):
                raise FileNotFoundError(f"Bundled config directory not found at {bundle_config_dir}")
            
        return cls._instance

    def get_logo_path(self, logo_filename: str) -> Optional[str]:
        """
        Get the full path for a logo file in the bundled logo_image_files directory
        
        Args:
            logo_filename: Name of the logo file
            
        Returns:
            Optional[str]: Full path to the logo file or None if not found
        """
        logo_path = os.path.join(self._logo_files_dir, logo_filename)
        if os.path.exists(logo_path):
            # Quote the path if it contains spaces
            return f'"{logo_path}"' if ' ' in logo_path else logo_path
        return None
    
    def get_available_policies(self) -> list[str]:
        """Get all available policy files from both bundled and user directories"""
        policies = set()  # Use set to avoid duplicates
        
        # Get bundled policies
        if os.path.exists(self._bundled_policies_dir):
            policies.update(f for f in os.listdir(self._bundled_policies_dir) if f.endswith('.xml'))
            
        # Get user policies
        if os.path.exists(self._user_policies_dir):
            policies.update(f for f in os.listdir(self._user_policies_dir) if f.endswith('.xml'))
            
        return sorted(list(policies))

    def get_policy_path(self, policy_name: str) -> Optional[str]:
        """Get the full path for a policy file, checking user dir first then bundled"""
        # Check user directory first
        user_path = os.path.join(self._user_policies_dir, policy_name)
        if os.path.exists(user_path):
            # Quote the path if it contains spaces
            return f'"{user_path}"' if ' ' in user_path else user_path
            
        # Then check bundled policies
        bundled_path = os.path.join(self._bundled_policies_dir, policy_name)
        if os.path.exists(bundled_path):
            # Quote the path if it contains spaces
            return f'"{bundled_path}"' if ' ' in bundled_path else bundled_path
            
        return None

    def find_file(self, filename: str, user_config: bool = False) -> Optional[str]:
        """
        Find a config file in either the bundled configs or user config directory.
        
        Args:
            filename: Name of the config file
            user_config: If True, look in user config directory instead of bundled configs
        """
        if user_config:
            file_path = os.path.join(self._user_config_dir, filename)
        else:
            file_path = os.path.join(self._bundle_dir, 'config', filename)
            
        return file_path if os.path.exists(file_path) else None

    def _load_json_config(self, config_name: str, last_used: bool = False) -> dict:
        """
        Load JSON config file by name
        Args:
            config_name: Name of the config file
            last_used: If True, look for last_used_{config_name}_config.json in user directory
        """
        if last_used:
            last_used_path = self.find_file(
                f"last_used_{config_name}_config.json",
                user_config=True
            )
            if last_used_path and os.path.exists(last_used_path):
                try:
                    with open(last_used_path, 'r') as f:
                        data = json.load(f)
                        # logger.debug(f"Loaded last used config from {last_used_path}")
                        return data
                except json.JSONDecodeError:
                    logger.critical(f"Error loading last used config, falling back to defaults")

        # Load default config from bundled configs
        json_path = self.find_file(f"{config_name}_config.json", user_config=False)
        
        if not json_path or not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Default config file {config_name}_config.json not found in bundled configs."
            )
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                # logger.debug(f"Loaded default config from {json_path}")
                return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing {config_name}_config.json: {str(e)}")

    def save_last_used_config(self, config_name: str) -> None:
        """Save current config state as last used settings in user config directory"""
        config = self._configs.get(config_name)
        if not config:
            logger.error(f"No config found for {config_name}, cannot save last used settings")
            return
            
        last_used_path = os.path.join(
            self._user_config_dir,
            f"last_used_{config_name}_config.json"
        )
        
        try:
            with open(last_used_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            # logger.debug(f"Saved last used config to {last_used_path}")
        except Exception as e:
            logger.critical(f"Error saving last used config for {config_name}: {str(e)}")


    def _create_dataclass_instance(self, cls: Type[T], data: dict) -> T:
        """
        Create a dataclass instance from a dictionary, ensuring nested structures
        are also converted to dataclasses.
        """
        if not data:
            raise ValueError(f"No data provided to create {cls.__name__} instance")

        type_hints = get_type_hints(cls)
        processed_data = {}
        
        for field_name, field_type in type_hints.items():
            if field_name not in data:
                logger.warning(f"Field {field_name} not found in data for {cls.__name__}")
                continue
                
            field_value = data[field_name]
            
            # Handle None values
            if field_value is None:
                processed_data[field_name] = None
                continue
            
            # Extract actual type from Optional
            if str(field_type).startswith('typing.Optional'):
                field_type = field_type.__args__[0]
            
            # Handle nested dataclasses
            if hasattr(field_type, '__dataclass_fields__'):
                if isinstance(field_value, dict):
                    processed_data[field_name] = self._create_dataclass_instance(
                        field_type, field_value
                    )
                else:
                    raise ValueError(
                        f"Expected dict for {field_name}, got {type(field_value)}"
                    )
                    
            # Handle List types
            elif str(field_type).startswith('typing.List'):
                if not isinstance(field_value, list):
                    raise ValueError(
                        f"Expected list for {field_name}, got {type(field_value)}"
                    )
                # Get the type of list elements
                element_type = field_type.__args__[0]
                if hasattr(element_type, '__dataclass_fields__'):
                    # Convert each list element to a dataclass if needed
                    processed_data[field_name] = [
                        self._create_dataclass_instance(element_type, item)
                        if isinstance(item, dict) else item
                        for item in field_value
                    ]
                else:
                    processed_data[field_name] = field_value
                    
            # Handle Dict types
            elif str(field_type).startswith('typing.Dict'):
                if not isinstance(field_value, dict):
                    raise ValueError(
                        f"Expected dict for {field_name}, got {type(field_value)}"
                    )
                # Get the value type of the dict
                value_type = field_type.__args__[1]
                if hasattr(value_type, '__dataclass_fields__'):
                    # Convert each dict value to a dataclass if needed
                    processed_data[field_name] = {
                        k: self._create_dataclass_instance(value_type, v)
                        if isinstance(v, dict) else v
                        for k, v in field_value.items()
                    }
                else:
                    processed_data[field_name] = field_value
            
            # Handle basic types
            else:
                processed_data[field_name] = field_value
        
        return cls(**processed_data)

    def update_config(self, config_name: str, updates: dict) -> None:
        """
        Update config while maintaining dataclass structure throughout.
        """
        current_config = self._configs.get(config_name)
        if not current_config:
            logger.error(f"No current {config_name} config found")
            return

        def update_recursively(target, source):
            for key, value in source.items():
                if not hasattr(target, key):
                    logger.error(f"Field '{key}' not found in config")
                    continue
                    
                current_value = getattr(target, key)
                
                if isinstance(value, dict):
                    if hasattr(current_value, '__dataclass_fields__'):
                        # If current value is a dataclass, update it recursively
                        update_recursively(current_value, value)
                    else:
                        # If we're updating a dict field
                        setattr(target, key, value)
                else:
                    # Update the value directly
                    setattr(target, key, value)

        # Perform the update
        update_recursively(current_config, updates)
        # logger.debug(f"Updated {config_name} config")

    def get_config(self, config_name: str, config_class: Type[T]) -> T:
        """
        Get config, ensuring it's always returned as a proper dataclass instance.
        """
        # If not in cache, load the default config:
        if config_name not in self._configs:
            # Load default config first
            default_config = self._load_json_config(config_name, last_used=False)
            
            # Only try to load and merge last used config if it exists
            last_used_path = self.find_file(
                f"last_used_{config_name}_config.json",
                user_config=True
            )
            if last_used_path and os.path.exists(last_used_path):
                try:
                    last_used_data = self._load_json_config(config_name, last_used=True)
                    self._deep_merge_dict(default_config, last_used_data)
                    # logger.debug(f"using config from {last_used_path}")
                except (FileNotFoundError, json.JSONDecodeError):
                    logger.debug(f"No valid last used config found for {config_name}")
                
            # Create dataclass instance
            self._configs[config_name] = self._create_dataclass_instance(
                config_class, # This is either SpexConfig or ChecksConfig from setup_config.py
                default_config
            )
            
        return self._configs[config_name]

    def _deep_merge_dict(self, target: dict, source: dict) -> None:
        """
        Recursively merge source dict into target dict while preserving types.
        """
        for key, value in source.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    # Special handling for fn_sections - replace entirely
                    if key == 'fn_sections' or key == 'filename_profiles':
                        target[key] = value.copy()
                    else:
                        self._deep_merge_dict(target[key], value)
                else:
                    target[key] = value

    def save_config(self, config_name: str) -> None:
        """Save current config state to user config directory"""
        config = self._configs.get(config_name)
        if not config:
            return
            
        json_path = os.path.join(self._user_config_dir, f"{config_name}_config.json")
        
        try:
            with open(json_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            # logger.debug(f"Saved config to {json_path}")
        except Exception as e:
            logger.critical(f"Error saving config for {config_name}: {str(e)}")

    def set_config(self, config_name: str, config: Any) -> None:
        """
        Set config value, ensuring it maintains proper dataclass structure.
        
        Args:
            config_name: Name of the config to set
            config: Configuration value to set. Can be either a dataclass instance
                or a dictionary that can be converted to the appropriate dataclass.
        """
        # If it's already a dataclass instance, store it directly
        if hasattr(config, '__dataclass_fields__'):
            self._configs[config_name] = config
            return
            
        # If it's a dict, try to convert it to the appropriate dataclass
        if isinstance(config, dict):
            # Get the appropriate dataclass type from existing config
            if config_name in self._configs:
                config_class = self._configs[config_name].__class__
                self._configs[config_name] = self._create_dataclass_instance(config_class, config)
            else:
                logger.error(f"Cannot determine dataclass type for {config_name}")
                return
        else:
            logger.error(f"Config must be either a dataclass instance or a dictionary, got {type(config)}")
            return
        
        # Save the updated config
        self.save_last_used_config(config_name)

    def reset_config(self, config_name: str, config_class: Type[T]) -> T:
        """
        Reset config to default values by removing the last used config 
        and reloading from bundled defaults.
        
        Args:
            config_name: Name of the config to reset
            config_class: Type of the config class
            
        Returns:
            T: The reset config instance
        """
        # Remove last used config if it exists
        last_used_path = os.path.join(
            self._user_config_dir,
            f"last_used_{config_name}_config.json"
        )
        if os.path.exists(last_used_path):
            try:
                os.remove(last_used_path)
            except Exception as e:
                logger.error(f"Failed to remove last used config file: {str(e)}")
        
        # Clear the cached config
        if config_name in self._configs:
            del self._configs[config_name]
        
        # Load default config from bundled configs
        default_config = self._load_json_config(config_name, last_used=False)
        
        # Create and cache dataclass instance
        self._configs[config_name] = self._create_dataclass_instance(
            config_class, default_config
        )
        
        logger.info(f"Reset {config_name} config to default values")
        return self._configs[config_name]