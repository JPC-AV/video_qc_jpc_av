from dataclasses import asdict
from typing import Optional, TypeVar, Type, List, Dict, Union, Any, get_type_hints, get_origin, get_args
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

    def _load_json_config(self, config_name: str, use_last_used: bool = False) -> dict:
        """Load JSON config from file."""
        config_path = None
        
        if use_last_used:
            last_used_path = os.path.join(self._user_config_dir, f"last_used_{config_name}_config.json")
            if os.path.exists(last_used_path):
                config_path = last_used_path
        
        if not config_path:
            # Use bundled config
            config_path = os.path.join(self._bundle_dir, 'config', f"{config_name}_config.json")
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing config file {config_path}: {str(e)}")

    # Dataclass conversion functions
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


    def save_config(self, config_name: str, is_last_used: bool = False) -> None:
        """
        Save current config to a file.
        
        Args:
            config_name: Name of the config to save
            is_last_used: If True, save as last_used config, otherwise as regular config
        """
        config = self._configs.get(config_name)
        if not config:
            # logger.error(f"No config found for {config_name}, cannot save")
            return
            
        filename = f"{'last_used_' if is_last_used else ''}{config_name}_config.json"
        save_path = os.path.join(self._user_config_dir, filename)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            logger.info(f"Saved config to {save_path}")
        except Exception as e:
            logger.error(f"Error saving config to {save_path}: {str(e)}")

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
        
    def _update_dict_recursively(self, target: dict, source: dict) -> None:
        """Update dictionary recursively."""
        for key, value in source.items():
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    self._update_dict_recursively(target[key], value)
                else:
                    target[key] = value

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
    
    def refresh_configs(self):
        """Force reload of all configs from disk"""
        self._configs = {}
        logger.debug(f"Config cache cleared, configs will be reloaded from disk\n")