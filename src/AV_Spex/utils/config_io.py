from dataclasses import asdict
import json
import os
from typing import Optional, Union, List
from datetime import datetime

from ..utils.log_setup import logger

from ..utils.config_setup import SpexConfig, ChecksConfig, FilenameConfig, SignalflowConfig
from ..utils.config_manager import ConfigManager

class ConfigIO:
    def __init__(self, config_mgr: ConfigManager):
        self.config_mgr = config_mgr

    def export_configs(self, config_types: Optional[List[str]] = None) -> dict:
        """Export specified configs or all configs if none specified"""
        if isinstance(config_types, str):
            config_types = [config_types]
        elif not config_types:
            config_types = ['spex', 'checks']
        
        export_data = {}
        for config_type in config_types:
            if config_type == 'spex':
                config = self.config_mgr.get_config('spex', SpexConfig)
            elif config_type == 'checks':
                config = self.config_mgr.get_config('checks', ChecksConfig)
            elif config_type == 'filename':
                config = self.config_mgr.get_config('filename', FilenameConfig)
            elif config_type == 'signalflow':
                config = self.config_mgr.get_config('signalflow', SignalflowConfig)
            else:
                continue
            export_data[config_type] = asdict(config)
        
        return export_data

    def save_config_files(self, filename: Optional[str] = None, config_types: Optional[List[str]] = None) -> str:
        """Save configs to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'av_spex_config_export_{timestamp}.json'
        
        export_data = self.export_configs(config_types)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename

    def import_configs(self, config_file: str) -> None:
        """
        Import configs from JSON file.
        
        Loads configs from the provided JSON file and updates the cached configs,
        ensuring the changes are properly saved to disk and reflected in any
        consuming components.
        """
        # Open and parse the config file
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Process spex config if present in the imported data
        if 'spex' in config_data:
            # Create dataclass instance with proper nested structure handling
            spex_config = self.config_mgr._deserialize_dataclass(SpexConfig, config_data['spex'])
            
            # Update the config manager's cached config
            self.config_mgr._configs['spex'] = spex_config
            
            # Save to disk as last_used config
            self.config_mgr.save_config('spex', is_last_used=True)
        
        # Process checks config if present in the imported data
        if 'checks' in config_data:
            # Create dataclass instance with proper nested structure handling
            checks_config = self.config_mgr._deserialize_dataclass(ChecksConfig, config_data['checks'])
            
            # Update the config manager's cached config
            self.config_mgr._configs['checks'] = checks_config
            
            # Save to disk as last_used config
            self.config_mgr.save_config('checks', is_last_used=True)

        # Process filename config if present
        if 'filename' in config_data and 'filename_profiles' in config_data['filename']:
            try:
                # Prepare filename profiles for replacement
                imported_profiles = {}
                
                # Process each profile in the imported data
                for profile_name, profile_data in config_data['filename']['filename_profiles'].items():
                    # Create filename sections
                    fn_sections = {}
                    for section_key, section_data in profile_data['fn_sections'].items():
                        fn_sections[section_key] = {
                            'value': section_data['value'],
                            'section_type': section_data['section_type']
                        }
                    
                    # Create the profile structure
                    imported_profiles[profile_name] = {
                        'fn_sections': fn_sections,
                        'FileExtension': profile_data['FileExtension']
                    }
                
                # Replace the entire filename_profiles section
                self.config_mgr.replace_config_section('filename', 'filename_profiles', imported_profiles)
                logger.info("Imported and updated filename configuration")
            except Exception as e:
                logger.error(f"Error importing filename config: {str(e)}")
        
        # Process signalflow config if present
        if 'signalflow' in config_data and 'signalflow_profiles' in config_data['signalflow']:
            try:
                # Prepare signalflow profiles for replacement
                imported_profiles = {}
                
                # Process each profile in the imported data
                for profile_name, profile_data in config_data['signalflow']['signalflow_profiles'].items():
                    # Create the profile structure with correct format
                    imported_profiles[profile_name] = {
                        'name': profile_data.get('name', profile_name),
                        'Source_VTR': profile_data.get('Source_VTR', []),
                        'TBC_Framesync': profile_data.get('TBC_Framesync', []),
                        'ADC': profile_data.get('ADC', []),
                        'Capture_Device': profile_data.get('Capture_Device', []),
                        'Computer': profile_data.get('Computer', [])
                    }
                
                # Replace the entire signalflow_profiles section
                self.config_mgr.replace_config_section('signalflow', 'signalflow_profiles', imported_profiles)
                logger.info("Imported and updated signalflow configuration")
            except Exception as e:
                logger.error(f"Error importing signalflow config: {str(e)}")


def handle_config_io(args, config_mgr: ConfigManager):
    """Handle config I/O operations based on arguments"""
    config_io = ConfigIO(config_mgr)
    
    if args.export_config:
        config_types = ['spex', 'checks'] if args.export_config == 'all' else [args.export_config]
        filename = config_io.save_config_files(args.export_file, config_types)
        logger.debug(f"Configs exported to: {filename}")
    
    if args.import_config:
        config_io.import_configs(args.import_config)
        logger.debug(f"Configs imported from: {args.import_config}")
