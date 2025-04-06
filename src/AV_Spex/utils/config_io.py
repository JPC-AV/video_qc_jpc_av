from dataclasses import asdict
import json
import os
from typing import Optional, Union, List
from datetime import datetime
from ..utils.config_setup import SpexConfig, ChecksConfig
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
            else:
                continue
            export_data[config_type] = asdict(config)
        
        return export_data

    def save_configs(self, filename: Optional[str] = None, config_types: Optional[List[str]] = None) -> str:
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
        """Import configs from JSON file"""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        if 'spex' in config_data:
            # Use ConfigManager's _create_dataclass_instance to properly handle nested dataclasses
            spex_config = self.config_mgr._create_dataclass_instance(SpexConfig, config_data['spex'])
            self.config_mgr.set_config('spex', spex_config)
            self.config_mgr.save_last_used_config('spex')
        
        if 'checks' in config_data:
            checks_config = self.config_mgr._create_dataclass_instance(ChecksConfig, config_data['checks'])
            self.config_mgr.set_config('checks', checks_config)
            self.config_mgr.save_last_used_config('checks')

def handle_config_io(args, config_mgr: ConfigManager):
    """Handle config I/O operations based on arguments"""
    config_io = ConfigIO(config_mgr)
    
    if args.export_config:
        config_types = ['spex', 'checks'] if args.export_config == 'all' else [args.export_config]
        filename = config_io.save_configs(args.export_file, config_types)
        print(f"Configs exported to: {filename}")
    
    if args.import_config:
        config_io.import_configs(args.import_config)
        print(f"Configs imported from: {args.import_config}")
