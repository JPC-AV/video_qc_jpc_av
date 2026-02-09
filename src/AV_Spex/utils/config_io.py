from dataclasses import asdict
import json
import os
from typing import Optional, Union, List, Tuple
from datetime import datetime

from AV_Spex.utils.log_setup import logger

from AV_Spex.utils.config_setup import (
    SpexConfig, ChecksConfig, FilenameConfig, SignalflowConfig,
    ChecksProfilesConfig, ChecksProfile, ExiftoolConfig
)
from AV_Spex.utils.config_manager import ConfigManager

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
            elif config_type == 'profiles_checks':
                config = self.config_mgr.get_config('profiles_checks', ChecksProfilesConfig, use_last_used=True)
            elif config_type == 'exiftool':
                config = self.config_mgr.get_config('exiftool', ExiftoolConfig)
            else:
                continue
            export_data[config_type] = asdict(config)
        
        return export_data

    def export_single_profile(self, profile_type: str, profile_name: str) -> Optional[dict]:
        """
        Export a single custom profile by name.
        
        Wraps the profile in the same structure used by full config export
        so that import_configs can handle it seamlessly.
        
        Args:
            profile_type: The type of profile to export.
                          Currently supports 'profiles_checks' and 'exiftool'.
            profile_name: Name of the specific profile to export
            
        Returns:
            dict ready for JSON serialization, or None if profile not found
        """
        if profile_type == 'profiles_checks':
            try:
                profiles_config = self.config_mgr.get_config(
                    'profiles_checks', ChecksProfilesConfig, use_last_used=True
                )
            except Exception as e:
                logger.error(f"Could not load checks profiles config: {e}")
                return None
            
            if profile_name not in profiles_config.custom_profiles:
                logger.warning(f"Checks profile '{profile_name}' not found for export")
                return None
            
            profile = profiles_config.custom_profiles[profile_name]
            return {
                'profiles_checks': {
                    'custom_profiles': {
                        profile_name: asdict(profile)
                    }
                }
            }
        
        elif profile_type == 'exiftool':
            try:
                exiftool_config = self.config_mgr.get_config('exiftool', ExiftoolConfig)
            except Exception as e:
                logger.error(f"Could not load exiftool config: {e}")
                return None
            
            if profile_name not in exiftool_config.exiftool_profiles:
                logger.warning(f"Exiftool profile '{profile_name}' not found for export")
                return None
            
            profile = exiftool_config.exiftool_profiles[profile_name]
            profile_data = asdict(profile) if hasattr(profile, '__dataclass_fields__') else profile
            return {
                'exiftool': {
                    'exiftool_profiles': {
                        profile_name: profile_data
                    }
                }
            }
        
        else:
            logger.warning(f"Unknown profile type for export: {profile_type}")
            return None

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

    def save_single_profile(self, profile_type: str, profile_name: str, 
                            filename: Optional[str] = None) -> Optional[str]:
        """
        Export a single profile to a JSON file.
        
        Args:
            profile_type: 'profiles_checks' or 'exiftool'
            profile_name: Name of the profile to export
            filename: Output file path (auto-generated if not provided)
            
        Returns:
            Path to the saved file, or None if profile not found
        """
        export_data = self.export_single_profile(profile_type, profile_name)
        if not export_data:
            return None
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_name = profile_name.replace(' ', '_').replace('/', '_')
            filename = f'av_spex_profile_{safe_name}_{timestamp}.json'
        
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported profile '{profile_name}' to: {filename}")
        return filename

    def _resolve_profile_name_collision(self, desired_name: str, 
                                         existing_names: set) -> Tuple[str, bool]:
        """
        Resolve a name collision by appending '(imported)' suffix.
        
        If the suffixed name also collides, appends a numeric counter.
        
        Args:
            desired_name: The original profile name from the import file
            existing_names: Set of profile names already present locally
            
        Returns:
            Tuple of (resolved_name, was_renamed)
        """
        if desired_name not in existing_names:
            return desired_name, False
        
        # Try with (imported) suffix
        candidate = f"{desired_name} (imported)"
        if candidate not in existing_names:
            return candidate, True
        
        # Fallback: add numeric counter
        counter = 2
        while f"{desired_name} (imported {counter})" in existing_names:
            counter += 1
        return f"{desired_name} (imported {counter})", True

    def import_configs(self, config_file: str) -> dict:
        """
        Import configs from JSON file.
        
        Loads configs from the provided JSON file and updates the cached configs,
        ensuring the changes are properly saved to disk and reflected in any
        consuming components.
        
        For profile configs (profiles_checks, exiftool), imported profiles are
        *merged* alongside existing profiles rather than replacing them.
        Name collisions are resolved by appending an '(imported)' suffix.
        
        Returns:
            dict summarizing the import results. Keys may include:
                'renamed_profiles': list of (original_name, new_name) tuples
                    for any profiles that were renamed due to collisions
        """
        # Open and parse the config file
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        import_results = {
            'renamed_profiles': []
        }
        
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

        # Process custom checks profiles if present (merge alongside existing)
        if 'profiles_checks' in config_data and 'custom_profiles' in config_data['profiles_checks']:
            try:
                renamed = self._import_checks_profiles(config_data['profiles_checks']['custom_profiles'])
                import_results['renamed_profiles'].extend(renamed)
            except Exception as e:
                logger.error(f"Error importing checks profiles: {str(e)}")

        # Process exiftool profiles if present (merge alongside existing)
        if 'exiftool' in config_data and 'exiftool_profiles' in config_data['exiftool']:
            try:
                renamed = self._import_exiftool_profiles(config_data['exiftool']['exiftool_profiles'])
                import_results['renamed_profiles'].extend(renamed)
            except Exception as e:
                logger.error(f"Error importing exiftool profiles: {str(e)}")

        return import_results

    def _import_checks_profiles(self, incoming_profiles: dict) -> List[Tuple[str, str]]:
        """
        Merge incoming checks profiles alongside existing ones.
        
        Args:
            incoming_profiles: Dict of profile_name -> profile_data from the import file
            
        Returns:
            List of (original_name, renamed_name) tuples for any collisions
        """
        # Load current profiles (or create empty config if none exists yet)
        try:
            profiles_config = self.config_mgr.get_config(
                'profiles_checks', ChecksProfilesConfig, use_last_used=True
            )
        except Exception:
            # No profiles config exists yet — create a default empty one
            # and place it in the cache so replace_config_section can operate on it
            logger.info("No existing checks profiles config found, creating empty config for import")
            profiles_config = ChecksProfilesConfig()
            self.config_mgr._configs['profiles_checks'] = profiles_config
        
        existing_names = set(profiles_config.custom_profiles.keys())
        
        # Build the merged profiles dict starting from existing profiles
        merged_profiles = {}
        for name, profile in profiles_config.custom_profiles.items():
            merged_profiles[name] = asdict(profile) if hasattr(profile, '__dataclass_fields__') else profile
        
        renamed = []
        
        for original_name, profile_data in incoming_profiles.items():
            # Resolve any name collision
            final_name, was_renamed = self._resolve_profile_name_collision(
                original_name, existing_names
            )
            
            if was_renamed:
                renamed.append((original_name, final_name))
                logger.info(
                    f"Checks profile '{original_name}' renamed to '{final_name}' "
                    f"to avoid collision with existing profile"
                )
            
            # Update the profile's internal name field to match
            if isinstance(profile_data, dict) and 'name' in profile_data:
                profile_data['name'] = final_name
            
            merged_profiles[final_name] = profile_data
            # Track the new name so subsequent imports in the same batch
            # don't collide with each other
            existing_names.add(final_name)
        
        # Write the merged set back
        self.config_mgr.replace_config_section('profiles_checks', 'custom_profiles', merged_profiles)
        logger.info(
            f"Imported {len(incoming_profiles)} checks profile(s) "
            f"({len(renamed)} renamed due to collisions)"
        )
        
        return renamed

    def _import_exiftool_profiles(self, incoming_profiles: dict) -> List[Tuple[str, str]]:
        """
        Merge incoming exiftool profiles alongside existing ones.
        
        Args:
            incoming_profiles: Dict of profile_name -> profile_data from the import file
            
        Returns:
            List of (original_name, renamed_name) tuples for any collisions
        """
        # Load current profiles (or create empty config if none exists yet)
        try:
            exiftool_config = self.config_mgr.get_config('exiftool', ExiftoolConfig)
        except Exception:
            # No exiftool config exists yet — create a default empty one
            # and place it in the cache so replace_config_section can operate on it
            logger.info("No existing exiftool config found, creating empty config for import")
            exiftool_config = ExiftoolConfig()
            self.config_mgr._configs['exiftool'] = exiftool_config
        
        existing_names = set(exiftool_config.exiftool_profiles.keys())
        
        # Build the merged profiles dict starting from existing profiles
        merged_profiles = {}
        for name, profile in exiftool_config.exiftool_profiles.items():
            merged_profiles[name] = asdict(profile) if hasattr(profile, '__dataclass_fields__') else profile
        
        renamed = []
        
        for original_name, profile_data in incoming_profiles.items():
            # Resolve any name collision
            final_name, was_renamed = self._resolve_profile_name_collision(
                original_name, existing_names
            )
            
            if was_renamed:
                renamed.append((original_name, final_name))
                logger.info(
                    f"Exiftool profile '{original_name}' renamed to '{final_name}' "
                    f"to avoid collision with existing profile"
                )
            
            merged_profiles[final_name] = profile_data
            existing_names.add(final_name)
        
        # Write the merged set back
        self.config_mgr.replace_config_section('exiftool', 'exiftool_profiles', merged_profiles)
        logger.info(
            f"Imported {len(incoming_profiles)} exiftool profile(s) "
            f"({len(renamed)} renamed due to collisions)"
        )
        
        return renamed


def handle_config_io(args, config_mgr: ConfigManager):
    """Handle config I/O operations based on arguments"""
    config_io = ConfigIO(config_mgr)
    
    if args.export_config:
        if args.export_config == 'all':
            config_types = ['spex', 'checks', 'profiles_checks', 'exiftool']
        else:
            config_types = [args.export_config]
        filename = config_io.save_config_files(args.export_file, config_types)
        logger.debug(f"Configs exported to: {filename}")
    
    # Export a single named profile (e.g. --export-profile checks:MyProfile)
    if hasattr(args, 'export_profile') and args.export_profile:
        try:
            profile_type, profile_name = args.export_profile.split(':', 1)
            # Map short names to config keys
            type_map = {
                'checks': 'profiles_checks',
                'profiles_checks': 'profiles_checks',
                'exiftool': 'exiftool'
            }
            resolved_type = type_map.get(profile_type)
            if not resolved_type:
                logger.error(
                    f"Unknown profile type '{profile_type}'. "
                    f"Use 'checks' or 'exiftool'."
                )
            else:
                out_file = getattr(args, 'export_file', None)
                result = config_io.save_single_profile(resolved_type, profile_name, out_file)
                if result:
                    logger.debug(f"Profile exported to: {result}")
                else:
                    logger.error(f"Profile '{profile_name}' not found in {profile_type}")
        except ValueError:
            logger.error(
                "Invalid --export-profile format. "
                "Use 'type:name' (e.g. 'checks:My Profile')"
            )
    
    if args.import_config:
        import_results = config_io.import_configs(args.import_config)
        logger.debug(f"Configs imported from: {args.import_config}")
        
        # Log any renames that occurred
        if import_results.get('renamed_profiles'):
            for original, renamed in import_results['renamed_profiles']:
                logger.info(f"Profile '{original}' was renamed to '{renamed}' to avoid name collision")