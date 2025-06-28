from dataclasses import asdict
from typing import List, Dict, Union, Optional

import json
import os

from AV_Spex.utils.log_setup import logger
from AV_Spex.utils.config_setup import ChecksConfig, SpexConfig, FilenameProfile, FilenameValues, FilenameSection, SignalflowConfig, ChecksProfile, ChecksProfilesConfig
from AV_Spex.utils.config_manager import ConfigManager


config_mgr = ConfigManager() # Gets the singleton instance


def format_config_value(value, indent=0, is_nested=False):
    """Format config values for display."""
    spacer = " " * indent
    
    if isinstance(value, dict):
        formatted_str = "\n" if is_nested else ""
        for k, v in value.items():
            formatted_str += f"{spacer}{k}: {format_config_value(v, indent + 2, True)}\n"
        return formatted_str
    
    if isinstance(value, list):
        return ', '.join(str(item) for item in value)
        
    if value == 'yes': return "✅"
    if value == 'no': return "❌"
    return str(value)


def print_config(config_spec='all'):
    """
    Print config state for specified config type(s) and optional subsections.
    
    Args:
        config_spec (str): Specification of what to print. Can be:
            - 'all': Print all configs
            - 'checks' or 'spex': Print entire specified config
            - 'checks,tools' or 'spex,filename_values': Print specific subsection
    """
    if not validate_config_spec(config_spec):
        logger.error(f"Invalid config specification: {config_spec}.")
        logger.error(f"Format should be 'config[,subsection]' where config is one of: all, spex, checks - subsection (optional) is a valid section of the specified config\n")
    
    configs = {}
    
    # Parse the config specification
    parts = [p.strip() for p in config_spec.split(',')]
    config_type = parts[0]
    subsection = parts[1] if len(parts) > 1 else None
    
    # Load the requested config(s)
    if config_type in ['all', 'checks']:
        configs['Checks Config'] = config_mgr.get_config('checks', ChecksConfig)
    if config_type in ['all', 'spex']:
        configs['Spex Config'] = config_mgr.get_config('spex', SpexConfig)
    
    # Print the configs
    for config_name, config in configs.items():
        print(f"\n{config_name}:")
        config_dict = asdict(config)
        
        if subsection:
            # Print only the specified subsection if it exists
            if subsection in config_dict:
                print(f"{subsection}:")
                print(format_config_value(config_dict[subsection], indent=2))
            else:
                print(f"Subsection '{subsection}' not found in {config_name}")
        else:
            # Print entire config
            for key, value in config_dict.items():
                print(f"{key}:")
                print(format_config_value(value, indent=2))


def validate_config_spec(config_spec: str) -> bool:
    """
    Validate the config specification format.
    
    Args:
        config_spec: String specification of config to print
        
    Returns:
        bool: True if valid, False if invalid
    """
    if not config_spec:
        return False
        
    parts = [p.strip() for p in config_spec.split(',')]
    
    # Check base config type
    if parts[0] not in ['all', 'spex', 'checks']:
        return False
        
    # If subsection specified, validate against known subsections
    if len(parts) > 1:
        config_type = parts[0]
        subsection = parts[1]
        
        valid_subsections = {
            'spex': ['filename_values', 'mediainfo_values', 'exiftool_values', 
                    'ffmpeg_values', 'mediatrace_values', 'qct_parse_values'],
            'checks': ['outputs', 'fixity', 'tools']
        }
        
        # Only check subsection validity for specific configs (not 'all')
        if config_type != 'all':
            return subsection in valid_subsections[config_type]
            
    return True


def resolve_config(args, config_mapping):
    return config_mapping.get(args, None)


def apply_filename_profile(selected_profile: FilenameProfile):
    """
    Apply a FilenameProfile dataclass to the current configuration.
    
    Completely replaces the existing filename configuration with the selected profile,
    ensuring all sections are properly saved and persisted.
    """
    # Debug information about the provided profile
    logger.debug(f"==== APPLYING FILENAME PROFILE ====")
    logger.debug(f"Profile has {len(selected_profile.fn_sections)} sections")
    for idx, (key, section) in enumerate(sorted(selected_profile.fn_sections.items()), 1):
        logger.debug(f"  Section {idx}: {key} = {section.value} ({section.section_type})")
    
    # Create new sections dictionary by copying from the selected profile
    new_sections = {}
    for section_key, section_value in selected_profile.fn_sections.items():
        new_sections[section_key] = {
            'value': section_value.value,
            'section_type': section_value.section_type
        }
    
    # Replace the entire fn_sections dictionary
    config_mgr.replace_config_section('spex', 'filename_values.fn_sections', new_sections)
    
    # Replace the FileExtension
    config_mgr.replace_config_section('spex', 'filename_values.FileExtension', selected_profile.FileExtension)
    
    # Force a refresh to ensure changes are persisted
    config_mgr.refresh_configs()
    
    # Verify changes persisted
    final_config = config_mgr.get_config('spex', SpexConfig)
    logger.debug(f"Final verification after refresh: Config has {len(final_config.filename_values.fn_sections)} sections")


def get_signalflow_profile(profile_name: str):
    """
    Get a signalflow profile by name from the configuration.
    
    Args:
        profile_name (str): The name of the profile to retrieve
        
    Returns:
        SignalflowProfile or None: The requested profile or None if not found
    """
    config_mgr = ConfigManager()
    signalflow_config = config_mgr.get_config('signalflow', SignalflowConfig)
    
    if profile_name in signalflow_config.signalflow_profiles:
        return signalflow_config.signalflow_profiles[profile_name]
    
    return None

def apply_signalflow_profile(selected_profile):
    """
    Apply a SignalflowProfile dataclass or dict to the current configuration.
    
    Completely replaces the existing signalflow configuration with the selected profile,
    ensuring all sections are properly saved and persisted.
    
    Args:
        selected_profile (SignalflowProfile or dict): The signalflow profile to apply
    """
    # Debug information about the provided profile
    logger.debug(f"==== APPLYING SIGNALFLOW PROFILE ====")
    
    # Convert dict to proper structure if needed
    encoder_settings = {}
    
    if isinstance(selected_profile, dict):
        # If given a direct dict (from the hardcoded profiles or custom UI)
        if "name" in selected_profile:
            # This is already in the right format from the JSON config
            for key in ["Source_VTR", "TBC_Framesync", "ADC", "Capture_Device", "Computer"]:
                if key in selected_profile:
                    encoder_settings[key] = selected_profile[key]
        else:
            # This is from the old hardcoded dict format, just use as is
            encoder_settings = selected_profile
    else:
        # If given a SignalflowProfile dataclass, convert to dict
        encoder_settings = {
            "Source_VTR": selected_profile.Source_VTR,
            "TBC_Framesync": selected_profile.TBC_Framesync,
            "ADC": selected_profile.ADC,
            "Capture_Device": selected_profile.Capture_Device,
            "Computer": selected_profile.Computer
        }
    
    # Debug the settings we're going to apply
    for idx, (key, value) in enumerate(sorted(encoder_settings.items()), 1):
        logger.debug(f"  Setting {idx}: {key} = {value}")
    
    # Get the current spex config to check structure
    config_mgr = ConfigManager()
    spex_config = config_mgr.get_config('spex', SpexConfig)
    
    # Update mediatrace_values.ENCODER_SETTINGS
    # Create a new ENCODER_SETTINGS object with the profile values
    current_encoder_settings = {}
    if hasattr(spex_config.mediatrace_values.ENCODER_SETTINGS, '__dict__'):
        # Get existing attributes that aren't in the selected profile
        for key, value in spex_config.mediatrace_values.ENCODER_SETTINGS.__dict__.items():
            if not key.startswith('_') and key not in encoder_settings:
                current_encoder_settings[key] = value
    
    # Add all settings from the profile
    for key, value in encoder_settings.items():
        current_encoder_settings[key] = value
    
    # Replace the entire ENCODER_SETTINGS object
    config_mgr.replace_config_section('spex', 'mediatrace_values.ENCODER_SETTINGS', current_encoder_settings)
    
    # Check if ffmpeg_values.format.tags exists and update it
    if (hasattr(spex_config, 'ffmpeg_values') and 
        'format' in spex_config.ffmpeg_values and 
        'tags' in spex_config.ffmpeg_values['format']):
        
        # Get current ENCODER_SETTINGS dict or create a new one
        current_ffmpeg_settings = {}
        if ('ENCODER_SETTINGS' in spex_config.ffmpeg_values['format']['tags'] and 
            spex_config.ffmpeg_values['format']['tags']['ENCODER_SETTINGS'] is not None):
            # Copy existing settings that aren't in the selected profile
            for key, value in spex_config.ffmpeg_values['format']['tags']['ENCODER_SETTINGS'].items():
                if key not in encoder_settings:
                    current_ffmpeg_settings[key] = value
        
        # Add all settings from the profile
        for key, value in encoder_settings.items():
            current_ffmpeg_settings[key] = value
        
        # Replace the entire ENCODER_SETTINGS dictionary
        config_mgr.replace_config_section('spex', 'ffmpeg_values.format.tags.ENCODER_SETTINGS', current_ffmpeg_settings)
    
    # Force a refresh to ensure changes are persisted
    config_mgr.refresh_configs()
    
    # Verify changes persisted
    final_config = config_mgr.get_config('spex', SpexConfig)
    logger.debug(f"Final verification after refresh: Confirming encoder settings persisted")
    
    # Detailed verification
    final_mediatrace_keys = []
    if hasattr(final_config.mediatrace_values.ENCODER_SETTINGS, '__dict__'):
        final_mediatrace_keys = list(final_config.mediatrace_values.ENCODER_SETTINGS.__dict__.keys())
    
    final_ffmpeg_keys = []
    if (hasattr(final_config, 'ffmpeg_values') and 
        'format' in final_config.ffmpeg_values and 
        'tags' in final_config.ffmpeg_values['format'] and
        'ENCODER_SETTINGS' in final_config.ffmpeg_values['format']['tags']):
        final_ffmpeg_keys = list(final_config.ffmpeg_values['format']['tags']['ENCODER_SETTINGS'].keys())
    
    logger.debug(f"Mediatrace encoder settings keys: {final_mediatrace_keys}")
    logger.debug(f"FFmpeg encoder settings keys: {final_ffmpeg_keys}")


def apply_profile(selected_profile):
    """Apply profile changes to checks_config.
    
    Args:
        selected_profile (dict): The profile configuration to apply
    """
    checks_config = config_mgr.get_config('checks', ChecksConfig)
    
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


def get_custom_profiles_config():
    """Get the custom profiles configuration."""
    try:
        # Force reload from disk by clearing cache first
        if 'profiles_checks' in config_mgr._configs:
            del config_mgr._configs['profiles_checks']
            logger.debug("Cleared profiles_checks from cache")
            
        # Check what files actually exist
        bundled_path = os.path.join(config_mgr._bundle_dir, 'config', 'profiles_checks_config.json')
        last_used_path = os.path.join(config_mgr._user_config_dir, 'last_used_profiles_checks_config.json')
        
        logger.debug(f"Checking bundled config at: {bundled_path} - exists: {os.path.exists(bundled_path)}")
        logger.debug(f"Checking last_used config at: {last_used_path} - exists: {os.path.exists(last_used_path)}")
        
        if os.path.exists(last_used_path):
            # Read and debug the last_used file directly
            with open(last_used_path, 'r') as f:
                content = f.read()
                logger.debug(f"Last used file content: {content[:200]}...")  # First 200 chars
        
        # Use last_used=True to load saved profiles, falling back to bundled config
        config = config_mgr.get_config('profiles_checks', ChecksProfilesConfig, use_last_used=True)
        logger.debug(f"Loaded custom profiles config with {len(config.custom_profiles)} profiles: {list(config.custom_profiles.keys())}")
        return config
    except FileNotFoundError:
        logger.debug("profiles_checks.json not found, creating new empty config")
        # If the file doesn't exist, create a new empty config
        empty_config = ChecksProfilesConfig()
        
        # Save it directly to user config directory as last_used only
        config_file_path = os.path.join(config_mgr._user_config_dir, 'last_used_profiles_checks_config.json')
        with open(config_file_path, 'w') as f:
            json.dump(asdict(empty_config), f, indent=2)
        
        # Set in cache
        config_mgr._configs['profiles_checks'] = empty_config
        logger.debug(f"Created empty profiles config at: {config_file_path}")
        return empty_config
    

def get_available_custom_profiles() -> List[str]:
    """Get list of available custom profile names."""
    profiles_config = get_custom_profiles_config()
    return list(profiles_config.custom_profiles.keys())

def get_custom_profile(profile_name: str) -> Optional[ChecksProfile]:
    """Get a specific custom profile by name."""
    profiles_config = get_custom_profiles_config()
    return profiles_config.custom_profiles.get(profile_name)


def save_custom_profile(profile: ChecksProfile):
    """Save a custom profile using ConfigManager's replace_config_section method."""
    logger.debug(f"=== SAVING CUSTOM PROFILE ===")
    logger.debug(f"Profile name: {profile.name}")
    
    try:
        # Get current profiles
        profiles_config = get_custom_profiles_config()
        logger.debug(f"Current profiles before save: {list(profiles_config.custom_profiles.keys())}")
        
        # Create updated profiles dict with the new profile
        updated_profiles = {}
        
        # Add existing profiles
        for name, existing_profile in profiles_config.custom_profiles.items():
            updated_profiles[name] = asdict(existing_profile)
        
        # Add the new profile
        updated_profiles[profile.name] = asdict(profile)
        
        logger.debug(f"Updated profiles dict will have: {list(updated_profiles.keys())}")
        
        # Debug: Check config before replace_config_section
        logger.debug(f"Config in cache before replace: {config_mgr._configs.get('profiles_checks', 'NOT FOUND')}")
        
        # Use replace_config_section to replace the entire custom_profiles dict
        config_mgr.replace_config_section('profiles_checks', 'custom_profiles', updated_profiles)
        
        # Debug: Check config after replace_config_section
        cached_config = config_mgr._configs.get('profiles_checks')
        if cached_config:
            logger.debug(f"Config in cache after replace: {len(cached_config.custom_profiles)} profiles")
        
        # Debug: Check if file was actually written
        last_used_path = os.path.join(config_mgr._user_config_dir, 'last_used_profiles_checks_config.json')
        if os.path.exists(last_used_path):
            with open(last_used_path, 'r') as f:
                content = f.read()
                logger.debug(f"File content after save: {content[:300]}...")  # First 300 chars
        else:
            logger.error(f"Last used file not found after save at: {last_used_path}")
        
        logger.info(f"Successfully saved custom profile: {profile.name}")
        
        # Verify the save worked
        verification_config = get_custom_profiles_config()
        if profile.name in verification_config.custom_profiles:
            logger.debug(f"Verification: Profile '{profile.name}' confirmed saved")
        else:
            logger.error(f"Verification failed: Profile '{profile.name}' not found after save")
            logger.debug(f"Available profiles after save: {list(verification_config.custom_profiles.keys())}")
        
    except Exception as e:
        logger.error(f"Error saving custom profile '{profile.name}': {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def delete_custom_profile(profile_name: str) -> bool:
    """Delete a custom profile using ConfigManager's replace_config_section method."""
    profiles_config = get_custom_profiles_config()
    if profile_name not in profiles_config.custom_profiles:
        logger.warning(f"Profile '{profile_name}' not found, cannot delete")
        return False
    
    try:
        # Create updated profiles dict without the deleted profile
        updated_profiles = {k: asdict(v) for k, v in profiles_config.custom_profiles.items() if k != profile_name}
        
        # Use replace_config_section to replace the entire custom_profiles dict
        config_mgr.replace_config_section('profiles_checks', 'custom_profiles', updated_profiles)
        logger.info(f"Deleted custom profile: {profile_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting custom profile '{profile_name}': {str(e)}")
        return False

def apply_custom_profile(profile_name: str):
    """Apply a custom profile to the current checks configuration."""
    profile = get_custom_profile(profile_name)
    if not profile:
        logger.error(f"Custom profile '{profile_name}' not found")
        return False
    
    try:
        # Convert the profile to the format expected by apply_profile
        profile_dict = {
            "outputs": asdict(profile.outputs),
            "fixity": asdict(profile.fixity),
            "tools": asdict(profile.tools)
        }
        
        apply_profile(profile_dict)
        logger.info(f"Applied custom profile: {profile_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error applying custom profile '{profile_name}': {str(e)}")
        return False

def create_profile_from_current_config(profile_name: str, description: str = "") -> ChecksProfile:
    """Create a new custom profile from the current checks configuration."""
    current_config = config_mgr.get_config('checks', ChecksConfig)
    
    # Create a new profile with the current configuration
    new_profile = ChecksProfile(
        name=profile_name,
        description=description,
        outputs=current_config.outputs,
        fixity=current_config.fixity,
        tools=current_config.tools
    )
    
    return new_profile

def get_all_profiles() -> Dict[str, Union[dict, ChecksProfile]]:
    """Get all available profiles (both built-in and custom)."""
    all_profiles = {}
    
    # Add built-in profiles
    all_profiles.update({
        "Step 1 Profile": profile_step1,
        "Step 2 Profile": profile_step2, 
        "All Off Profile": profile_allOff
    })
    
    # Add custom profiles
    custom_profiles = get_custom_profiles_config().custom_profiles
    all_profiles.update(custom_profiles)
    
    return all_profiles


profile_step1 = {
    "tools": {
        "qctools": {
            "run_tool": "no"   
        },
        "exiftool": {
            "check_tool": "yes",
            "run_tool": "yes"
        },
        "ffprobe": {
            "check_tool": "yes",
            "run_tool": "yes"
        },
        "mediaconch": {
            "mediaconch_policy": "JPC_AV_NTSC_MKV_2025_03_26.xml",
            "run_mediaconch": "yes"
        },
        "mediainfo": {
            "check_tool": "yes",
            "run_tool": "yes"
        },
        "mediatrace": {
            "check_tool": "yes",
            "run_tool": "yes"
        },
        "qctools": {
            "run_tool": "no"
        },
        "qct_parse": {
            "run_tool": "no"
        }
    },
    "outputs": {
        "access_file": "no",
        "report": "no",
        "qctools_ext": "qctools.xml.gz"
    },
    "fixity": {
        "check_fixity": "no",
        "validate_stream_fixity": "no",
        "embed_stream_fixity": "yes",
        "output_fixity": "yes",
        "overwrite_stream_fixity": "no"
    }
}

profile_step2 = {
    "tools": {
        "exiftool": {
            "check_tool": "yes",
            "run_tool": "no"
        },
        "ffprobe": {
            "check_tool": "yes",
            "run_tool": "no"
        },
        "mediaconch": {
            "mediaconch_policy": "JPC_AV_NTSC_MKV_2024-09-20.xml",
            "run_mediaconch": "yes"
        },
        "mediainfo": {
            "check_tool": "yes",
            "run_tool": "no"
        },
        "mediatrace": {
            "check_tool": "yes",
            "run_tool": "no"
        },
        "qctools": {
            "run_tool": "yes"
        },
        "qct_parse": {
            "run_tool": "yes",
            "barsDetection": True,
            "evaluateBars": True,
            "contentFilter": [],
            "profile": [],
            "tagname": None,
            "thumbExport": True
        }
    },
    "outputs": {
        "access_file": "no",
        "report": "yes",
        "qctools_ext": "qctools.xml.gz"
    },
    "fixity": {
        "check_fixity": "yes",
        "validate_stream_fixity": "yes",
        "embed_stream_fixity": "no",
        "output_fixity": "no",
        "overwrite_stream_fixity": "no"
    }
}

profile_allOff = {
    "tools": {
        "exiftool": {
            "check_tool": "no",
            "run_tool": "no"
        },
        "ffprobe": {
            "check_tool": "no",
            "run_tool": "no"
        },
        "mediaconch": {
            "mediaconch_policy": "JPC_AV_NTSC_MKV_2024-09-20.xml",
            "run_mediaconch": "no"
        },
        "mediainfo": {
            "check_tool": "no",
            "run_tool": "no"
        },
        "mediatrace": {
            "check_tool": "no",
            "run_tool": "no"
        },
        "qctools": {
            "run_tool": "no"
        },
        "qct_parse": {
            "run_tool": "no",
            "barsDetection": False,
            "evaluateBars": False,
            "contentFilter": [],
            "profile": [],
            "tagname": None,
            "thumbExport": False
        }
    },
    "outputs": {
        "access_file": "no",
        "report": "no",
        "qctools_ext": "qctools.xml.gz"
    },
    "fixity": {
        "check_fixity": "no",
        "validate_stream_fixity": "no",
        "embed_stream_fixity": "no",
        "output_fixity": "no",
        "overwrite_stream_fixity": "no"
    }
}

JPC_AV_SVHS = {
    "Source_VTR": ["SVO5800", "SN 122345", "composite", "analog balanced"], 
    "TBC_Framesync": ["DPS575 with flash firmware h2.16", "SN 15230", "SDI", "audio embedded"], 
    "ADC": ["DPS575 with flash firmware h2.16", "SN 15230", "SDI"], 
    "Capture_Device": ["Black Magic Ultra Jam", "SN B022159", "Thunderbolt"],
    "Computer": ["2023 Mac Mini", "Apple M2 Pro chip", "SN H9HDW53JMV", "OS 14.5", "vrecord v2023-08-07", "ffmpeg"]
}

BVH3100 = {
    "Source_VTR": ["Sony BVH3100", "SN 10525", "composite", "analog balanced"],
    "TBC_Framesync": ["Sony BVH3100", "SN 10525", "composite", "analog balanced"],
    "ADC": ["Leitch DPS575 with flash firmware h2.16", "SN 15230", "SDI", "embedded"],
    "Capture_Device": ["Blackmagic Design UltraStudio 4K Extreme", "SN B022159", "Thunderbolt"],
    "Computer": ["2023 Mac Mini", "Apple M2 Pro chip", "SN H9HDW53JMV", "OS 14.5", "vrecord v2023-08-07", "ffmpeg"]
}
