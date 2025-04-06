from dataclasses import asdict
from typing import List

from ..utils.log_setup import logger
from ..utils.config_setup import ChecksConfig, SpexConfig
from ..utils.config_manager import ConfigManager


config_mgr = ConfigManager()


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


def apply_signalflow_profile(selected_profile: dict):
    """Apply signalflow profile changes to spex_config.
    
    Args:
        selected_profile (dict): The signalflow profile to apply (encoder settings)
    """
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
    
    # Save the updated config
    config_mgr.set_config('spex', spex_config)
    
    # Save the last used config
    config_mgr.save_last_used_config('spex')


def apply_profile(selected_profile):
    """Apply profile changes to checks_config.
    
    Args:
        selected_profile (dict): The profile configuration to apply
        config_mgr (ConfigManager): Instance of the config manager
    """
    checks_config = config_mgr.get_config('checks', ChecksConfig)
    
    if 'outputs' in selected_profile:
        for key, value in selected_profile["outputs"].items():
            setattr(checks_config.outputs, key, value)

    if 'tools' in selected_profile:
        for tool_name, updates in selected_profile["tools"].items():
            # Handle each tool type differently based on its structure
            if tool_name == 'mediaconch':
                # MediaConch has mediaconch_policy and run_mediaconch
                if isinstance(updates, dict):
                    for key, value in updates.items():
                        setattr(checks_config.tools.mediaconch, key, value)
            
            elif tool_name == 'qct_parse':
                # QCT Parse has a unique structure with boolean and list fields
                if isinstance(updates, dict):
                    for key, value in updates.items():
                        setattr(checks_config.tools.qct_parse, key, value)
            
            else:
                # Standard tools with check_tool and run_tool
                tool = getattr(checks_config.tools, tool_name)
                if isinstance(updates, dict):
                    if 'check_tool' in updates:
                        tool.check_tool = updates['check_tool']
                    if 'run_tool' in updates:
                        tool.run_tool = updates['run_tool']

    if 'fixity' in selected_profile:
        for key, value in selected_profile["fixity"].items():
            if hasattr(checks_config.fixity, key):
                setattr(checks_config.fixity, key, value)
    
    # Save the updated config
    config_mgr.set_config('checks', checks_config)


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

bowser_filename = {
    "Collection": "2012_79",
    "MediaType": "2",
    "ObjectID": r"\d{3}_\d{1}[a-zA-Z]",
    "DigitalGeneration": "PM",
    "FileExtension": "mkv"
}

JPCAV_filename = {
    "Collection": "JPC",
    "MediaType": "AV",
    "ObjectID": r"\d{5}",
    "DigitalGeneration": None,
    "FileExtension": "mkv"
}
