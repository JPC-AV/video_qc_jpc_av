#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import toml
from art import text2art
from dataclasses import dataclass, asdict
from typing import List, Optional, Any

from .processing import processing_mgmt
from .processing.avspex_processor import AVSpexProcessor
from .utils import dir_setup
from .utils import config_edit
from .utils.log_setup import logger
from .utils.config_setup import SpexConfig, FilenameConfig
from .utils.config_manager import ConfigManager
from .utils.config_io import ConfigIO

# Create lazy loader for GUI components
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
            from .gui.gui_checks_tab.gui_checks_tab import ChecksWindow
            from .gui.gui_main import MainWindow
            cls._QApplication = QApplication
            cls._ChecksWindow = ChecksWindow
            cls._MainWindow = MainWindow
            
    @classmethod
    def get_application(cls):
        cls.load_gui_components()
        if cls._app is None:
            cls._app = cls._QApplication(sys.argv)
            
            # Connect the aboutToQuit signal to save configs
            def save_configs_on_quit():
                print("Application about to quit - saving configs")
                from .utils.config_manager import ConfigManager
                config_mgr = ConfigManager()
                config_mgr.save_config('checks', is_last_used=True)
                config_mgr.save_config('spex', is_last_used=True)
            
            # Make sure to explicitly connect to the instance
            cls._app.aboutToQuit.connect(save_configs_on_quit)
        
        return cls._app
    
    @classmethod
    def get_main_window(cls):
        cls.load_gui_components()
        return cls._MainWindow()
    
config_mgr = ConfigManager()

@dataclass
class ParsedArguments:
    source_directories: List[str]
    selected_profile: Optional[Any]
    sn_config_changes: Optional[Any]
    fn_config_changes: Optional[Any]
    print_config_profile: Optional[str]
    dry_run_only: bool
    tools_on_names: List[str]
    tools_off_names: List[str]
    gui: Optional[Any]
    export_config: Optional[str]
    export_file: Optional[str] 
    import_config: Optional[str]
    mediaconch_policy: Optional[str]
    use_default_config: bool


PROFILE_MAPPING = {
    "step1": config_edit.profile_step1,
    "step2": config_edit.profile_step2,
    "off": config_edit.profile_allOff
}


SIGNALFLOW_MAPPING = {
    "JPC_AV_SVHS": config_edit.JPC_AV_SVHS,
    "BVH3100": config_edit.BVH3100
}


filename_config = config_mgr.get_config("filename", FilenameConfig)

FILENAME_MAPPING = {
    "jpc": filename_config.filename_profiles["JPC Filename Profile"],
    "bowser": filename_config.filename_profiles["Bowser Filename Profile"]
}


SIGNAL_FLOW_CONFIGS = {
    "JPC_AV_SVHS": {
        "format_tags": {"ENCODER_SETTINGS": config_edit.JPC_AV_SVHS},
        "mediatrace": {"ENCODER_SETTINGS": config_edit.JPC_AV_SVHS}
    },
    "BVH3100": {
        "format_tags": {"ENCODER_SETTINGS": config_edit.BVH3100}, 
        "mediatrace": {"ENCODER_SETTINGS": config_edit.BVH3100}
    }
}



def parse_arguments():
    # Get the version from __init__
    from AV_Spex import __version__
    version_string = __version__

    parser = argparse.ArgumentParser(
        description=f"""\
%(prog)s {version_string}

AV Spex is a python application designed to help process digital audio and video media created from analog sources.
The scripts will confirm that the digital files conform to predetermined specifications.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--version', action='version', version=f'%(prog)s {version_string}')
    parser.add_argument("paths", nargs='*', help="Path to the input -f: video file(s) or -d: directory(ies)")
    parser.add_argument("-dr","--dryrun", action="store_true", 
                        help="Flag to run av-spex w/out outputs or checks. Use to change config profiles w/out processing video.")
    parser.add_argument("--profile", choices=list(PROFILE_MAPPING.keys()), 
                        help="Select processing profile or turn checks off")
    parser.add_argument("--on", 
                        action='append', 
                         metavar="{tool_name.run_tool, tool_name.check_tool}",
                         help="Turns on specific tool run_ or check_ option (format: tool.check_tool or tool.run_tool, e.g. mediainfo.run_tool)")
    parser.add_argument("--off", 
                        action='append', 
                         metavar="{tool_name.run_tool, tool_name.check_tool}",
                         help="Turns off specific tool run_ or check_ option (format: tool.check_tool or tool.run_tool, e.g. mediainfo.run_tool)")
    parser.add_argument("-sn","--signalflow", choices=['JPC_AV_SVHS', 'BVH3100'],
                        help="Select signal flow config type (JPC_AV_SVHS or BVH3100)")
    parser.add_argument("-fn","--filename", choices=['jpc', 'bowser'], 
                        help="Select file name config type (jpc or bowser)")
    parser.add_argument("-pp", "--printprofile", type=str, nargs='?', const='all', default=None, 
                        help="Show config profile(s) and optional subsection. Format: 'config[,subsection]'. Examples: 'all', 'spex', 'checks', 'checks,tools', 'spex,filename_values'")
    parser.add_argument("-d","--directory", action="store_true", 
                        help="Flag to indicate input is a directory")
    parser.add_argument("-f","--file", action="store_true", 
                        help="Flag to indicate input is a video file")
    parser.add_argument('--gui', action='store_true', 
                        help='Force launch in GUI mode')
    parser.add_argument("--use-default-config", action="store_true",
                       help="Reset to default config by removing any saved configurations")
    
    # Config export/import arguments
    parser.add_argument('--export-config', 
                    choices=['all', 'spex', 'checks'],
                    help='Export current config(s) to JSON')
    parser.add_argument('--export-file',
                    help='Specify export filename (default: auto-generated)')
    parser.add_argument('--import-config',
                    help='Import configs from JSON file')
    parser.add_argument("--mediaconch-policy",
                    help="Path to custom MediaConch policy XML file")

    args = parser.parse_args()

    input_paths = args.paths if args.paths else []
    source_directories = dir_setup.validate_input_paths(input_paths, args.file)

    selected_profile = config_edit.resolve_config(args.profile, PROFILE_MAPPING)
    sn_config_changes = config_edit.resolve_config(args.signalflow, SIGNALFLOW_MAPPING)
    fn_config_changes = config_edit.resolve_config(args.filename, FILENAME_MAPPING)

    if args.use_default_config:
        try:
            # Remove user config files from user config directory
            user_config_dir = config_mgr._user_config_dir
            try:
                os.remove(os.path.join(user_config_dir, "last_used_checks_config.json"))
                os.remove(os.path.join(user_config_dir, "last_used_spex_config.json"))
                os.remove(os.path.join(user_config_dir, "last_used_filename_config.json"))
                print("Reset to default configuration")
            except FileNotFoundError:
                # It's okay if the files don't exist
                print("Already using default configuration")
        except PermissionError:
            print("Error: Unable to reset config - permission denied. Try running with administrator privileges.")
        except OSError as e:
            print(f"Error: Unable to reset config - {str(e)}")
        except Exception as e:
            print(f"Error: Unexpected error while resetting config - {str(e)}")

    return ParsedArguments(
        source_directories=source_directories,
        selected_profile=selected_profile,
        sn_config_changes=sn_config_changes,
        fn_config_changes=fn_config_changes,
        print_config_profile=args.printprofile,
        dry_run_only=args.dryrun,
        tools_on_names=args.on or [],
        tools_off_names=args.off or [],
        gui=args.gui,
        export_config=args.export_config,
        export_file=args.export_file,
        import_config=args.import_config,
        mediaconch_policy=args.mediaconch_policy,
        use_default_config=args.use_default_config
    )


def apply_filename_profile(config_type: str, profile_name: str):
    spex_config = config_mgr.get_config('spex', SpexConfig)
            
    if config_type == 'filename':
        if not isinstance(profile_name, dict):
            logger.critical(f"Invalid filename settings: {profile_name}")
            return
            
        updates = {
            "filename_values": profile_name
        }
        # Update and save config
        config_mgr.update_config('spex', updates)
        
    else:
        logger.critical(f"Invalid configuration type: {config_type}")
        return
        
    # Save the last used config
    config_mgr.save_last_used_config('spex')

def print_av_spex_logo():
    avspex_icon = text2art("A-V Spex", font='5lineoblique')
    print(f'{avspex_icon}\n')


def run_cli_mode(args):
    print_av_spex_logo()

    # Update checks config
    if args.selected_profile:
        config_edit.apply_profile(args.selected_profile)
        config_mgr.save_last_used_config('checks')
    if args.tools_on_names:
        config_edit.toggle_on(args.tools_on_names)
        config_mgr.save_last_used_config('checks')
    if args.tools_off_names:
        config_edit.toggle_off(args.tools_off_names)
        config_mgr.save_last_used_config('checks')

    if args.mediaconch_policy:
        processing_mgmt.setup_mediaconch_policy(args.mediaconch_policy)

    # Update spex config
    if args.sn_config_changes:
        config_edit.apply_signalflow_profile(args.sn_config_changes)
    if args.fn_config_changes:
        filename_profile = args.fn_config_changes
        config_edit.apply_filename_profile(filename_profile)
        config_mgr.save_last_used_config('spex')

    # Handle config I/O operations
    if args.export_config:
        config_types = ['spex', 'checks'] if args.export_config == 'all' else [args.export_config]
        config_io = ConfigIO(config_mgr)
        filename = config_io.save_configs(args.export_file, config_types)
        print(f"Configs exported to: {filename}")
        if args.dry_run_only:
            sys.exit(0)
    
    if args.import_config:
        config_io = ConfigIO(config_mgr)
        config_io.import_configs(args.import_config)
        print(f"Configs imported from: {args.import_config}")

    if args.print_config_profile:
        config_edit.print_config(args.print_config_profile)

    if args.dry_run_only:
        logger.critical("Dry run selected. Exiting now.")
        sys.exit(1)


def run_avspex(source_directories, signals=None):
    processor = AVSpexProcessor(signals=signals)
    try:
        processor.initialize()
        formatted_time = processor.process_directories(source_directories)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def main_gui():
    args = parse_arguments()
    
    # Get application (will show splash screen)
    app = LazyGUILoader.get_application()
    
    # Get main window (will close splash screen after delay)
    window = LazyGUILoader.get_main_window()
    window.show()
    
    return app.exec()


def main_cli():
    args = parse_arguments()

    if args.gui:
       main_gui()
    else:
        run_cli_mode(args)
        if args.source_directories:
            run_avspex(args.source_directories)


def main():
    args = parse_arguments()

    if args.gui or (args.source_directories is None and not sys.argv[1:]):
        main_gui()
    else:
        main_cli()


if __name__ == "__main__":
    main()

