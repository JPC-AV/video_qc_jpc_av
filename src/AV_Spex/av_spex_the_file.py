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
from .utils.dependency_checker import DependencyManager, cli_deps_check

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
    enable_border_detection: Optional[str]
    enable_brng_analysis: Optional[str]
    enable_signalstats: Optional[str]
    frame_borders: Optional[str]
    frame_border_pixels: Optional[int]
    frame_no_colorbar_skip: bool
    frame_brng_duration: Optional[int]
    exiftool_profile: Optional[str]
    mediainfo_profile: Optional[str]
    ffprobe_profile: Optional[str]


PROFILE_MAPPING = {
    "step1": config_edit.profile_step1,
    "step2": config_edit.profile_step2,
    "off": config_edit.profile_allOff
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
    parser.add_argument("-sn","--signalflow",
                        help="Select signal flow profile by name (e.g. 'JPC_AV_SVHS Signal Flow'). Use -pp signalflow to list available profiles.")
    parser.add_argument("-fn","--filename",
                        help="Select filename profile by name (e.g. 'JPC Filename Profile'). Use -pp to list available profiles.")
    parser.add_argument("-pp", "--printprofile", type=str, nargs='?', const='all', default=None,
                        help="Show config profile(s) and optional subsection. For current settings: 'config[,subsection]'. Examples: 'all', 'spex', 'checks', 'checks,tools', 'spex,filename_values'. For custom profiles: 'exiftool', 'mediainfo', 'ffprobe', 'signalflow'")
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
    parser.add_argument("--exiftool-profile",
                    help="Apply an exiftool expected-values profile by name. Use -pp exiftool to list available exiftool profiles.")
    parser.add_argument("--mediainfo-profile",
                    help="Apply a MediaInfo expected-values profile by name. Use -pp mediainfo to list available mediainfo profiles.")
    parser.add_argument("--ffprobe-profile",
                    help="Apply an FFprobe expected-values profile by name. Use -pp ffprobe to list available ffprobe profiles.")

    # args for frame analysis
    # Enable/disable individual sub-steps
    parser.add_argument(
        '--enable-border-detection',
        choices=['yes', 'no'],
        help='Enable/disable border detection in frame analysis'
    )
    parser.add_argument(
        '--enable-brng-analysis', 
        choices=['yes', 'no'],
        help='Enable/disable BRNG analysis in frame analysis'
    )
    parser.add_argument(
        '--enable-signalstats',
        choices=['yes', 'no'],
        help='Enable/disable signalstats in frame analysis'
    )
    parser.add_argument(
        '--frame-borders',
        choices=['simple', 'sophisticated'],
        help='Border detection mode for frame analysis (simple or sophisticated)'
    )
    parser.add_argument(
        '--frame-border-pixels',
        type=int,
        help='Number of pixels to crop from each edge in simple border mode'
    )
    parser.add_argument(
        '--frame-no-colorbar-skip',
        action='store_true',
        help='Disable automatic skipping of color bars detected by qct-parse'
    )
    parser.add_argument(
        '--frame-brng-duration',
        type=int,
        help='Maximum duration in seconds for BRNG analysis'
    )

    args = parser.parse_args()

    input_paths = args.paths if args.paths else []
    source_directories = dir_setup.validate_input_paths(input_paths, args.file)

    selected_profile = config_edit.resolve_config(args.profile, PROFILE_MAPPING)
    sn_config_changes = args.signalflow
    fn_config_changes = args.filename

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
        use_default_config=args.use_default_config,
        enable_border_detection=getattr(args, 'enable_border_detection', None),
        enable_brng_analysis=getattr(args, 'enable_brng_analysis', None),
        enable_signalstats=getattr(args, 'enable_signalstats', None),
        frame_borders=getattr(args, 'frame_borders', None),
        frame_border_pixels=getattr(args, 'frame_border_pixels', None),
        frame_no_colorbar_skip=getattr(args, 'frame_no_colorbar_skip', False),
        frame_brng_duration=getattr(args, 'frame_brng_duration', None),
        exiftool_profile=args.exiftool_profile,
        mediainfo_profile=args.mediainfo_profile,
        ffprobe_profile=args.ffprobe_profile,
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
    config_mgr.save_config('spex', is_last_used=True)

def print_av_spex_logo():
    avspex_icon = text2art("A-V Spex", font='5lineoblique')
    print(f'{avspex_icon}\n')


def run_cli_mode(args):
    print_av_spex_logo()

    cli_deps_check()

    # Update checks config
    if args.selected_profile:
        config_edit.apply_profile(args.selected_profile)
        config_mgr.save_config('checks', is_last_used=True)
    if args.tools_on_names:
        config_edit.toggle_on(args.tools_on_names)
        config_mgr.save_config('checks', is_last_used=True)
    if args.tools_off_names:
        config_edit.toggle_off(args.tools_off_names)
        config_mgr.save_config('checks', is_last_used=True)

    if args.mediaconch_policy:
        processing_mgmt.setup_mediaconch_policy(args.mediaconch_policy)

    if args.exiftool_profile:
        profile = config_edit.get_exiftool_profile(args.exiftool_profile)
        if profile:
            config_edit.apply_exiftool_profile(profile)
            config_mgr.save_config('spex', is_last_used=True)
        else:
            available = config_edit.get_available_exiftool_profiles()
            print(f"Error: exiftool profile '{args.exiftool_profile}' not found. Available: {available}")

    if args.mediainfo_profile:
        profile = config_edit.get_mediainfo_profile(args.mediainfo_profile)
        if profile:
            config_edit.apply_mediainfo_profile(profile)
            config_mgr.save_config('spex', is_last_used=True)
        else:
            available = config_edit.get_available_mediainfo_profiles()
            print(f"Error: MediaInfo profile '{args.mediainfo_profile}' not found. Available: {available}")

    if args.ffprobe_profile:
        profile = config_edit.get_ffprobe_profile(args.ffprobe_profile)
        if profile:
            config_edit.apply_ffprobe_profile(profile)
            config_mgr.save_config('spex', is_last_used=True)
        else:
            available = config_edit.get_available_ffprobe_profiles()
            print(f"Error: FFprobe profile '{args.ffprobe_profile}' not found. Available: {available}")

    # Update spex config
    if args.sn_config_changes:
        _sn_aliases = {'JPC_AV_SVHS': 'JPC_AV_SVHS Signal Flow', 'BVH3100': 'BVH3100 Signal Flow'}
        sn_name = _sn_aliases.get(args.sn_config_changes, args.sn_config_changes)
        sn_profile = config_edit.get_signalflow_profile(sn_name)
        if sn_profile:
            config_edit.apply_signalflow_profile(sn_profile)
            config_mgr.save_config('spex', is_last_used=True)
        else:
            from .utils.config_setup import SignalflowConfig
            available = list(config_mgr.get_config('signalflow', SignalflowConfig).signalflow_profiles.keys())
            print(f"Error: signalflow profile '{args.sn_config_changes}' not found. Available: {available}")

    if args.fn_config_changes:
        _fn_aliases = {'jpc': 'JPC Filename Profile', 'bowser': 'Bowser Filename Profile'}
        fn_name = _fn_aliases.get(args.fn_config_changes, args.fn_config_changes)
        fn_config = config_mgr.get_config('filename', FilenameConfig)
        if fn_name in fn_config.filename_profiles:
            config_edit.apply_filename_profile(fn_config.filename_profiles[fn_name])
            config_mgr.save_config('spex', is_last_used=True)
        else:
            available = list(fn_config.filename_profiles.keys())
            print(f"Error: filename profile '{args.fn_config_changes}' not found. Available: {available}")

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

    # Handle individual frame analysis sub-step configuration
    frame_updates = {'outputs': {'frame_analysis': {}}}

    # Handle enabling/disabling individual sub-steps
    if args.enable_border_detection:
        frame_updates['outputs']['frame_analysis']['enable_border_detection'] = (args.enable_border_detection == 'yes')

    if args.enable_brng_analysis:
        frame_updates['outputs']['frame_analysis']['enable_brng_analysis'] = (args.enable_brng_analysis == 'yes')

    if args.enable_signalstats:
        frame_updates['outputs']['frame_analysis']['enable_signalstats'] = (args.enable_signalstats == 'yes')

    # Handle configuration of how sub-steps work
    if args.frame_borders is not None:
        frame_updates['outputs']['frame_analysis']['border_detection_mode'] = args.frame_borders

    if args.frame_border_pixels is not None:
        frame_updates['outputs']['frame_analysis']['simple_border_pixels'] = args.frame_border_pixels

    if args.frame_no_colorbar_skip:
        frame_updates['outputs']['frame_analysis']['brng_skip_color_bars'] = False

    if args.frame_brng_duration is not None:
        frame_updates['outputs']['frame_analysis']['brng_duration_limit'] = args.frame_brng_duration

    # Only update config if there are actual changes
    if frame_updates['outputs']['frame_analysis']:
        config_mgr.update_config('checks', frame_updates)
        config_mgr.save_config('checks', is_last_used=True)

    if args.dry_run_only:
        logger.critical("Dry run selected. Exiting now.")
        sys.exit(1)


def run_avspex(source_directories, signals=None):
    processor = AVSpexProcessor(signals=signals)
    try:
        formatted_time = processor.process_directories(source_directories)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def main_gui():
    args = parse_arguments()
    
    # Get application (will show splash screen)
    app = LazyGUILoader.get_application()

    # Check dependencies before showing main window
    if not DependencyManager.check_dependencies_gui():
        # User chose to close the application
        sys.exit(0)
    
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

