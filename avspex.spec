# avspex.spec - Hybrid approach: Include Plotly normally, handle problematic files during signing
import os
from pathlib import Path

# Get the directory where this spec file is located (now the root directory)
root_dir = os.path.abspath('.')

# Define the source directory where the actual code is
src_dir = os.path.join(root_dir, 'src')

# Define the packaging assets directory for icon files
icon_path = os.path.join(root_dir, 'av_spex_the_logo.icns')

block_cipher = None

a = Analysis(['av_spex_launcher.py'],  # New launcher in root directory
    pathex=[
        root_dir,  # Add root to the Python path
        src_dir    # Add source dir to the Python path
    ],
    binaries=[],
    datas=[
        # Update paths to be relative to the repository structure
        (os.path.join(src_dir, 'AV_Spex/config'), 'AV_Spex/config'),
        (os.path.join(src_dir, 'AV_Spex/logo_image_files'), 'AV_Spex/logo_image_files'),
        (os.path.join(root_dir, 'pyproject.toml'), '.'),
    ],
    hiddenimports=[
        'AV_Spex',
        'AV_Spex.av_spex_the_file',  # Important for main_gui function
        'AV_Spex.processing',
        'AV_Spex.utils',
        'AV_Spex.checks',
        'AV_Spex.gui',
        'AV_Spex.gui.gui_main',
        'AV_Spex.gui.gui_theme_manager',
        'AV_Spex.gui.gui_signals',
        'AV_Spex.gui.gui_import_tab',
        'AV_Spex.gui.gui_spex_tab',
        'AV_Spex.gui.gui_processing_window',
        'AV_Spex.gui.gui_processing_window_console',
        'AV_Spex.gui.gui_main_window',
        'AV_Spex.gui.gui_main_window.gui_main_window_processing',
        'AV_Spex.gui.gui_main_window.gui_main_window_signals',
        'AV_Spex.gui.gui_main_window.gui_main_window_theme',
        'AV_Spex.gui.gui_main_window.gui_main_window_ui',
        'AV_Spex.gui.gui_checks_tab',
        'AV_Spex.gui.gui_checks_tab.gui_checks_tab',
        'AV_Spex.gui.gui_checks_tab.gui_checks_window',
        'PyQt6',
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        # Additional imports for macOS support
        'AppKit',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Only exclude Qt modules we definitely don't need
        'PyQt6.QtDBus', 'PyQt6.QtPdf', 'PyQt6.QtSvg', 'PyQt6.QtNetwork',
        # Only exclude clearly unnecessary plotly modules
        'plotly.matplotlylib', 
        'plotly.figure_factory',
        'plotly.io.orca',
        'plotly.io.kaleido',
        # Let PyInstaller handle the rest of plotly automatically
    ],
    noarchive=False,
    cipher=block_cipher
)

# Only filter out the specific files that cause signing issues, not entire modules
print("Filtering only specific problematic files that cause signing issues...")
original_count = len(a.datas)

a.datas = [
    (dest, source, kind) for dest, source, kind in a.datas
    if not any([
        # Only remove the specific files that were causing signing problems
        'iris.csv.gz' in dest,  # This specific file was in the error
        'plotly.min.js' in dest and 'package_data' in dest,  # Large JS files that aren't needed with CDN
        '/datasets/' in dest and '.csv' in dest and 'plotly' in dest,  # Dataset CSV files
        # Keep templates and other essential files
    ])
]

new_count = len(a.datas)
print(f"Removed {original_count - new_count} problematic data files")

# Debug: Print plotly files being included
print("Plotly files being included:")
plotly_files = [(dest, source, kind) for dest, source, kind in a.datas if 'plotly' in dest.lower()]
print(f"  Total plotly files: {len(plotly_files)}")
for dest, source, kind in plotly_files[:10]:  # Show first 10
    print(f"  {dest}")
if len(plotly_files) > 10:
    print(f"  ... and {len(plotly_files) - 10} more")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AV-Spex',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,  # Set to False for production
    codesign_identity=None,
    entitlements_file=None, 
    target_arch=None,
    universal2=True,
    icon=icon_path
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name='AV-Spex'
)

# Create a full Info.plist with all macOS app requirements
app = BUNDLE(coll,
    name='AV-Spex.app',
    icon=icon_path,
    bundle_identifier='com.jpc.avspex',
    info_plist={
        'CFBundleName': 'AV-Spex',
        'CFBundleDisplayName': 'AV-Spex',
        'CFBundleExecutable': 'AV-Spex',
        'CFBundlePackageType': 'APPL',
        'CFBundleInfoDictionaryVersion': '6.0',
        'NSHumanReadableCopyright': 'Copyright © 2025',
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13.0',
        'LSApplicationCategoryType': 'public.app-category.utilities',
        'LSUIElement': False,
        'LSBackgroundOnly': False,
        'CFBundleIconFile': 'av_spex_the_logo.icns',
        # Add signing-friendly properties
        'CFBundleSupportedPlatforms': ['MacOSX'],
        'DTSDKName': 'macosx',
        'LSRequiresIPhoneOS': False,
    }
)