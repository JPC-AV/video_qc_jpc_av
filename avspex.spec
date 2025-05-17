# avspex.spec
import os
from pathlib import Path

# Get the directory where this spec file is located (now the root directory)
root_dir = os.path.abspath('.')

# Define the source directory where the actual code is
src_dir = os.path.join(root_dir, 'src')

# Define the packaging assets directory for icon files
# Option 1: If you keep the icon in app_packaging
icon_path = os.path.join(root_dir, 'app_packaging', 'av_spex_the_logo.icns')
# Option 2: If you move the icon to root (uncomment this if you move the icon)
# icon_path = os.path.join(root_dir, 'av_spex_the_logo.icns')

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
        (os.path.join(root_dir, 'pyproject.toml'), '.')
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
        'PyQt6.QtGui'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt6.QtDBus', 'PyQt6.QtPdf', 'PyQt6.QtSvg', 'PyQt6.QtNetwork',
        'plotly.matplotlylib', 'plotly.figure_factory'
    ],
    noarchive=False,
    cipher=block_cipher
)

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
    console=True,  # Keep True for debugging, change to False for production
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

app = BUNDLE(coll,
    name='AV-Spex.app',
    icon=icon_path,
    bundle_identifier='com.jpc.avspex'
)