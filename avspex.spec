import os
from pathlib import Path

# Get the directory where this spec file is located (now the root directory)
root_dir = os.path.abspath('.')

# Define the source directory where the actual code is
src_dir = os.path.join(root_dir, 'src')

# Define the packaging assets directory for icon files
icon_path = os.path.join(root_dir, 'av_spex_the_logo.icns')

# Verify icon exists, use fallback if not
if not os.path.exists(icon_path):
    print(f"Warning: Icon not found at {icon_path}")
    icon_path = None

block_cipher = None

a = Analysis(['av_spex_launcher.py'],  # Your launcher file
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
        'AV_Spex.av_spex_the_file',
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
        # Add any missing imports that might be needed in CI
        'pkg_resources',
        'setuptools',
        'sqlite3',  # Add this only if you use SQLite
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Only exclude what you originally excluded
        'PyQt6.QtDBus', 'PyQt6.QtPdf', 'PyQt6.QtSvg', 'PyQt6.QtNetwork',
        'plotly.matplotlylib', 'plotly.figure_factory',
        # Additional excludes to reduce bundle size
        'tkinter',
        'matplotlib',
        'scipy',
        'numpy.testing',
        'test',
        'tests',
    ],
    noarchive=False,  # CRITICAL: Keep this False to allow proper symlink structure
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
    console=False,  # Set to False for production
    codesign_identity=None,  # Will be handled by GitHub Actions
    entitlements_file=None,  # Will be handled by GitHub Actions
    target_arch=None,  # Let each runner build for its native architecture
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
    bundle_identifier='com.jpc.avspex',
    version='0.7.8.5',  # Consider making this dynamic
    info_plist={
        'CFBundleName': 'AV-Spex',
        'CFBundleDisplayName': 'AV-Spex',
        'CFBundleIdentifier': 'com.jpc.avspex',
        'CFBundleVersion': '0.7.8.5',
        'CFBundleShortVersionString': '0.7.8.5',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.12',  # Minimum macOS version
        'NSAppleEventsUsageDescription': 'AV-Spex needs access to Apple Events for automation features.',
        'NSCameraUsageDescription': 'AV-Spex may access camera for video processing.',
        'NSMicrophoneUsageDescription': 'AV-Spex may access microphone for audio processing.',
    }
)