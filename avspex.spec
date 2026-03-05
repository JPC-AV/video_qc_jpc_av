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

# Read version from pyproject.toml so the spec never goes stale
import re
_pyproject = os.path.join(root_dir, 'pyproject.toml')
_version = '0.0.0'
if os.path.exists(_pyproject):
    with open(_pyproject) as _f:
        _m = re.search(r'^version\s*=\s*"([^"]+)"', _f.read(), re.MULTILINE)
        if _m:
            _version = _m.group(1)
print(f"Building AV-Spex version {_version}")

a = Analysis(['av_spex_launcher.py'],
    pathex=[
        root_dir,
        src_dir
    ],
    binaries=[],
    datas=[
        (os.path.join(src_dir, 'AV_Spex/config'), 'AV_Spex/config'),
        (os.path.join(src_dir, 'AV_Spex/logo_image_files'), 'AV_Spex/logo_image_files'),
        (os.path.join(root_dir, 'pyproject.toml'), '.'),
    ],
    hiddenimports=[
        # Core AV_Spex modules
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
        
        # PyQt6 modules
        'PyQt6',
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        
        # scipy 
        'scipy',
        'scipy.ndimage',
        'scipy.signal',
        
        # matplotlib 
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.patches',
        'matplotlib.backends.backend_agg',
        
        # opencv 
        'cv2',
        
        # numpy
        'numpy',
        
        # System modules
        'AppKit',
        'pkg_resources',
        'setuptools',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Qt modules not needed
        'PyQt6.QtDBus', 'PyQt6.QtPdf', 'PyQt6.QtSvg', 'PyQt6.QtNetwork',
        # Plotly sub-packages not needed
        'plotly.matplotlylib', 'plotly.figure_factory',
        # Test/dev frameworks
        'tkinter',
        'numpy.testing',
        'test',
        'tests',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

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
    console=False,
    codesign_identity=None,
    entitlements_file=None,
    target_arch=None,
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
    version=_version,
    info_plist={
        'CFBundleName': 'AV-Spex',
        'CFBundleDisplayName': 'AV-Spex',
        'CFBundleIdentifier': 'com.jpc.avspex',
        'CFBundleVersion': _version,
        'CFBundleShortVersionString': _version,
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '14.0',
        'NSAppleEventsUsageDescription': 'AV-Spex needs access to Apple Events for automation features.',
        'NSRequiresAquaSystemAppearance': False,
    }
)
