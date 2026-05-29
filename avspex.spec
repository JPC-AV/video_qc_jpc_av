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
        (os.path.join(src_dir, 'AV_Spex/_version.txt'), 'AV_Spex'),
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
        'PyQt6.QtWebEngineCore', 'PyQt6.QtWebEngineWidgets', 'PyQt6.QtWebChannel',
        'PyQt6.QtQml', 'PyQt6.QtQuick', 'PyQt6.QtQuickWidgets',
        'PyQt6.QtMultimedia', 'PyQt6.QtMultimediaWidgets',
        'PyQt6.QtPositioning', 'PyQt6.QtBluetooth', 'PyQt6.QtNfc',
        'PyQt6.QtTest', 'PyQt6.QtSensors', 'PyQt6.QtSerialPort',
        # Plotly sub-packages not needed (report uses include_plotlyjs='cdn')
        'plotly.matplotlylib', 'plotly.figure_factory', 'plotly.offline',
        # scikit-image removed; exclude it and its transitive tail so a stray
        # import can't silently pull tens of MB back into the bundle
        'skimage', 'scikit_image', 'networkx', 'tifffile', 'imageio', 'pywt',
        # Test suites bundled inside heavy packages
        'matplotlib.tests', 'numpy.tests', 'scipy.tests', 'PIL.ImageQt',
        # Test/dev frameworks
        'tkinter',
        'test',
        'tests',
    ],
    noarchive=False,
)

# --- Post-analysis pruning ---------------------------------------------------
# Drop binaries/data files that nothing we ship actually uses. These are data
# files and transitively-pulled Qt frameworks, so `excludes` (module-level)
# can't reach them — they must be filtered out of the TOC here.
#
# Qt removals were verified with `otool -L` against a real build: none of the
# kept binaries (QtCore/QtGui/QtWidgets, libqcocoa, the PyQt6 bindings) link
# these. NOTE: QtGui *does* link QtDBus, so QtDBus is intentionally NOT pruned.
_prune_substrings = (
    'QtPdf',                              # PDF module — unused (~7 MB)
    'QtNetwork',                          # only used by QtPdf + the TUIO plugin (~1.6 MB)
    'qtuiotouchplugin',                   # TUIO multitouch input — unused
    'plotly/package_data/plotly.min.js',  # report uses include_plotlyjs='cdn'; never read (~3.5 MB)
    'mpl-data/sample_data',              # matplotlib example datasets — unused
    'mpl-data/fonts/afm',                # Adobe Font Metrics — only the PS backend uses these
    'mpl-data/fonts/pdfcorefonts',       # PDF-core fonts — only the PDF backend uses these
)

def _keep(entry):
    dest = entry[0].replace(os.sep, '/')
    return not any(s in dest for s in _prune_substrings)

a.binaries = [b for b in a.binaries if _keep(b)]
a.datas = [d for d in a.datas if _keep(d)]

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
