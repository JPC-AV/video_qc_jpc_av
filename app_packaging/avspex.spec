# avspex.spec

import os
from pathlib import Path

# Get the directory where this spec file is located
spec_dir = os.path.abspath('.')
# Get the project root directory (one level up from spec_dir)
root_dir = os.path.dirname(spec_dir)

block_cipher = None

a = Analysis(['gui_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        (os.path.join(root_dir, 'src/AV_Spex/config'), 'AV_Spex/config'),
        (os.path.join(root_dir, 'src/AV_Spex/logo_image_files'), 'AV_Spex/logo_image_files'),
        (os.path.join(root_dir, 'pyproject.toml'), '.')
    ],
    hiddenimports=[
        'AV_Spex.processing',
        'AV_Spex.utils',
        'AV_Spex.checks'
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
    console=False,
    codesign_identity=None,
    entitlements_file=None, 
    target_arch=None,  # Build for the current architecture
    universal2=True,   # Build a universal binary (both Intel and Apple Silicon)
    icon='av_spex_the_logo.icns'
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
    icon=os.path.join(spec_dir, 'av_spex_the_logo.icns'),
    bundle_identifier='com.jpc.avspex'
)
