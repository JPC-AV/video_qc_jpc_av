# avspex.spec

import os
import sys

# Get the directory where this spec file is located
spec_dir = os.path.abspath('.')
# Get the project root directory (same as spec_dir in GitHub Actions)
root_dir = spec_dir

# Set paths for resources
config_path = os.path.join(root_dir, 'src/AV_Spex/config')
logo_path = os.path.join(root_dir, 'src/AV_Spex/logo_image_files')
pyproject_path = os.path.join(root_dir, 'pyproject.toml')

# Print paths for debugging
print(f"spec_dir: {spec_dir}")
print(f"root_dir: {root_dir}")
print(f"config_path: {config_path}")
print(f"logo_path: {logo_path}")
print(f"pyproject_path: {pyproject_path}")

block_cipher = None

a = Analysis(['gui_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        (config_path, 'AV_Spex/config'),
        (logo_path, 'AV_Spex/logo_image_files'),
        (pyproject_path, '.')
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
    target_arch=None,  
    universal2=True,
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