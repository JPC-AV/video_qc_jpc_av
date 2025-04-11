# avspex.spec

import os
from pathlib import Path

# Get the directory where this spec file is located
spec_dir = os.path.abspath('.')

# Determine if we're in local dev environment or GitHub Actions
# by checking common directory structures
is_github_actions = os.path.exists(os.path.join(spec_dir, '.github')) or 'GITHUB_WORKSPACE' in os.environ

# Set paths that work in both environments
if is_github_actions:
    # In GitHub Actions, resources are typically at the repository root
    root_dir = spec_dir
    src_dir = os.path.join(root_dir, 'src')
else:
    # In local development, spec file might be in a subdirectory
    root_dir = os.path.dirname(spec_dir)  # One level up from spec dir
    src_dir = os.path.join(root_dir, 'src')

# Function to find path that exists
def find_existing_path(*possible_paths):
    for path in possible_paths:
        if os.path.exists(path):
            return path
    print(f"Warning: None of the paths exist: {possible_paths}")
    return possible_paths[0]  # Return first option even if it doesn't exist

# Find key directories/files
config_path = find_existing_path(
    os.path.join(src_dir, 'AV_Spex', 'config'),
    os.path.join(root_dir, 'AV_Spex', 'config'),
    os.path.join(root_dir, 'src', 'AV_Spex', 'config')
)

logo_path = find_existing_path(
    os.path.join(src_dir, 'AV_Spex', 'logo_image_files'),
    os.path.join(root_dir, 'AV_Spex', 'logo_image_files'),
    os.path.join(root_dir, 'src', 'AV_Spex', 'logo_image_files')
)

pyproject_path = find_existing_path(
    os.path.join(root_dir, 'pyproject.toml'),
    os.path.join(spec_dir, 'pyproject.toml')
)

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