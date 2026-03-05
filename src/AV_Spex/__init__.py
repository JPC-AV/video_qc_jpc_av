# src/AV_Spex/__init__.py
try:
    from importlib.metadata import version
    __version__ = version("AV_Spex")
except Exception:
    import os
    _version_file = os.path.join(os.path.dirname(__file__), "_version.txt")
    if os.path.exists(_version_file):
        with open(_version_file) as f:
            __version__ = f.read().strip()
    else:
        __version__ = "unknown"
