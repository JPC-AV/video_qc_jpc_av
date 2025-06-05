#!/usr/bin/env python3
"""
Theme Detection Diagnostic Script for macOS PyQt6 Apps

This script isolates and tests the theme detection mechanisms used in your app
to help identify why theme detection fails when code-signed through GitHub Actions.

Integration with existing AV_Spex logging and configuration systems.
"""

import sys
import os
import subprocess
import platform
from datetime import datetime
from typing import Optional, Dict, Any

# Import your existing logging system
from AV_Spex.utils.log_setup import logger

def check_macos_environment() -> Dict[str, Any]:
    """Check macOS system information and return results"""
    logger.info("=== macOS Environment Check ===")
    results = {}
    
    # Basic system info
    results['platform'] = platform.platform()
    results['macos_version'] = platform.mac_ver()[0]
    results['architecture'] = platform.machine()
    
    logger.info(f"Platform: {results['platform']}")
    logger.info(f"macOS Version: {results['macos_version']}")
    logger.info(f"Architecture: {results['architecture']}")
    
    # Check if running in sandbox
    try:
        home_dir = os.path.expanduser("~")
        results['is_sandboxed'] = "Containers" in home_dir
        
        if results['is_sandboxed']:
            logger.warning("Running in App Sandbox")
        else:
            logger.info("Not sandboxed")
    except Exception as e:
        logger.error(f"Error checking sandbox status: {e}")
        results['sandbox_error'] = str(e)
    
    # Check system appearance setting
    try:
        result = subprocess.run([
            'defaults', 'read', '-g', 'AppleInterfaceStyle'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            results['system_appearance'] = result.stdout.strip()
            logger.info(f"System appearance: {results['system_appearance']}")
        else:
            results['system_appearance'] = "Light"
            logger.info("System appearance: Light (default - no Dark mode setting)")
    except subprocess.TimeoutExpired:
        logger.error("Timeout checking system appearance")
        results['appearance_error'] = "Timeout"
    except Exception as e:
        logger.error(f"Error checking system appearance: {e}")
        results['appearance_error'] = str(e)
    
    return results

def check_app_signature() -> Dict[str, Any]:
    """Check code signing information"""
    logger.info("=== Code Signing Check ===")
    results = {}
    
    try:
        # Get the path to the current executable
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller bundle
            app_path = sys.executable
            results['bundle_type'] = 'pyinstaller'
            logger.info(f"Running as bundle: {app_path}")
        else:
            # Running as script
            app_path = sys.executable
            results['bundle_type'] = 'script'
            logger.info(f"Running as script with Python: {app_path}")
        
        results['app_path'] = app_path
        
        # Check code signature
        result = subprocess.run([
            'codesign', '-dv', app_path
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            results['is_signed'] = True
            results['signature_details'] = result.stderr
            logger.info("App is code signed")
            logger.debug(f"Signature details:\n{result.stderr}")
        else:
            results['is_signed'] = False
            logger.info("App is not code signed")
            
    except subprocess.TimeoutExpired:
        logger.error("Timeout checking code signature")
        results['signature_error'] = "Timeout"
    except Exception as e:
        logger.error(f"Error checking code signature: {e}")
        results['signature_error'] = str(e)
    
    return results

def test_pyqt_theme_detection() -> Dict[str, Any]:
    """Test PyQt6 theme detection mechanisms"""
    logger.info("=== PyQt6 Theme Detection Test ===")
    results = {}
    
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QPalette
        from PyQt6.QtCore import Qt
        
        # Create minimal application or use existing
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            results['created_new_app'] = True
        else:
            results['created_new_app'] = False
            
        logger.info("QApplication created/retrieved")
        
        # Get system palette
        palette = app.palette()
        logger.info("System palette retrieved")
        
        # Test the exact same logic your app uses
        window_color = palette.color(QPalette.ColorRole.Window)
        lightness = window_color.lightness()
        is_dark = lightness < 128
        
        results['window_color'] = window_color.name()
        results['window_lightness'] = lightness
        results['detected_theme'] = 'Dark' if is_dark else 'Light'
        
        logger.info(f"Window color: {results['window_color']}")
        logger.info(f"Window color lightness: {results['window_lightness']}")
        logger.info(f"Detected theme: {results['detected_theme']}")
        
        # Test other color roles that might be affected
        color_roles = [
            ('Window', QPalette.ColorRole.Window),
            ('Base', QPalette.ColorRole.Base),
            ('Text', QPalette.ColorRole.Text),
            ('Button', QPalette.ColorRole.Button),
            ('ButtonText', QPalette.ColorRole.ButtonText),
            ('Highlight', QPalette.ColorRole.Highlight),
            ('HighlightedText', QPalette.ColorRole.HighlightedText),
        ]
        
        results['color_analysis'] = {}
        logger.info("--- Palette Color Analysis ---")
        for role_name, role in color_roles:
            color = palette.color(role)
            color_info = {
                'hex': color.name(),
                'lightness': color.lightness()
            }
            results['color_analysis'][role_name] = color_info
            logger.debug(f"{role_name}: {color.name()} (lightness: {color.lightness()})")
        
        # Test if palette changes are detected
        logger.info("--- Palette Change Detection Test ---")
        change_detected = [False]
        
        def on_palette_changed():
            change_detected[0] = True
            logger.info("Palette change signal received")
        
        app.paletteChanged.connect(on_palette_changed)
        logger.info("Connected to paletteChanged signal")
        
        # Force a palette update to test signal
        app.setPalette(palette)
        app.processEvents()
        
        results['palette_signal_working'] = change_detected[0]
        if change_detected[0]:
            logger.info("Palette change detection working")
        else:
            logger.warning("Palette change signal may not be working")
        
        results['pyqt_success'] = True
        return results
        
    except ImportError as e:
        logger.error(f"PyQt6 import error: {e}")
        results['pyqt_success'] = False
        results['error'] = f"Import error: {e}"
        return results
    except Exception as e:
        logger.error(f"PyQt6 theme detection error: {e}")
        results['pyqt_success'] = False
        results['error'] = str(e)
        return results

def test_alternative_theme_detection() -> Dict[str, Any]:
    """Test alternative methods for theme detection"""
    logger.info("=== Alternative Theme Detection Methods ===")
    results = {}
    
    # Method 1: Check NSUserDefaults directly (requires pyobjc)
    try:
        from Foundation import NSUserDefaults
        defaults = NSUserDefaults.standardUserDefaults()
        interface_style = defaults.stringForKey_("AppleInterfaceStyle")
        
        results['nsuserdefaults_method'] = {
            'available': True,
            'theme': 'Dark' if interface_style == "Dark" else 'Light'
        }
        
        if interface_style == "Dark":
            logger.info("NSUserDefaults method: Dark theme detected")
        else:
            logger.info("NSUserDefaults method: Light theme detected")
            
    except ImportError:
        logger.info("pyobjc not available for NSUserDefaults method")
        results['nsuserdefaults_method'] = {'available': False, 'error': 'pyobjc not available'}
    except Exception as e:
        logger.error(f"NSUserDefaults method error: {e}")
        results['nsuserdefaults_method'] = {'available': False, 'error': str(e)}
    
    # Method 2: Use subprocess to check defaults
    try:
        result = subprocess.run([
            'defaults', 'read', '-g', 'AppleInterfaceStyle'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            style = result.stdout.strip()
            results['subprocess_method'] = {
                'available': True,
                'theme': style
            }
            logger.info(f"subprocess defaults method: {style} theme detected")
        else:
            results['subprocess_method'] = {
                'available': True,
                'theme': 'Light'
            }
            logger.info("subprocess defaults method: Light theme (no setting)")
            
    except Exception as e:
        logger.error(f"subprocess defaults method error: {e}")
        results['subprocess_method'] = {'available': False, 'error': str(e)}
    
    return results

def check_entitlements() -> Dict[str, Any]:
    """Check app entitlements that might affect theme detection"""
    logger.info("=== Entitlements Check ===")
    results = {}
    
    try:
        if getattr(sys, 'frozen', False):
            app_path = sys.executable
        else:
            logger.info("Not running as app bundle, skipping entitlements check")
            results['available'] = False
            results['reason'] = 'Not app bundle'
            return results
            
        result = subprocess.run([
            'codesign', '-d', '--entitlements', ':-', app_path
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            results['available'] = True
            results['entitlements'] = result.stdout
            logger.info("Entitlements retrieved")
            logger.debug(f"Entitlements:\n{result.stdout}")
        else:
            results['available'] = False
            results['reason'] = 'No entitlements found or not code signed'
            logger.info("No entitlements found or not code signed")
            
    except Exception as e:
        logger.error(f"Error checking entitlements: {e}")
        results['available'] = False
        results['error'] = str(e)
    
    return results

def run_full_diagnostic() -> Dict[str, Any]:
    """Run complete diagnostic and return structured results"""
    logger.info("=" * 60)
    logger.info("THEME DETECTION DIAGNOSTIC REPORT")
    logger.info("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'macos_environment': check_macos_environment(),
        'app_signature': check_app_signature(),
        'entitlements': check_entitlements(),
        'pyqt_theme': test_pyqt_theme_detection(),
        'alternative_methods': test_alternative_theme_detection()
    }
    
    # Generate recommendations
    recommendations = []
    
    if not results['pyqt_theme'].get('pyqt_success', False):
        recommendations.extend([
            "PyQt6 theme detection failed completely",
            "Check PyQt6 installation",
            "Verify system compatibility"
        ])
    else:
        recommendations.extend([
            "Basic PyQt6 theme detection working",
            "Issue may be timing-related",
            "Check application startup sequence"
        ])
    
    # Specific recommendations for GitHub Actions signing issues
    recommendations.extend([
        "",
        "GitHub Actions Code Signing Solutions:",
        "1. Add hardened runtime entitlements:",
        "   - com.apple.security.cs.allow-jit",
        "   - com.apple.security.cs.allow-unsigned-executable-memory", 
        "   - com.apple.security.cs.disable-library-validation",
        "2. Ensure Info.plist includes LSUIElement or LSBackgroundOnly if needed",
        "3. Test with different notarization options",
        "4. Consider fallback theme detection methods"
    ])
    
    results['recommendations'] = recommendations
    
    logger.info("=== Recommendations ===")
    for rec in recommendations:
        if rec.strip():
            logger.info(rec)
    
    return results

def quick_theme_check() -> str:
    """Quick theme detection for use in your app - returns 'Dark' or 'Light'"""
    try:
        # First try PyQt6 method
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QPalette
        
        app = QApplication.instance()
        if app:
            palette = app.palette()
            window_color = palette.color(QPalette.ColorRole.Window)
            is_dark = window_color.lightness() < 128
            return 'Dark' if is_dark else 'Light'
    except:
        pass
    
    # Fallback to subprocess method
    try:
        result = subprocess.run([
            'defaults', 'read', '-g', 'AppleInterfaceStyle'
        ], capture_output=True, text=True, timeout=2)
        
        if result.returncode == 0:
            return 'Dark'
        else:
            return 'Light'
    except:
        pass
    
    # Final fallback
    return 'Light'

# CLI interface for standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Theme Detection Diagnostic Tool")
    parser.add_argument('--quick', action='store_true', 
                       help='Just return current theme (Dark/Light)')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    
    args = parser.parse_args()
    
    if args.quick:
        print(quick_theme_check())
    else:
        results = run_full_diagnostic()
        
        if args.json:
            import json
            print(json.dumps(results, indent=2))