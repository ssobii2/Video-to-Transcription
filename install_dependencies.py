#!/usr/bin/env python3
"""
Dependency installation dispatcher for Video to Transcription Service
Automatically detects platform (Windows/Linux) and package manager (UV/pip)
Delegates to the appropriate specialized installation script
"""

import subprocess
import sys
import os
import platform
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_platform():
    """Detect the current platform"""
    system = platform.system().lower()
    if system == 'windows':
        return 'windows'
    elif system in ['linux', 'darwin']:  # Darwin is macOS, but we'll treat it like Linux
        return 'linux'
    else:
        logger.warning(f"Unsupported platform: {system}, defaulting to Linux")
        return 'linux'

def detect_package_manager():
    """Detect and recommend UV for best experience, fallback to pip/venv"""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, check=True, text=True)
        logger.info(f"UV detected: {result.stdout.strip()}")
        return 'uv'
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("UV not found - falling back to traditional pip/venv")
        return 'pip'

def get_installation_script(platform_name, package_manager):
    """Get the appropriate installation script filename"""
    script_name = f"install_dependencies_{platform_name}_{package_manager}.py"
    script_path = Path(script_name)
    
    if script_path.exists():
        return script_name
    else:
        logger.error(f"Installation script not found: {script_name}")
        return None

def run_installation_script(script_name):
    """Run the selected installation script"""
    try:
        logger.info(f"Running installation script: {script_name}")
        result = subprocess.run([sys.executable, script_name], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Installation script failed: {e}")
        return False
    except FileNotFoundError:
        logger.error(f"Installation script not found: {script_name}")
        return False

def main():
    """Main dispatcher function"""
    logger.info("=== Video to Transcription Service - Installation Dispatcher ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    
    # Detect platform and package manager
    platform_name = detect_platform()
    package_manager = detect_package_manager()
    
    logger.info(f"Detected platform: {platform_name}")
    logger.info(f"Detected package manager: {package_manager}")
    
    # Get the appropriate installation script
    script_name = get_installation_script(platform_name, package_manager)
    
    if script_name is None:
        logger.error("❌ No suitable installation script found")
        logger.info("Available options:")
        logger.info("• Windows + UV: install_dependencies_windows_uv.py")
        logger.info("• Windows + pip: install_dependencies_windows_pip.py")
        logger.info("• Linux + UV: install_dependencies_linux_uv.py")
        logger.info("• Linux + pip: install_dependencies_linux_pip.py")
        
        if package_manager == 'pip':
            logger.info("\n💡 For better performance, consider installing UV:")
            if platform_name == 'windows':
                logger.info("   Windows: https://docs.astral.sh/uv/getting-started/installation/")
            else:
                logger.info("   Linux: curl -LsSf https://astral.sh/uv/install.sh | sh")
        
        sys.exit(1)
    
    # Provide information about the chosen approach
    logger.info(f"\n🎯 Using installation script: {script_name}")
    
    if package_manager == 'uv':
        logger.info("✨ UV detected - using modern Python package management")
        logger.info("   • Faster dependency resolution and installation")
        logger.info("   • Automatic Python version management")
        logger.info("   • Better dependency isolation")
    else:
        logger.info("🔧 Using traditional pip/venv approach")
        logger.info("💡 Consider installing UV for better performance:")
        if platform_name == 'windows':
            logger.info("   Windows: https://docs.astral.sh/uv/getting-started/installation/")
        else:
            logger.info("   Linux: curl -LsSf https://astral.sh/uv/install.sh | sh")
    
    # Run the installation script
    logger.info(f"\n🚀 Starting installation using {script_name}...")
    logger.info("=" * 60)
    
    success = run_installation_script(script_name)
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Installation completed successfully!")
        logger.info("The specific installation script handled all the details.")
    else:
        logger.error("\n" + "=" * 60)
        logger.error("❌ Installation failed!")
        logger.error("Check the output above for specific error details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 