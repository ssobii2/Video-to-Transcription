#!/usr/bin/env python3
"""
Linux pip/venv Dependency installation script for Video to Transcription Service
Automatically detects CUDA version and installs appropriate PyTorch
Uses traditional pip and venv for package management on Linux
Ensures proper environment isolation for both local and server deployments
"""

import subprocess
import sys
import os
import platform
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_package_manager():
    """Linux pip/venv package manager detection"""
    logger.info("Using traditional Python virtual environment")
    return 'venv'

def get_best_python_executable():
    """Find the best Python 3.11+ executable available on the system"""
    # List of possible Python executables to try, in order of preference
    python_candidates = [
        sys.executable,  # Current Python interpreter
        'python3.11',
        'python3.12', 
        'python3.13',
        'python3',
        'python'
    ]
    
    for python_cmd in python_candidates:
        try:
            # Test if this Python version works and is 3.11+
            result = subprocess.run([python_cmd, '--version'], capture_output=True, text=True, check=True)
            version_output = result.stdout.strip()
            
            # Extract version numbers
            import re
            version_match = re.search(r'Python (\d+)\.(\d+)', version_output)
            if version_match:
                major, minor = int(version_match.group(1)), int(version_match.group(2))
                if major >= 3 and minor >= 11:
                    logger.info(f"Found suitable Python: {python_cmd} ({version_output})")
                    return python_cmd
                else:
                    logger.debug(f"Python {python_cmd} is too old: {version_output}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug(f"Python executable not found or failed: {python_cmd}")
            continue
    
    logger.error("No suitable Python 3.11+ executable found!")
    logger.error("Available candidates tested: " + ", ".join(python_candidates))
    return None

def setup_venv():
    """Setup Python virtual environment if not using UV"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        logger.info("✅ Virtual environment already exists")
        # Verify it's working
        venv_python = get_venv_python()
        if Path(venv_python).exists():
            logger.info(f"🎯 Using existing virtual environment: {venv_python}")
            return True
        else:
            logger.warning("⚠️  Existing venv appears corrupted, recreating...")
            # Remove corrupted venv and recreate
            import shutil
            shutil.rmtree(venv_path)
    
    logger.info("Creating Python virtual environment...")
    
    # Find the best Python executable to use
    python_executable = get_best_python_executable()
    if not python_executable:
        logger.error("Cannot create virtual environment: No suitable Python found")
        logger.error("Please ensure Python 3.11 is installed")
        return False
    
    try:
        # First try to ensure we have venv module
        test_venv = subprocess.run([python_executable, "-m", "venv", "--help"], capture_output=True, check=False)
        if test_venv.returncode != 0:
            logger.error(f"venv module not available for {python_executable}")
            logger.error("Try installing: sudo apt install python3-venv or python3.11-venv depending on how many python versions you have installed")
            return False
        
        # Create virtual environment with more verbose output
        logger.info(f"Using Python: {python_executable}")
        result = subprocess.run([python_executable, "-m", "venv", "venv"], check=True, capture_output=True, text=True)
        logger.info("✅ Virtual environment created successfully")
        
        # Verify the venv was created properly
        venv_python = get_venv_python()
        if not Path(venv_python).exists():
            logger.error(f"Virtual environment creation succeeded but Python executable not found: {venv_python}")
            
            # Debug: list what's actually in the venv/bin directory
            venv_bin = Path("venv") / "bin"
            if venv_bin.exists():
                logger.error(f"Contents of {venv_bin}:")
                try:
                    for item in venv_bin.iterdir():
                        logger.error(f"  - {item.name}")
                except Exception as e:
                    logger.error(f"  Could not list directory: {e}")
            else:
                logger.error(f"Directory {venv_bin} does not exist")
            return False
        
        logger.info(f"✅ Virtual environment verified: {venv_python}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.error(f"Error details: {e.stderr}")
        logger.error("Common solutions:")
        logger.error("1. Install python3-venv/python3.11-venv: sudo apt install python3-venv or sudo apt install python3.11-venv depending on how many python versions you have installed")
        logger.error("2. Try using a different Python version")
        logger.error(f"3. Check permissions in directory: {os.getcwd()}")
        return False

def get_venv_python():
    """Get path to Python executable in virtual environment for Linux"""
    venv_bin = Path("venv") / "bin"
    
    # Try different possible Python executable names in the venv
    candidates = ["python", "python3", "python3.11", "python3.12", "python3.13"]
    
    for candidate in candidates:
        python_path = venv_bin / candidate
        if python_path.exists():
            return str(python_path)
    
    # Fallback to the standard name
    return str(venv_bin / "python")

def get_venv_pip():
    """Get path to pip executable in virtual environment for Linux"""
    venv_bin = Path("venv") / "bin"
    
    # Try different possible pip executable names in the venv
    candidates = ["pip", "pip3", "pip3.11", "pip3.12", "pip3.13"]
    
    for candidate in candidates:
        pip_path = venv_bin / candidate
        if pip_path.exists():
            return str(pip_path)
    
    # Fallback to the standard name
    return str(venv_bin / "pip")

def get_python_command():
    """Get the appropriate Python command"""
    # Use virtual environment
    venv_python = get_venv_python()
    if Path(venv_python).exists():
        logger.info("Using virtual environment Python")
        return [venv_python]
    else:
        logger.warning("Virtual environment not found, using system Python")
        return [sys.executable]

def get_pip_command():
    """Get the appropriate pip command - prefer python -m pip over direct pip"""
    venv_python = get_venv_python()
    if Path(venv_python).exists():
        # Use python -m pip instead of direct pip to avoid permission issues
        return [venv_python, "-m", "pip"]
    else:
        return [sys.executable, "-m", "pip"]

def run_command(command, check=True, capture_output=True, show_progress=False):
    """Run a command and return the result"""
    try:
        # Only log essential commands, not detailed scripts
        should_log = not show_progress
        if isinstance(command, list) and len(command) > 2:
            # Don't log if it's a Python script command (contains -c)
            if '-c' in command:
                should_log = False
            # Don't log if it's a pip install command (too verbose)
            elif 'install' in command:
                # Just log a simplified version
                if 'pip' in ' '.join(command):
                    logger.info("Running: pip install (packages)")
                should_log = False
        
        if should_log:
            logger.info(f"Running: {' '.join(command) if isinstance(command, list) else command}")
        
        if show_progress:
            # For progress display, don't capture output - let it stream to console
            result = subprocess.run(
                command,
                check=check,
                text=True,
                shell=isinstance(command, str)
            )
        else:
            # For silent commands, capture output as before
            result = subprocess.run(
                command,
                check=check,
                capture_output=capture_output,
                text=True,
                shell=isinstance(command, str)
            )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if capture_output and hasattr(e, 'stderr') and e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return e

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 11):
        logger.error(f"Python 3.11 required. Current version: {sys.version}")
        return False
    logger.info(f"Python {sys.version.split()[0]} detected")
    return True

def detect_cuda_version():
    """Detect CUDA version if available"""
    try:
        # Try nvidia-smi first
        result = run_command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], check=False)
        if result.returncode == 0:
            logger.info("NVIDIA GPU detected")
            
            # Try to get CUDA version from nvcc
            nvcc_result = run_command(['nvcc', '--version'], check=False)
            if nvcc_result.returncode == 0:
                # Parse CUDA version from nvcc output
                match = re.search(r'release (\d+\.\d+)', nvcc_result.stdout)
                if match:
                    cuda_version = match.group(1)
                    logger.info(f"CUDA {cuda_version} detected")
                    return cuda_version
            
            # If nvcc fails, try checking driver version
            try:
                # Extract driver version from nvidia-smi output
                driver_version = result.stdout.strip()
                driver_major = int(driver_version.split('.')[0])
                
                # Map driver versions to CUDA versions (rough approximation)
                if driver_major >= 535:
                    cuda_version = "12.2"
                elif driver_major >= 525:
                    cuda_version = "12.0"
                elif driver_major >= 470:
                    cuda_version = "11.8"
                else:
                    cuda_version = "11.8"  # Safe fallback
                
                logger.info(f"CUDA {cuda_version} detected via driver mapping (driver {driver_version})")
                return cuda_version
            except:
                pass
            
            # Default CUDA version for Linux with NVIDIA GPU
            logger.info("CUDA detected but version unknown, defaulting to 12.1")
            return "12.1"
        else:
            logger.info("No NVIDIA GPU detected, using CPU-only PyTorch")
            return None
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")
        return None

def get_torch_install_command(cuda_version=None):
    """Get PyTorch installation command based on CUDA version"""
    base_packages = ["torch", "torchvision", "torchaudio"]
    
    if cuda_version is None:
        # CPU-only installation
        logger.info("Installing CPU-only PyTorch")
        return base_packages + ["--index-url", "https://download.pytorch.org/whl/cpu"]
    else:
        # CUDA installation
        cuda_major = cuda_version.split('.')[0]
        cuda_minor = cuda_version.split('.')[1] if '.' in cuda_version else '0'
        
        if cuda_major == "12":
            if cuda_minor in ["0", "1"]:
                logger.info(f"Installing PyTorch with CUDA 12.1 support")
                return base_packages + ["--index-url", "https://download.pytorch.org/whl/cu121"]
            else:  # 12.2, 12.3, etc.
                logger.info(f"Installing PyTorch with CUDA 12.4 support")
                return base_packages + ["--index-url", "https://download.pytorch.org/whl/cu124"]
        elif cuda_major == "11":
            logger.info(f"Installing PyTorch with CUDA 11.8 support")
            return base_packages + ["--index-url", "https://download.pytorch.org/whl/cu118"]
        else:
            logger.warning(f"Unsupported CUDA version {cuda_version}, falling back to CPU")
            return base_packages + ["--index-url", "https://download.pytorch.org/whl/cpu"]

def are_base_requirements_installed():
    """Check if base requirements are already installed using pip"""
    try:
        # Check a few key packages
        key_packages = ['fastapi', 'uvicorn', 'python-multipart', 'faster-whisper']
        
        python_cmd = get_python_command()
        
        for package in key_packages:
            test_cmd = python_cmd + ["-c", f"""
try:
    import {package.replace('-', '_')}
    print(f"✅ {package} already installed")
except ImportError:
    print(f"❌ {package} not found")
    raise
"""]
            result = run_command(test_cmd, check=False)
            if result.returncode != 0:
                logger.info(f"Base requirements not fully installed (missing {package})")
                return False
        
        logger.info("✅ All base requirements appear to be installed")
        return True
        
    except Exception as e:
        logger.info(f"Base requirements check failed: {e}")
        return False

def install_base_requirements():
    """Install base requirements using pip"""
    try:
        if are_base_requirements_installed():
            logger.info("✅ Base requirements already installed, skipping")
            return True
        
        logger.info("Installing base requirements...")
        
        # Install from requirements.txt using pip
        if Path("requirements.txt").exists():
            pip_cmd = get_pip_command()
            install_cmd = pip_cmd + ['install', '-r', 'requirements.txt']
            logger.info("Installing from requirements.txt using pip...")
            result = run_command(install_cmd, show_progress=True, capture_output=False)
            
            if result.returncode == 0:
                logger.info("✅ Base requirements installed successfully")
                return True
            else:
                logger.error("❌ Failed to install base requirements from requirements.txt")
                return False
        else:
            logger.error("❌ requirements.txt not found")
            return False
            
    except Exception as e:
        logger.error(f"Base requirements installation failed: {e}")
        return False

def is_torch_installed():
    """Check if PyTorch is already installed"""
    try:
        python_cmd = get_python_command()
        test_cmd = python_cmd + ["-c", "import torch; print(f'PyTorch {torch.__version__} installed')"]
        result = run_command(test_cmd, check=False)
        if result.returncode == 0:
            logger.info("✅ PyTorch already installed")
            return True
        else:
            return False
    except Exception:
        return False

def install_torch():
    """Install PyTorch with appropriate CUDA support"""
    try:
        if is_torch_installed():
            logger.info("✅ PyTorch already installed, skipping")
            return True
        
        # Detect CUDA version
        cuda_version = detect_cuda_version()
        
        # Get installation command
        install_packages = get_torch_install_command(cuda_version)
        
        # Install using pip
        pip_cmd = get_pip_command()
        install_cmd = pip_cmd + ['install'] + install_packages
        
        logger.info("Installing PyTorch (this may take a few minutes)...")
        result = run_command(install_cmd, show_progress=True, capture_output=False)
        
        if result.returncode == 0:
            logger.info("✅ PyTorch installation completed")
            
            # Verify installation
            python_cmd = get_python_command()
            verify_cmd = python_cmd + ["-c", """
import torch
print(f'✅ PyTorch {torch.__version__} verified')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.get_device_name(0)}')
"""]
            verify_result = run_command(verify_cmd, show_progress=True, capture_output=False)
            return verify_result.returncode == 0
        else:
            logger.error("❌ PyTorch installation failed")
            return False
            
    except Exception as e:
        logger.error(f"PyTorch installation failed: {e}")
        return False

def get_available_models_for_hardware(cuda_version=None):
    """Get list of available models based on hardware"""
    if cuda_version:
        # GPU available - all models
        return {
            "tiny": "Fastest, lowest quality (~39 MB)",
            "base": "Good balance of speed and quality (~74 MB)", 
            "small": "Better quality, slower (~244 MB)",
            "medium": "High quality, much slower (~769 MB)",
            "large-v2": "Highest quality, very slow (~1550 MB)",
            "large-v3": "Latest large model, very slow (~1550 MB)"
        }
    else:
        # CPU only - recommend smaller models
        return {
            "tiny": "Fastest, lowest quality (~39 MB) - Recommended for CPU",
            "base": "Good balance, usable on CPU (~74 MB)",
            "small": "Better quality, slow on CPU (~244 MB)",
        }

def interactive_model_selection(cuda_version=None):
    """Interactive model selection for download"""
    available_models = get_available_models_for_hardware(cuda_version)
    
    print("\n" + "="*60)
    print("🎙️  WHISPER MODEL SELECTION")
    print("="*60)
    
    if cuda_version:
        print(f"🎮 GPU detected (CUDA {cuda_version}) - All models available")
    else:
        print("💻 CPU-only mode - Smaller models recommended")
    
    print("\nAvailable models:")
    for i, (model, description) in enumerate(available_models.items(), 1):
        print(f"{i}. {model}: {description}")
    
    print(f"{len(available_models) + 1}. Skip model download (download later through web interface)")
    
    while True:
        try:
            choice = input(f"\nSelect models to download (1-{len(available_models) + 1}, or comma-separated for multiple): ").strip()
            
            if not choice:
                continue
                
            if choice == str(len(available_models) + 1):
                return []
            
            # Parse selection
            selected_indices = []
            for part in choice.split(','):
                try:
                    idx = int(part.strip())
                    if 1 <= idx <= len(available_models):
                        selected_indices.append(idx - 1)
                    else:
                        print(f"Invalid selection: {idx}")
                        break
                except ValueError:
                    print(f"Invalid input: {part}")
                    break
            else:
                # All selections valid
                model_names = list(available_models.keys())
                selected_models = [model_names[i] for i in selected_indices]
                
                if selected_models:
                    print(f"\nSelected models: {', '.join(selected_models)}")
                    confirm = input("Proceed with download? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        return selected_models
                    else:
                        continue
                else:
                    print("No models selected")
                    continue
            
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            return []

def is_model_already_downloaded(model_name):
    """Check if a model is already downloaded"""
    try:
        python_cmd = get_python_command()
        check_cmd = python_cmd + ["-c", f"""
import os
from faster_whisper import WhisperModel

# Check if model exists in cache
model_name = "{model_name}"
print(f"Checking for model: {{model_name}}")

try:
    # This will check if model is already downloaded
    # faster-whisper downloads to ~/.cache/huggingface/hub/
    import huggingface_hub
    
    # Try to find the model in cache
    cache_dir = huggingface_hub.constants.HF_HUB_CACHE
    model_dir_pattern = f"models--openai--whisper-{{model_name}}"
    
    for item in os.listdir(cache_dir):
        if model_dir_pattern in item:
            model_path = os.path.join(cache_dir, item)
            if os.path.isdir(model_path):
                # Check if model files exist
                snapshots_dir = os.path.join(model_path, "snapshots")
                if os.path.exists(snapshots_dir) and os.listdir(snapshots_dir):
                    print(f"✅ Model {{model_name}} found in cache")
                    exit(0)
    
    print(f"❌ Model {{model_name}} not found in cache")
    exit(1)
    
except Exception as e:
    print(f"❌ Error checking model cache: {{e}}")
    exit(1)
"""]
        
        result = run_command(check_cmd, check=False)
        if result.returncode == 0:
            logger.info(f"✅ Model {model_name} already downloaded")
            return True
        else:
            logger.info(f"Model {model_name} not found, will download")
            return False
            
    except Exception as e:
        logger.warning(f"Could not check if model {model_name} is downloaded: {e}")
        return False

def download_whisper_models(models_to_download=None):
    """Download Whisper models using faster-whisper"""
    if not models_to_download:
        logger.info("No models selected for download")
        return True
    
    logger.info(f"Downloading models: {', '.join(models_to_download)}")
    
    success = True
    for model_name in models_to_download:
        try:
            if is_model_already_downloaded(model_name):
                logger.info(f"✅ Model {model_name} already downloaded, skipping")
                continue
            
            logger.info(f"📥 Downloading model: {model_name}")
            logger.info("This may take several minutes depending on model size and your internet speed...")
            
            python_cmd = get_python_command()
            download_cmd = python_cmd + ["-c", f"""
from faster_whisper import WhisperModel
import sys

try:
    print(f"Initializing download for model: {model_name}")
    model = WhisperModel("{model_name}", device="cpu", compute_type="int8")
    print(f"✅ Model {model_name} downloaded and verified successfully")
except Exception as e:
    print(f"❌ Failed to download model {model_name}: {{e}}")
    sys.exit(1)
"""]
            
            result = run_command(download_cmd, show_progress=True, capture_output=False)
            
            if result.returncode == 0:
                logger.info(f"✅ Model {model_name} downloaded successfully")
            else:
                logger.error(f"❌ Failed to download model {model_name}")
                success = False
                
        except KeyboardInterrupt:
            logger.info(f"\nDownload of {model_name} cancelled by user")
            break
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            success = False
    
    if success:
        logger.info("✅ All selected models downloaded successfully")
    else:
        logger.warning("⚠️  Some models failed to download")
    
    return success

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible"""
    try:
        result = run_command(['ffmpeg', '-version'], check=False)
        if result.returncode == 0:
            # Extract version info
            version_line = result.stdout.split('\n')[0]
            logger.info(f"✅ {version_line}")
            return True
        else:
            logger.warning("❌ FFmpeg not found or not working")
            logger.info("📥 Install FFmpeg:")
            logger.info("   Ubuntu/Debian: sudo apt install ffmpeg")
            logger.info("   CentOS/RHEL: sudo yum install ffmpeg")
            logger.info("   Fedora: sudo dnf install ffmpeg")
            logger.info("   Arch: sudo pacman -S ffmpeg")
            return False
    except FileNotFoundError:
        logger.warning("❌ FFmpeg not found in PATH")
        logger.info("📥 Install FFmpeg:")
        logger.info("   Ubuntu/Debian: sudo apt install ffmpeg")
        logger.info("   CentOS/RHEL: sudo yum install ffmpeg")
        logger.info("   Fedora: sudo dnf install ffmpeg")
        logger.info("   Arch: sudo pacman -S ffmpeg")
        return False
    except Exception as e:
        logger.error(f"FFmpeg check failed: {e}")
        return False

def create_env_file():
    """Create .env file with default configuration"""
    env_file = Path(".env")
    
    if env_file.exists():
        logger.info("✅ .env file already exists")
        return
    
    logger.info("Creating .env configuration file...")
    
    env_content = """# =================================================================
# Video to Transcription Service v2.0 - Configuration
# =================================================================

# OpenAI Configuration (optional - for AI text processing features)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1    # gpt-4.1 is the default model, you can change it to any other model you want to use

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Processing Configuration
MAX_FILE_SIZE_MB=1000
CHUNK_DURATION=30

# =================================================================
# Advanced Configuration (usually not needed)
# =================================================================

# Force specific environment type (auto-detection is recommended)
# ENVIRONMENT=auto  # Options: auto, local, server

# Model override (auto-selection is recommended)
# FORCE_MODEL=large-v3  # Options: tiny, base, small, medium, large-v2, large-v3, turbo

# Hardware override (auto-detection is recommended)
# FORCE_DEVICE=auto  # Options: auto, cuda, cpu

# =================================================================
# Notes:
# =================================================================
# 
# 1. AI Features:
#    - Add your OpenAI API key to enable AI text processing
#    - Leave commented out to disable AI features
#
# 2. Hardware Detection:
#    - The system automatically detects your hardware capabilities
#    - GPU with 6GB+ VRAM: Uses large-v3 model for best accuracy
#    - GPU with 4-6GB VRAM: Uses medium model
#    - CPU only: Uses base or tiny model based on RAM
#
# 3. Model Selection:
#    - Local (GPU): Prioritizes accuracy with large-v3 model
#    - Server (CPU): Optimizes for efficiency with base/tiny models
#
# 4. File Size Limits:
#    - MAX_FILE_SIZE_MB: Maximum upload file size
#    - Large files automatically use smaller models on limited hardware
#
# ================================================================= 
"""
    
    try:
        env_file.write_text(env_content)
        logger.info("✅ .env file created successfully")
        logger.info("💡 Edit .env file to configure your OpenAI API key and other settings")
    except Exception as e:
        logger.error(f"Failed to create .env file: {e}")

def verify_virtual_environment():
    """Verify that the virtual environment is properly set up"""
    try:
        logger.info("Verifying virtual environment...")
        
        # Test virtual environment
        venv_python = get_venv_python()
        if not Path(venv_python).exists():
            logger.error("❌ Virtual environment Python not found")
            return False
            
        test_cmd = [venv_python, "-c", """
import sys
import os
print(f"🐍 Python executable: {sys.executable}")
print(f"🎯 Python version: {sys.version.split()[0]}")
print(f"📁 Virtual environment: {'venv' in sys.executable}")
print(f"📦 Package manager: pip (venv)")
print(f"🏠 Working directory: {os.getcwd()}")
if 'venv' in sys.executable:
    print(f"✅ Using virtual environment: {sys.executable}")
else:
    print(f"⚠️  Not in virtual environment: {sys.executable}")
"""]
        
        result = run_command(test_cmd, show_progress=True, capture_output=False)
        
        if result.returncode == 0:
            logger.info("✅ Virtual environment verification completed")
            return True
        else:
            logger.error("❌ Virtual environment verification failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Virtual environment verification error: {e}")
        return False

def check_system_requirements():
    """Check if system has required packages for Python virtual environments"""
    logger.info("Checking system requirements...")
    
    # Check if python3-venv is available
    try:
        test_result = subprocess.run([sys.executable, "-m", "venv", "--help"], capture_output=True, check=False)
        if test_result.returncode != 0:
            logger.error("❌ python3-venv module not available")
            logger.error("Install with: sudo apt install python3-venv python3-pip")
            return False
        else:
            logger.info("✅ python3-venv module available")
    except Exception as e:
        logger.error(f"Error checking venv module: {e}")
        return False
    
    # Check if we can write to current directory
    try:
        test_file = Path("test_write_permissions.tmp")
        test_file.write_text("test")
        test_file.unlink()
        logger.info("✅ Write permissions available")
    except Exception as e:
        logger.error(f"❌ Cannot write to current directory: {e}")
        logger.error("Check directory permissions")
        return False
    
    return True

def main():
    """Main installation process for Linux pip/venv"""
    logger.info("=== Video to Transcription Service - Linux pip/venv Installation ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("System requirements check failed")
        sys.exit(1)
    
    detect_package_manager()
    
    # Setup environment
    logger.info("\n=== STEP 1: Environment Setup ===")
    # Setup traditional virtual environment
    if not setup_venv():
        logger.error("Failed to setup virtual environment")
        sys.exit(1)
    
    # Upgrade pip in virtual environment
    logger.info("Upgrading pip in virtual environment...")
    pip_cmd = get_pip_command()  # This returns [python, -m, pip] to avoid permission issues
    
    # Upgrade pip, setuptools, and wheel using python -m pip
    upgrade_result = run_command(pip_cmd + ["install", "--upgrade", "pip", "setuptools", "wheel"], check=False)
    if upgrade_result.returncode == 0:
        logger.info("✅ pip, setuptools, and wheel upgraded successfully")
    else:
        logger.warning("⚠️  pip upgrade failed, continuing anyway")
        logger.warning("This may cause installation issues later")
    
    logger.info("✅ Virtual environment setup completed")
    
    # Verify virtual environment
    logger.info("\n=== STEP 2: Virtual Environment Verification ===")
    if not verify_virtual_environment():
        logger.error("Virtual environment verification failed")
        sys.exit(1)
    
    success = True
    
    # Step 3: Install base requirements
    logger.info("\n=== STEP 3: Installing Base Requirements ===")
    if not install_base_requirements():
        logger.error("❌ Base requirements installation failed")
        success = False
    else:
        logger.info("✅ Base requirements installed successfully")
    
    # Step 4: Install PyTorch with appropriate CUDA support
    logger.info("\n=== STEP 4: Installing PyTorch ===")
    if not install_torch():
        logger.error("❌ PyTorch installation failed")
        success = False
    else:
        logger.info("✅ PyTorch installed successfully")
    
    # Step 5: Verify installation before proceeding
    logger.info("\n=== STEP 5: Verifying Dependencies ===")
    try:
        python_cmd = get_python_command()
        test_cmd = python_cmd + ["-c", """
import torch
import faster_whisper
print('✅ All core dependencies verified')
print(f'🎮 PyTorch version: {torch.__version__}')
print(f'🎮 CUDA available: {torch.cuda.is_available()}')
print(f'🎙️  faster-whisper available: {faster_whisper.__version__}')
"""]
        result = run_command(test_cmd, show_progress=True, capture_output=False)
        if result.returncode == 0:
            logger.info("✅ All dependencies verified successfully")
        else:
            logger.error("❌ Dependency verification failed")
            success = False
    except Exception as e:
        logger.error(f"❌ Dependency verification failed: {e}")
        success = False
    
    # Only proceed with model selection if dependencies are installed
    if success:
        # Step 6: Check FFmpeg
        logger.info("\n=== STEP 6: Checking FFmpeg ===")
        if not check_ffmpeg():
            logger.warning("⚠️  FFmpeg check failed - you may need to install it manually")
        else:
            logger.info("✅ FFmpeg verified")
        
        # Step 7: Interactive model selection and download
        logger.info("\n=== STEP 7: Model Selection and Download ===")
        try:
            cuda_version = detect_cuda_version()
            models_to_download = interactive_model_selection(cuda_version)
            if models_to_download:
                download_whisper_models(models_to_download)
            else:
                logger.info("✅ Model selection completed (download skipped)")
        except KeyboardInterrupt:
            logger.info("\nModel selection cancelled. Models can be downloaded later through the web interface.")
        except Exception as e:
            logger.warning(f"Model selection/download failed: {e}")
            logger.info("Models will be downloaded automatically when first used")
    else:
        logger.error("❌ Cannot proceed with model selection due to dependency installation failures")
        logger.info("Please fix the above errors and run the installation again")
    
    # Step 8: Create .env file
    logger.info("\n=== STEP 8: Configuration Setup ===")
    create_env_file()
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("🎉 INSTALLATION SUMMARY")
    logger.info("=" * 60)
    if success:
        logger.info("✅ Dependencies installed successfully!")
        logger.info("✅ Python virtual environment created and configured")
        logger.info("✅ PyTorch installed with appropriate GPU/CPU support")
        logger.info("✅ All dependencies verified")
        logger.info("✅ Virtual environment properly configured")
        logger.info("\n🚀 READY TO RUN!")
        logger.info("You can now start the application with:")
        logger.info("  💻 Run: python3 or python3.11 app.py")
        logger.info("  🌐 Open: http://localhost:8000")
    else:
        logger.error("❌ Installation failed. Please check the logs above.")
        logger.info("You may need to install missing components manually.")
        logger.info("Common fixes:")
        logger.info("1. Check internet connection for package downloads")
        logger.info("2. Ensure sufficient disk space")
        logger.info("3. Try running: rm -rf venv && python3 setup.py or python3.11 setup.py depending on how many python versions you have installed")
    
    logger.info("\n💡 Additional Information:")
    logger.info("• Add OpenAI API key to .env file for AI features")
    logger.info("• Additional models can be downloaded through the web interface")
    logger.info("• All dependencies are isolated in the venv/ folder")
    
    # Environment info
    logger.info("\n=== ENVIRONMENT INFORMATION ===")
    logger.info("📦 Package Manager: pip (virtual environment) for Linux")
    logger.info(f"🐍 Python: {get_venv_python()}")
    logger.info("💡 Remember to activate the virtual environment before running commands, source venv/bin/activate")

if __name__ == "__main__":
    main() 