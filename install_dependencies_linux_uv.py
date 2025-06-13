#!/usr/bin/env python3
"""
Linux UV Dependency installation script for Video to Transcription Service
Automatically detects CUDA version and installs appropriate PyTorch
Uses UV for package management on Linux
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
    """Linux UV package manager detection"""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, check=True, text=True)
        logger.info(f"UV detected: {result.stdout.strip()}")
        return 'uv'
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("UV not found - this script requires UV")
        logger.info("Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
        logger.info("Or visit: https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)

def check_uv_python():
    """Check if Python 3.11 is available in UV"""
    try:
        # List available Python versions
        result = subprocess.run(['uv', 'python', 'list'], capture_output=True, check=True, text=True)
        logger.info("Checking UV Python installations...")
        
        # Check if 3.11 is available
        if '3.11' in result.stdout:
            logger.info("Python 3.11 found in UV")
            return True
        else:
            logger.info("Python 3.11 not found in UV, installing...")
            install_result = subprocess.run(['uv', 'python', 'install', '3.11'], check=True)
            if install_result.returncode == 0:
                logger.info("✅ Python 3.11 installed via UV")
                return True
            else:
                logger.error("Failed to install Python 3.11 via UV")
                return False
                
    except subprocess.CalledProcessError as e:
        logger.error(f"UV Python check failed: {e}")
        return False

def setup_uv_venv():
    """Setup UV virtual environment using modern UV commands"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        logger.info("✅ UV virtual environment already exists")
        # Verify it's working
        venv_python = get_uv_python()
        if Path(venv_python).exists():
            logger.info(f"🎯 Using existing UV environment: {venv_python}")
            # Verify UV can detect the environment
            test_result = run_command(['uv', 'pip', 'list'], check=False, capture_output=True)
            if test_result.returncode == 0:
                logger.info("✅ UV can detect existing virtual environment")
                return True
            else:
                logger.warning("⚠️  UV cannot detect existing virtual environment, recreating...")
                import shutil
                shutil.rmtree(venv_path)
        else:
            logger.warning("⚠️  Existing venv appears corrupted, recreating...")
            # Remove corrupted venv and recreate
            import shutil
            shutil.rmtree(venv_path)
    
    logger.info("Creating UV virtual environment with Python 3.11...")
    try:
        # Create UV venv with Python 3.11
        result = subprocess.run(['uv', 'venv', 'venv', '-p', '3.11'], check=True)
        logger.info("✅ UV virtual environment created successfully")
        
        # Verify UV can detect the new environment
        logger.info("🔍 Verifying UV can detect the virtual environment...")
        test_result = run_command(['uv', 'pip', 'list'], check=False, capture_output=True)
        if test_result.returncode == 0:
            logger.info("✅ UV virtual environment verification successful")
            return True
        else:
            logger.error("❌ UV cannot detect the virtual environment")
            logger.error("This may be a UV configuration issue")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create UV virtual environment: {e}")
        return False

def get_uv_python():
    """Get path to Python executable in UV venv for Linux"""
    return str(Path("venv") / "bin" / "python")

def get_python_command():
    """Get the appropriate Python command for UV"""
    # Use UV run for executing Python in UV environment
    return ['uv', 'run', 'python']

def run_command(command, check=True, capture_output=True, show_progress=False):
    """Run a command and return the result"""
    try:
        # Only log essential commands, not detailed scripts
        if not show_progress and not (isinstance(command, list) and len(command) > 3 and '-c' in command):
            logger.info(f"Running: {' '.join(command) if isinstance(command, list) else command}")
        
        # Set up environment for UV commands
        env = os.environ.copy()
        
        # For UV commands, ensure we're using the virtual environment
        if isinstance(command, list) and len(command) > 0 and command[0] == 'uv':
            # Make sure we have a venv directory
            venv_path = Path("venv")
            if venv_path.exists():
                # Set VIRTUAL_ENV to help UV detect the environment
                env['VIRTUAL_ENV'] = str(venv_path.absolute())
                # Ensure we're in the project directory
                env['UV_PROJECT_DIR'] = str(Path.cwd())
                # Add UV link mode for shared filesystems
                env['UV_LINK_MODE'] = 'copy'
        
        if show_progress:
            # For progress display, don't capture output - let it stream to console
            result = subprocess.run(
                command,
                check=check,
                text=True,
                shell=isinstance(command, str),
                env=env
            )
        else:
            # For silent commands, capture output as before
            result = subprocess.run(
                command,
                check=check,
                capture_output=capture_output,
                text=True,
                shell=isinstance(command, str),
                env=env
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
    """Check if base requirements are already installed using UV"""
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
    """Install base requirements using UV"""
    try:
        if are_base_requirements_installed():
            logger.info("✅ Base requirements already installed, skipping")
            return True
        
        logger.info("Installing base requirements...")
        
        # Install from requirements.txt using UV directly
        if Path("requirements.txt").exists():
            install_cmd = ['uv', 'pip', 'install', '-r', 'requirements.txt']
            logger.info("Installing from requirements.txt using UV...")
            result = run_command(install_cmd, show_progress=True, capture_output=False)
            
            if result.returncode == 0:
                logger.info("✅ Base requirements installed successfully")
                return True
            else:
                logger.error("❌ Failed to install base requirements from requirements.txt")
                # Check if it's a permission error and provide guidance
                if hasattr(result, 'stderr') and result.stderr and 'Operation not permitted' in result.stderr:
                    logger.error("🔧 This appears to be a filesystem permission issue.")
                    logger.error("💡 Common solutions:")
                    logger.error("   1. Move project to a native Linux filesystem (not shared folder)")
                    logger.error("   2. Or just remove UV and install Python 3.11 and then run: python3 setup.py")
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
        
        # Install using UV directly
        install_cmd = ['uv', 'pip', 'install'] + install_packages
        
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
            # Check if it's a permission error and provide guidance
            if hasattr(result, 'stderr') and result.stderr and 'Operation not permitted' in result.stderr:
                logger.error("🔧 This appears to be a filesystem permission issue.")
                logger.error("💡 Common solutions:")
                logger.error("   1. Move project to a native Linux filesystem (not shared folder)")
                logger.error("   2. Or just remove UV and install Python 3.11 and then run: python3 setup.py")
            return False
            
    except Exception as e:
        logger.error(f"PyTorch installation failed: {e}")
        return False

def get_available_models_for_hardware(cuda_version=None):
    """Get list of available models based on hardware - prioritize accuracy"""
    import psutil
    
    # Get system specs
    cpu_cores = psutil.cpu_count(logical=False) or 1
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    if cuda_version:
        # GPU available - always recommend Large-v3 for highest accuracy
        return {
            "large-v3": "Highest accuracy, latest model (~1550 MB) - RECOMMENDED",
            "large-v2": "Very high accuracy (~1550 MB)",
            "medium": "Good accuracy (~769 MB)",
            "small": "Decent accuracy (~244 MB)", 
            "base": "Good balance (~74 MB)",
            "tiny": "Basic accuracy (~39 MB)"
        }
    else:
        # CPU only - recommend highest accuracy model based on hardware capability
        if ram_gb >= 15 and cpu_cores >= 8:
            # High-end CPU - can handle large models, recommend Large-v3
            return {
                "large-v3": "Highest accuracy (~1550 MB) - RECOMMENDED",
                "large-v2": "Very high accuracy (~1550 MB)",
                "medium": "Good accuracy (~769 MB)",
                "small": "Decent accuracy (~244 MB)",
                "base": "Good balance (~74 MB)",
                "tiny": "Basic accuracy (~39 MB)"
            }
        elif ram_gb >= 7 and cpu_cores >= 4:
            # Mid-range CPU - recommend Medium as highest practical accuracy
            return {
                "medium": "Highest practical accuracy (~769 MB) - RECOMMENDED",
                "small": "Good accuracy (~244 MB)",
                "base": "Decent accuracy (~74 MB)",
                "tiny": "Basic accuracy (~39 MB)"
            }
        else:
            # Limited CPU - recommend Small as highest practical accuracy
            return {
                "small": "Highest practical accuracy (~244 MB) - RECOMMENDED",
                "base": "Good accuracy (~74 MB)",
                "tiny": "Basic accuracy (~39 MB)",
                "medium": "High accuracy, very slow (~769 MB)"
            }

def interactive_model_selection(cuda_version=None):
    """Interactive model selection for download"""
    import psutil
    
    available_models = get_available_models_for_hardware(cuda_version)
    
    print("\n" + "="*60)
    print("🎙️  WHISPER MODEL SELECTION")
    print("="*60)
    
    # Show hardware info
    cpu_cores = psutil.cpu_count(logical=False) or 1
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    if cuda_version:
        print(f"🎮 GPU detected (CUDA {cuda_version}) - All models available")
    else:
        print(f"💻 CPU-only mode - Hardware: {cpu_cores} cores, {ram_gb:.1f}GB RAM")
        if ram_gb >= 15 and cpu_cores >= 8:
            print("🚀 High-end CPU detected - Can handle larger models")
        elif ram_gb >= 7 and cpu_cores >= 4:
            print("⚡ Mid-range CPU detected - Good performance with smaller models")
        else:
            print("💡 Limited CPU detected - Smaller models recommended")
    
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
MAX_FILE_SIZE_MB=5000
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
        logger.info("Verifying UV virtual environment...")
        
        # Test UV environment
        test_cmd = ['uv', 'run', 'python', "-c", """
import sys
import os
print(f"🐍 Python executable: {sys.executable}")
print(f"🎯 Python version: {sys.version.split()[0]}")
print(f"📁 Virtual environment: {'venv' in sys.executable}")
print(f"📦 Package manager: UV")
print(f"🏠 Working directory: {os.getcwd()}")
if 'venv' in sys.executable:
    print(f"✅ Using UV virtual environment: {sys.executable}")
else:
    print(f"⚠️  Not in UV virtual environment: {sys.executable}")
"""]
        
        result = run_command(test_cmd, show_progress=True, capture_output=False)
        
        if result.returncode == 0:
            logger.info("✅ UV virtual environment verification completed")
            return True
        else:
            logger.error("❌ UV virtual environment verification failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Virtual environment verification error: {e}")
        return False

def check_filesystem_compatibility():
    """Check if we're on a filesystem that might have permission issues"""
    try:
        # Check if current directory is on a shared filesystem
        import subprocess
        df_result = subprocess.run(['df', '-T', '.'], capture_output=True, text=True)
        if df_result.returncode == 0 and df_result.stdout:
            filesystem_info = df_result.stdout
            # Common shared filesystem types that can have issues
            problematic_fs = ['vboxsf', 'vmhgfs', 'nfs', 'cifs', 'smb']
            for fs_type in problematic_fs:
                if fs_type in filesystem_info.lower():
                    logger.warning("⚠️  Detected shared/network filesystem")
                    logger.warning("💡 This may cause permission issues with UV installations")
                    logger.warning("🔧 Consider moving to native Linux filesystem if issues occur")
                    return False
    except Exception:
        pass
    return True

def main():
    """Main installation process for Linux UV"""
    logger.info("=== Video to Transcription Service - Linux UV Installation ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check filesystem compatibility
    check_filesystem_compatibility()
    
    # Ensure UV is available
    detect_package_manager()
    
    # Setup environment
    logger.info("\n=== STEP 1: Environment Setup ===")
    # Check and setup UV Python 3.11
    if not check_uv_python():
        logger.error("Failed to setup Python 3.11 in UV")
        logger.error("Please install UV and Python 3.11 manually:")
        logger.error("1. Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
        logger.error("2. Install Python 3.11: uv python install 3.11")
        sys.exit(1)
    
    # Setup UV virtual environment
    if not setup_uv_venv():
        logger.error("Failed to setup UV virtual environment")
        sys.exit(1)
    
    logger.info("✅ UV environment setup completed")
    
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
        logger.info("✅ UV environment with Python 3.11 configured")
        logger.info("✅ PyTorch installed with appropriate GPU/CPU support")
        logger.info("✅ All dependencies verified")
        logger.info("✅ Virtual environment properly configured")
        logger.info("\n🚀 READY TO RUN!")
        logger.info("You can now start the application with:")
        logger.info("  💻 Run: uv run app.py")
        logger.info("  🌐 Open: http://localhost:8000")
    else:
        logger.error("❌ Installation failed. Please check the logs above.")
        logger.info("You may need to install missing components manually.")
        logger.info("Common fixes:")
        logger.info("1. Check internet connection for package downloads")
        logger.info("2. Ensure sufficient disk space")
        logger.info("3. Try running: rm -rf venv && uv run setup.py")
    
    logger.info("\n💡 Additional Information:")
    logger.info("• Add OpenAI API key to .env file for AI features")
    logger.info("• Additional models can be downloaded through the web interface")
    logger.info("• All dependencies are isolated in the venv/ folder")
    
    # Environment info
    logger.info("\n=== ENVIRONMENT INFORMATION ===")
    logger.info("📦 Package Manager: UV with Python 3.11 (Linux)")
    logger.info("🐍 Python: Managed by UV in venv/ folder")
    logger.info("💡 Use 'uv run <script>' to run Python commands")

if __name__ == "__main__":
    main() 