#!/usr/bin/env python3
"""
Dependency installation script for Video to Transcription Service
Automatically detects CUDA version and installs appropriate PyTorch
Supports UV for local development and venv for traditional Python
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
    """Detect and recommend UV for best experience, fallback to venv"""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, check=True, text=True)
        logger.info(f"UV detected: {result.stdout.strip()}")
        return 'uv'
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("UV not found - UV is recommended for best experience")
        logger.info("Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
        logger.info("Or visit: https://docs.astral.sh/uv/getting-started/installation/")
        logger.info("Falling back to traditional Python venv...")
        return 'venv'

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
            return True
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
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create UV virtual environment: {e}")
        return False

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
    try:
        # Create virtual environment
        result = subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        logger.info("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        return False

def get_uv_python():
    """Get path to Python executable in UV venv"""
    if platform.system() == "Windows":
        return str(Path("venv") / "Scripts" / "python.exe")
    else:
        return str(Path("venv") / "bin" / "python")

def get_venv_python():
    """Get path to Python executable in virtual environment"""
    if platform.system() == "Windows":
        return str(Path("venv") / "Scripts" / "python.exe")
    else:
        return str(Path("venv") / "bin" / "python")

def get_venv_pip():
    """Get path to pip executable in virtual environment"""
    if platform.system() == "Windows":
        return str(Path("venv") / "Scripts" / "pip.exe")
    else:
        return str(Path("venv") / "bin" / "pip")

def get_python_command():
    """Get the appropriate Python command"""
    pkg_manager = detect_package_manager()
    
    if pkg_manager == 'uv':
        # Use UV run for executing Python in UV environment
        return ['uv', 'run', 'python']
    else:
        # Use virtual environment
        venv_python = get_venv_python()
        if Path(venv_python).exists():
            logger.info("Using virtual environment Python")
            return [venv_python]
        else:
            logger.warning("Virtual environment not found, using system Python")
            return [sys.executable]

def run_command(command, check=True, capture_output=True, show_progress=False):
    """Run a command and return the result"""
    try:
        # Only log essential commands, not detailed scripts
        if not show_progress and not (isinstance(command, list) and len(command) > 3 and '-c' in command):
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
        logger.error(f"Python 3.11+ required. Current version: {sys.version}")
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
                cuda_match = re.search(r'release (\d+\.\d+)', nvcc_result.stdout)
                if cuda_match:
                    cuda_version = cuda_match.group(1)
                    logger.info(f"CUDA version detected: {cuda_version}")
                    return cuda_version
            
            # Fallback: try to detect from nvidia-smi
            smi_result = run_command(['nvidia-smi'], check=False)
            if smi_result.returncode == 0:
                cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', smi_result.stdout)
                if cuda_match:
                    cuda_version = cuda_match.group(1)
                    logger.info(f"CUDA version detected from nvidia-smi: {cuda_version}")
                    return cuda_version
                
                # If we can't detect specific version but GPU is there, assume modern CUDA
                logger.info("GPU detected but CUDA version unclear, assuming CUDA 11.8+")
                return "11.8"
        
        logger.info("No NVIDIA GPU detected")
        return None
        
    except Exception as e:
        logger.warning(f"Error detecting CUDA: {e}")
        return None

def get_torch_install_command(cuda_version=None):
    """Get the appropriate PyTorch installation command"""
    pkg_manager = detect_package_manager()
    
    if cuda_version is None:
        # CPU-only installation
        logger.info("Installing PyTorch for CPU")
        if pkg_manager == 'uv':
            # Use UV pip install with correct format
            return ['uv', 'pip', 'install',
                    'torch', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu']
        else:
            pip_cmd = get_venv_pip()
            return [pip_cmd, "install", 
                    "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]
    
    # GPU installation
    cuda_major = float(cuda_version)
    
    if cuda_major >= 12.1:
        index_url = "https://download.pytorch.org/whl/cu121"
        logger.info("Installing PyTorch for CUDA 12.1+")
    elif cuda_major >= 11.8:
        index_url = "https://download.pytorch.org/whl/cu118"
        logger.info("Installing PyTorch for CUDA 11.8+")
    elif cuda_major >= 11.7:
        index_url = "https://download.pytorch.org/whl/cu117"
        logger.info("Installing PyTorch for CUDA 11.7")
    else:
        # Fallback to CPU for very old CUDA versions
        logger.warning(f"CUDA {cuda_version} is too old, installing CPU version")
        return get_torch_install_command(None)
    
    if pkg_manager == 'uv':
        # Use UV pip install with correct format
        return ['uv', 'pip', 'install',
                "torch", "torchaudio", "--index-url", index_url]
    else:
        pip_cmd = get_venv_pip()
        return [pip_cmd, "install",
                "torch", "torchaudio", "--index-url", index_url]

def are_base_requirements_installed():
    """Check if base requirements are already installed in the virtual environment"""
    if not os.path.exists("requirements.txt"):
        return True  # No requirements to check
    
    # Read requirements and filter out torch
    requirements = []
    with open("requirements.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.lower().startswith("torch"):
                # Extract package name (before any version specifiers)
                package = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].split('~=')[0].split('!=')[0]
                requirements.append(package.strip())
    
    if not requirements:
        return True
    
    try:
        python_cmd = get_python_command()
        # Create a test script to check all packages
        check_script = "import sys; packages = " + str(requirements) + """; 
missing = []
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"Missing packages: {missing}")
    print("requirements_missing=True")
else:
    print("All base requirements are installed")
    print("requirements_installed=True")
"""
        
        test_cmd = python_cmd + ["-c", check_script]
        result = run_command(test_cmd, capture_output=True)
        
        if result.returncode == 0 and "requirements_installed=True" in result.stdout:
            return True
        
    except Exception:
        pass
    
    return False

def install_base_requirements():
    """Install base requirements from requirements.txt"""
    # Check if base requirements are already installed
    if are_base_requirements_installed():
        logger.info("✅ Base requirements are already installed")
        # Show what's installed
        try:
            python_cmd = get_python_command()
            info_cmd = python_cmd + ["-c", """
import pkg_resources
installed = [d.project_name for d in pkg_resources.working_set]
print(f"📦 {len(installed)} packages already installed in virtual environment")
print("Some installed packages:", ", ".join(sorted(installed)[:10]))
"""]
            run_command(info_cmd, show_progress=True, capture_output=False)
        except Exception:
            pass
        return True
    
    logger.info("Installing base requirements...")
    pkg_manager = detect_package_manager()
    
    # Read requirements.txt and filter out torch
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.lower().startswith("torch"):
                    requirements.append(line)
    
    if requirements:
        logger.info(f"📦 Installing {len(requirements)} base packages...")
        logger.info(f"Packages: {', '.join(requirements[:5])}{'...' if len(requirements) > 5 else ''}")
        
        if pkg_manager == 'uv':
            # Use UV pip install with correct format
            venv_python = get_uv_python()
            logger.info(f"🎯 Target environment: {venv_python}")
            
            with open("temp_requirements.txt", "w") as f:
                f.write("\n".join(requirements))
            
            logger.info("📥 Installing packages (this may take a while for large packages)...")
            result = run_command(['uv', 'pip', 'install', '-r', 'temp_requirements.txt'], 
                               show_progress=True, capture_output=False)
        else:
            # Use virtual environment pip and show progress
            pip_cmd = get_venv_pip()
            logger.info(f"🎯 Target environment: {pip_cmd}")
            
            with open("temp_requirements.txt", "w") as f:
                f.write("\n".join(requirements))
            
            logger.info("📥 Installing packages (this may take a while for large packages)...")
            result = run_command([pip_cmd, "install", "-r", "temp_requirements.txt"], 
                               show_progress=True, capture_output=False)
        
        # Clean up
        if os.path.exists("temp_requirements.txt"):
            os.remove("temp_requirements.txt")
        
        if result.returncode != 0:
            logger.error("Failed to install base requirements")
            return False
        
        logger.info("✅ Base requirements installation completed")
    
    return True

def is_torch_installed():
    """Check if PyTorch is already installed and working in the virtual environment"""
    try:
        python_cmd = get_python_command()
        test_cmd = python_cmd + ["-c", """
import torch
print(f"PyTorch {torch.__version__} is already installed")
print(f"CUDA available: {torch.cuda.is_available()}")
print("torch_installed=True")
"""]
        result = run_command(test_cmd, capture_output=True)
        if result.returncode == 0 and "torch_installed=True" in result.stdout:
            return True
    except Exception:
        pass
    return False

def install_torch():
    """Install appropriate PyTorch version"""
    # Check if PyTorch is already installed
    if is_torch_installed():
        logger.info("✅ PyTorch is already installed and working")
        python_cmd = get_python_command()
        # Show current installation info
        info_cmd = python_cmd + ["-c", """
import torch
print(f"🎮 Current PyTorch version: {torch.__version__}")
print(f"🎮 CUDA available: {torch.cuda.is_available()}")
print(f"🎯 Installation location: {torch.__file__}")
"""]
        run_command(info_cmd, show_progress=True, capture_output=False)
        return True
    
    logger.info("Installing PyTorch...")
    
    # Detect CUDA
    cuda_version = detect_cuda_version()
    
    # Get installation command
    install_cmd = get_torch_install_command(cuda_version)
    
    pkg_manager = detect_package_manager()
    
    if cuda_version:
        logger.info(f"🎮 GPU Environment detected - CUDA {cuda_version}")
        logger.info("📦 Installing PyTorch with CUDA support (this will take several minutes)")
    else:
        logger.info("💻 CPU Environment detected")
        logger.info("📦 Installing PyTorch CPU version (this will take several minutes)")
    
    if pkg_manager == 'uv':
        venv_python = get_uv_python()
        logger.info(f"🎯 Target environment: {venv_python}")
    else:
        pip_cmd = get_venv_pip()
        logger.info(f"🎯 Target environment: {pip_cmd}")
    
    logger.info("📥 Downloading and installing PyTorch (large download, please wait)...")
    logger.info("💡 This may appear frozen but PyTorch is downloading in the background")
    
    # Install PyTorch with progress display
    result = run_command(install_cmd, show_progress=True, capture_output=False)
    
    if result.returncode != 0:
        logger.error("Failed to install PyTorch")
        return False
    
    logger.info("✅ PyTorch installation completed")
    
    # Verify installation in the virtual environment
    logger.info("🔍 Verifying PyTorch installation...")
    try:
        python_cmd = get_python_command()
        test_cmd = python_cmd + ["-c", "import torch; print(f'✅ PyTorch {torch.__version__} installed successfully'); print(f'🎮 CUDA available: {torch.cuda.is_available()}'); print(f'🎯 Installation location: {torch.__file__}')"]
        result = run_command(test_cmd, show_progress=True, capture_output=False)
        if result.returncode == 0:
            logger.info("✅ PyTorch verification completed")
            if cuda_version:
                logger.info("🎮 GPU support should be available")
            return True
    except Exception as e:
        logger.warning(f"Could not verify PyTorch installation: {e}")
    
    return False

def get_available_models_for_hardware(cuda_version=None):
    """Get available models based on hardware"""
    if cuda_version:
        # Local environment with GPU
        return {
            "large-v3": {"name": "Large-v3", "size": "~3GB", "description": "Highest accuracy (recommended)", "recommended": True},
            "large-v2": {"name": "Large-v2", "size": "~3GB", "description": "High accuracy"},
            "turbo": {"name": "Turbo", "size": "~1.5GB", "description": "Fast with good accuracy"},
            "medium": {"name": "Medium", "size": "~1.5GB", "description": "Balanced speed/accuracy"},
            "base": {"name": "Base", "size": "~140MB", "description": "Fast, good accuracy"},
            "small": {"name": "Small", "size": "~240MB", "description": "Very fast"},
            "tiny": {"name": "Tiny", "size": "~40MB", "description": "Fastest, lower accuracy"}
        }
    else:
        # Server environment - CPU only
        return {
            "base": {"name": "Base", "size": "~140MB", "description": "Best for CPU (recommended)", "recommended": True},
            "tiny": {"name": "Tiny", "size": "~40MB", "description": "Fastest for limited resources"}
        }

def interactive_model_selection(cuda_version=None):
    """Interactive model selection during setup"""
    models = get_available_models_for_hardware(cuda_version)
    
    print("\n" + "=" * 50)
    print("MODEL SELECTION")
    print("=" * 50)
    
    if cuda_version:
        print("🎮 GPU detected! You can use larger models for better accuracy.")
    else:
        print("💻 CPU-only environment detected. Optimized models available.")
    
    print("\nAvailable models:")
    
    model_list = list(models.keys())
    for i, (model_key, model_info) in enumerate(models.items(), 1):
        recommended = " (RECOMMENDED)" if model_info.get("recommended") else ""
        print(f"{i}. {model_info['name']} - {model_info['size']} - {model_info['description']}{recommended}")
    
    print(f"{len(model_list) + 1}. Download all models")
    print(f"{len(model_list) + 2}. Skip model download (download during first use)")
    
    while True:
        try:
            choice = input(f"\nSelect models to download (1-{len(model_list) + 2}) or multiple (e.g., 1,3,5): ").strip()
            
            if choice == str(len(model_list) + 1):
                # Download all models
                return list(model_list)
            elif choice == str(len(model_list) + 2):
                # Skip download
                return []
            else:
                # Parse multiple selections
                selections = [int(x.strip()) for x in choice.split(',')]
                selected_models = []
                
                for sel in selections:
                    if 1 <= sel <= len(model_list):
                        selected_models.append(model_list[sel - 1])
                    else:
                        raise ValueError("Invalid selection")
                
                return selected_models
                
        except (ValueError, IndexError):
            print("❌ Invalid selection. Please try again.")

def is_model_already_downloaded(model_name):
    """Check if a specific Whisper model is already downloaded and cached"""
    try:
        python_cmd = get_python_command()
        check_cmd = python_cmd + ["-c", f"""
import sys
import os
from pathlib import Path

try:
    # Check HuggingFace cache directories where faster-whisper stores models
    # This is the same location the main application uses
    
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "whisper"  # Alternative cache location
    ]
    
    model_found = False
    model_name = "{model_name}"
    
    print(f"Checking cache directories for {model_name} model...")
    
    for cache_dir in cache_dirs:
        if cache_dir and cache_dir.exists():
            # Look for model directories that match the pattern
            # faster-whisper models are stored with specific naming patterns
            pattern_matches = list(cache_dir.glob(f"*whisper*{model_name}*"))
            pattern_matches.extend(list(cache_dir.glob(f"*{model_name}*whisper*")))
            pattern_matches.extend(list(cache_dir.glob(f"models--Systran--faster-whisper-{model_name}*")))
            pattern_matches.extend(list(cache_dir.glob(f"models--openai--whisper-{model_name}*")))
            
            if pattern_matches:
                print(f"Found {model_name} model cache in: {{cache_dir}}")
                model_found = True
                break
    
    if model_found:
        print(f"✅ Model '{model_name}' is cached and available")
        print("model_exists=True")
    else:
        print(f"Model '{model_name}' not found in cache - needs download")
        print("model_exists=False")
        
except Exception as e:
    print(f"Cache check failed: {{e}}")
    print("model_exists=False")
"""]
        
        result = run_command(check_cmd, capture_output=True)
        if result.returncode == 0 and "model_exists=True" in result.stdout:
            return True
            
    except Exception as e:
        logger.debug(f"Model check exception: {e}")
    
    return False

def download_whisper_models(models_to_download=None):
    """Pre-download Whisper models to avoid delays during first use"""
    if not models_to_download:
        logger.info("Skipping model download")
        return
    
    # Filter out models that are already downloaded
    models_needed = []
    models_already_cached = []
    
    logger.info(f"🔍 Checking which models are already downloaded...")
    for model in models_to_download:
        if is_model_already_downloaded(model):
            models_already_cached.append(model)
            logger.info(f"✅ {model} model is already cached")
        else:
            models_needed.append(model)
            logger.info(f"📥 {model} model needs to be downloaded")
    
    if models_already_cached:
        logger.info(f"✅ {len(models_already_cached)} models already cached: {', '.join(models_already_cached)}")
    
    if not models_needed:
        logger.info("✅ All requested models are already downloaded and cached")
        return
    
    logger.info(f"📥 Need to download {len(models_needed)} models: {', '.join(models_needed)}")
    logger.info("💡 Models will be cached in your virtual environment for faster access")
    
    python_cmd = get_python_command()
    
    # Convert models_needed to string representation for the script
    models_list_str = repr(models_needed)
    
    download_script = f'''
import os
from faster_whisper import WhisperModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

models = {models_list_str}

print("Target: Virtual environment location:", os.sys.executable)
print("Starting model downloads...")
print("Info: Models will be cached using faster-whisper's default cache location\\n")

for i, model_name in enumerate(models, 1):
    try:
        print(f"Download [{{i}}/{{len(models)}}] Downloading {{model_name}} model...")
        print(f"Info: This may take several minutes depending on your internet connection")
        print(f"Cache: Model will be cached in HuggingFace cache directory")
        
        # This will download the model if not present and cache it
        # Uses the same caching mechanism as the main application
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print(f"Success: {{model_name}} model downloaded and cached successfully")
        print(f"Cache: Model is now available for future use\\n")
        del model  # Free memory
    except Exception as e:
        print(f"Error: Failed to download {{model_name}} model: {{e}}")
        print(f"Info: {{model_name}} will be downloaded automatically when first used\\n")

print("Complete: Model download process completed!")
print("Info: All models are now cached and ready for use")
'''
    
    # Write temporary script with UTF-8 encoding
    with open("download_models.py", "w", encoding="utf-8") as f:
        f.write(download_script)
    
    try:
        # Run the download script with progress display
        download_cmd = python_cmd + ["download_models.py"]
        logger.info("🚀 Starting model download process...")
        logger.info("💡 Each model download will show individual progress")
        
        result = run_command(download_cmd, show_progress=True, capture_output=False)
        
        if result.returncode == 0:
            logger.info("✅ Model download process completed successfully")
        else:
            logger.warning("⚠️  Some models may not have downloaded correctly")
            logger.info("💡 Missing models will be downloaded automatically when first used")
    
    finally:
        # Clean up
        if os.path.exists("download_models.py"):
            os.remove("download_models.py")

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    logger.info("Checking FFmpeg installation...")
    
    ffmpeg_ok = False
    ffprobe_ok = False
    
    # Check FFmpeg
    result = run_command(['ffmpeg', '-version'], check=False)
    if result.returncode == 0:
        logger.info("FFmpeg is installed")
        ffmpeg_ok = True
    else:
        logger.warning("FFmpeg not found")
    
    # Check FFprobe
    result = run_command(['ffprobe', '-version'], check=False)
    if result.returncode == 0:
        logger.info("FFprobe is installed")
        ffprobe_ok = True
    else:
        logger.warning("FFprobe not found")
    
    if not (ffmpeg_ok and ffprobe_ok):
        logger.warning("\nFFmpeg/FFprobe installation required:")
        logger.warning("- Windows: Download from https://ffmpeg.org/")
        logger.warning("- Linux: sudo apt install ffmpeg")
        logger.warning("- macOS: brew install ffmpeg")
    
    return ffmpeg_ok and ffprobe_ok

def create_env_file():
    """Create .env file from example if it doesn't exist"""
    if os.path.exists(".env"):
        logger.info("✅ .env file already exists")
        logger.info("💡 Your existing configuration will be preserved")
        # Show some info about existing config
        try:
            with open(".env", "r") as f:
                lines = f.readlines()
            logger.info(f"📄 Current .env file has {len(lines)} lines")
        except Exception:
            pass
        return
        
    logger.info("📝 Creating .env file...")
    env_content = """# OpenAI Configuration (optional - for AI features)
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_MODEL=gpt-4.1

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
#    - Turbo model available for manual selection when GPU has 6GB+ VRAM
#
# 4. File Size Limits:
#    - MAX_FILE_SIZE_MB: Maximum upload file size
#    - Large files automatically use smaller models on limited hardware
#
# ================================================================= 
"""
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        logger.info("✅ .env file created successfully")
        logger.info("💡 Please edit .env file to add your OpenAI API key if needed")
    except Exception as e:
        logger.warning(f"Failed to create .env file: {e}")
        logger.info("You can create it manually later")

def verify_virtual_environment():
    """Verify that we're using the virtual environment correctly"""
    logger.info("🔍 Verifying virtual environment setup...")
    
    pkg_manager = detect_package_manager()
    python_cmd = get_python_command()
    
    try:
        # Check Python executable location
        if pkg_manager == 'uv':
            test_cmd = python_cmd + ["-c", """
import sys
import os
print(f"🐍 Python executable: {sys.executable}")
print(f"🎯 Python version: {sys.version.split()[0]}")
print(f"📁 Virtual environment: {'venv' in sys.executable or 'UV_PROJECT_DIR' in os.environ}")
print(f"📦 Package manager: UV")
print(f"🏠 Working directory: {os.getcwd()}")
if 'venv' in sys.executable:
    print(f"✅ Using virtual environment: {sys.executable}")
else:
    print(f"⚠️  Not in virtual environment: {sys.executable}")
"""]
        else:
            venv_python = get_venv_python()
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

def main():
    """Main installation process"""
    logger.info("=== Video to Transcription Service - Dependency Installation ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    pkg_manager = detect_package_manager()
    
    # Setup environment based on package manager
    logger.info("\n=== STEP 1: Environment Setup ===")
    if pkg_manager == 'uv':
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
            
    elif pkg_manager == 'venv':
        # Setup traditional virtual environment
        if not setup_venv():
            logger.error("Failed to setup virtual environment")
            sys.exit(1)
        
        # Upgrade pip in virtual environment
        logger.info("Upgrading pip in virtual environment...")
        pip_cmd = get_venv_pip()
        upgrade_result = run_command([pip_cmd, "install", "--upgrade", "pip"], check=False)
        if upgrade_result.returncode == 0:
            logger.info("✅ pip upgraded successfully")
        else:
            logger.warning("⚠️  pip upgrade failed, continuing anyway")
        
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
        
        if pkg_manager == 'uv':
            logger.info("✅ UV environment with Python 3.11 configured")
        else:
            logger.info("✅ Python virtual environment created and configured")
            
        logger.info("✅ PyTorch installed with appropriate GPU/CPU support")
        logger.info("✅ All dependencies verified")
        logger.info("✅ Virtual environment properly configured")
        logger.info("\n🚀 READY TO RUN!")
        logger.info("You can now start the application with:")
        
        if pkg_manager == 'uv':
            logger.info("  💻 Run: uv run python app.py")
        else:
            logger.info("  💻 Run: python app.py")
        logger.info("  🌐 Open: http://localhost:8000")
    else:
        logger.error("❌ Installation failed. Please check the logs above.")
        logger.info("You may need to install missing components manually.")
        logger.info("Common fixes:")
        logger.info("1. Check internet connection for package downloads")
        logger.info("2. Ensure sufficient disk space")
        logger.info("3. Try running: rm -rf venv && python setup.py")
    
    logger.info("\n💡 Additional Information:")
    logger.info("• Add OpenAI API key to .env file for AI features")
    logger.info("• Additional models can be downloaded through the web interface")
    logger.info("• All dependencies are isolated in the venv/ folder")
    
    # Environment info
    logger.info("\n=== ENVIRONMENT INFORMATION ===")
    if pkg_manager == 'uv':
        logger.info("📦 Package Manager: UV with Python 3.11 (recommended)")
        logger.info("🐍 Python: Managed by UV in venv/ folder")
        logger.info("💡 Use 'uv run <script>' to run Python commands")
    else:
        logger.info("📦 Package Manager: pip (virtual environment)")
        logger.info(f"🐍 Python: {get_venv_python()}")
        logger.info("💡 Remember to activate the virtual environment before running commands")

if __name__ == "__main__":
    main() 