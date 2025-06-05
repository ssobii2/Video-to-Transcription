#!/usr/bin/env python3
"""
Complete setup script for Video to Transcription Service v2.0
Handles installation, configuration, and initial setup
Supports UV for local development and venv for traditional Python
"""

import subprocess
import sys
import os
from pathlib import Path

def detect_package_manager():
    """Detect and recommend UV for best experience, fallback to venv"""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, check=True, text=True)
        print(f"✅ UV detected: {result.stdout.strip()}")
        return 'uv'
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  UV not found - UV is recommended for best experience")
        print("Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("Or visit: https://docs.astral.sh/uv/getting-started/installation/")
        print("ℹ️  Falling back to traditional Python venv...")
        return 'venv'

def get_uv_python():
    """Get path to Python executable in UV venv"""
    import platform
    if platform.system() == "Windows":
        return str(Path("venv") / "Scripts" / "python.exe")
    else:
        return str(Path("venv") / "bin" / "python")

def get_venv_python():
    """Get path to Python executable in virtual environment"""
    import platform
    if platform.system() == "Windows":
        return str(Path("venv") / "Scripts" / "python.exe")
    else:
        return str(Path("venv") / "bin" / "python")

def get_python_command():
    """Get the appropriate Python command"""
    pkg_manager = detect_package_manager()
    
    if pkg_manager == 'uv':
        # Use UV run for executing Python in UV environment
        return ['uv', 'run', 'python']
    else:
        # Use virtual environment if it exists
        venv_python = get_venv_python()
        if Path(venv_python).exists():
            return [venv_python]
        else:
            return [sys.executable]

def main():
    """Main setup process"""
    print("🚀 Video to Transcription Service v2.0 - Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ required. Current version:", sys.version)
        print("\nInstallation options:")
        print("- Local development: Install UV (https://docs.astral.sh/uv/) which includes Python 3.11")
        print("- Server deployment: Install Python 3.11 from https://python.org")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Detect package manager
    pkg_manager = detect_package_manager()
    
    # Step 1: Run dependency installation
    print("\n📦 Installing dependencies...")
    try:
        python_cmd = get_python_command()
        if pkg_manager == 'uv':
            result = subprocess.run(['uv', 'run', 'install_dependencies.py'], check=True)
        else:
            # For venv, we need to use the system Python to run install_dependencies.py
            # which will create the venv and install everything properly
            result = subprocess.run([sys.executable, "install_dependencies.py"], check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Dependency installation failed: {e}")
        print("Please run the installation script manually:")
        if pkg_manager == 'uv':
            print("  uv run install_dependencies.py")
        else:
            print("  python install_dependencies.py")
        sys.exit(1)
    
    # Update python_cmd after installation (venv may have been created)
    python_cmd = get_python_command()
    
    # Step 2: Create example environment file
    if not os.path.exists(".env"):
        print("\n⚙️ Creating .env configuration file...")
        with open(".env", "w") as f:
            f.write("# OpenAI Configuration (optional - for AI features)\n")
            f.write("# OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("# OPENAI_MODEL=gpt-3.5-turbo\n\n")
            f.write("# Server Configuration\n")
            f.write("HOST=0.0.0.0\n")
            f.write("PORT=8000\n")
            f.write("DEBUG=False\n\n")
            f.write("# Processing Configuration\n")
            f.write("MAX_FILE_SIZE_MB=500\n")
            f.write("CHUNK_DURATION=30\n")
        print("✅ .env file created")
        print("📝 Please edit .env file to add your OpenAI API key if you want AI features")
    else:
        print("✅ .env file already exists")
    
    # Step 3: Verify installation
    print("\n🔍 Verifying installation...")
    
    # Check PyTorch
    try:
        result = subprocess.run(
            python_cmd + ["-c", "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"],
            capture_output=True, text=True, check=True
        )
        print("✅ PyTorch:", result.stdout.strip().replace('\n', ', '))
    except Exception as e:
        print("❌ PyTorch verification failed:", e)
    
    # Check faster-whisper
    try:
        subprocess.run(python_cmd + ["-c", "import faster_whisper"], check=True)
        print("✅ faster-whisper installed")
    except Exception as e:
        print("❌ faster-whisper check failed:", e)
    
    # Check FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg installed")
    except:
        print("⚠️  FFmpeg not found - please install manually:")
        print("   Windows: Download from https://ffmpeg.org/")
        print("   Linux: sudo apt install ffmpeg")
        print("   macOS: brew install ffmpeg")
    
    # Step 4: Test basic functionality
    print("\n🧪 Testing basic functionality...")
    try:
        result = subprocess.run(
            python_cmd + ["-c", """
from config import Config
config = Config()
downloaded_models = config.get_downloaded_models_info()
print(f"Environment: {config.environment.value}")
print(f"Selected Model: {config.model_config.model_size.value}")
print(f"Device: {config.model_config.device}")
print(f"Downloaded Models: {downloaded_models}")
"""], capture_output=True, text=True, check=True)
        
        print("✅ Configuration test passed:")
        for line in result.stdout.strip().split('\n'):
            print(f"   {line}")
            
    except Exception as e:
        print("❌ Configuration test failed:", e)
        if hasattr(e, 'stderr') and e.stderr:
            print("Error details:", e.stderr)
    
    # Final instructions
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("Next steps:")
    
    if pkg_manager == 'uv':
        print("1. 🌐 Start the application: uv run python app.py")
        print()
        print("   🌐 Then open your browser to: http://localhost:8000")
        print()
    else:
        print("1. 🌐 Start the application: python app.py")
        print()
        print("   🌐 Then open your browser to: http://localhost:8000")
        print()
    
    print("2. 📁 Upload video/audio files for transcription")
    print("3. 🤖 Download additional models through the web interface")
    print()
    print("Optional:")
    print("• Add OpenAI API key to .env file for AI features")
    print("• Install FFmpeg if not already available")
    print()
    
    # Environment summary
    if pkg_manager == 'uv':
        print("📦 Environment: UV (isolated and managed)")
    else:
        print("📦 Environment: Python virtual environment (venv)")
        print("💡 Remember to activate the virtual environment before running commands")
    
    print("\nFor help, see the README.md file")

if __name__ == "__main__":
    main() 