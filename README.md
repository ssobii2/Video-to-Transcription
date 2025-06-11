# Video/Audio to Transcription Service v2.0

An intelligent video and audio transcription service with adaptive model selection, GPU acceleration, and AI-powered text processing. Designed to work optimally on both local machines (with GPU) and resource-limited servers (CPU-only).

**✨ Environment Isolation**: Recommended UV environment for modern Python management, with virtual environment (venv) fallback for traditional setups.

## 🚀 Key Features

### Adaptive Model Selection

- **Local Environment (GPU)**: Automatically uses Large-v3 model for maximum accuracy
- **Server Environment (CPU)**: Optimizes for Base or Tiny models based on available resources
- **Hardware Detection**: Automatically detects GPU memory, CPU cores, and RAM to select optimal configuration
- **Multiple Model Sizes**: Support for tiny, base, small, medium, large-v2, and large-v3 models

### Intelligent Processing

- **Dynamic Model Loading**: Models are loaded on-demand and cached for efficiency
- **File Size Optimization**: Automatically adjusts model size for large files on limited hardware
- **Progress Tracking**: Real-time updates via WebSocket for all processing stages
- **Error Recovery**: Robust error handling with cleanup on failures

### Multi-Format Support

- **Video**: MP4, AVI, MKV, MOV, WMV, FLV, WebM, MPEG, MPG, 3GP
- **Audio**: MP3, WAV, OGG, M4A, AAC, FLAC, WMA
- **GPU Acceleration**: CUDA acceleration for video conversion when available

### AI-Powered Text Processing

- **Custom Prompts**: Process transcriptions with custom AI prompts
- **Suggested Templates**: Pre-built prompts for common use cases (summaries, meeting minutes, etc.)
- **Multiple AI Models**: Support for different OpenAI models

### Modern Environment Management

- **UV Environment (Recommended)**: Modern Python 3.11 management with UV
- **Virtual Environment (Fallback)**: Traditional Python virtual environments for compatibility
- **No System Pollution**: All dependencies installed in isolated `venv/` folder
- **Easy Cleanup**: Complete environment isolation allows easy removal

## 📋 Requirements

### System Requirements

- **FFmpeg** and **FFprobe** (for media processing)
- **UV (Recommended)** or **Python 3.11**
- **CUDA** (optional, for GPU acceleration)

### Hardware Recommendations

#### Local Development (GPU)

- **GPU**: 4GB+ VRAM (6GB+ recommended for large models)
- **RAM**: 8GB+ system RAM
- **CPU**: 4+ cores

#### Server Deployment (CPU)

- **RAM**: 4GB+ (8GB+ recommended)
- **CPU**: 2+ cores (4+ recommended)
- **Storage**: 1GB+ free space for models and processing

## 🛠 Installation

### Quick Setup (Recommended)

The easiest way to get started is with our automated setup script:

```bash
git clone <repository-url>
cd Video-to-Transcription
python setup.py
```

**Note:** Use `python3` or `python3.11` if you have multiple Python versions installed.

## 🤖 What's Automatically Handled

Our setup scripts automatically take care of many steps that you would normally do manually:

### ✅ **Environment Detection & Setup**

- **Platform Detection**: Automatically detects Windows/Linux and chooses appropriate installation method
- **Package Manager Detection**: Detects UV (modern) vs pip (traditional) and uses the best available option
- **Python Version Management**: With UV, automatically installs Python 3.11 if needed
- **Virtual Environment Creation**: Creates isolated `venv/` folder automatically

### ✅ **Dependency Management**

- **PyTorch Installation**: Automatically detects CUDA version and installs appropriate PyTorch (GPU/CPU)
- **Hardware-Optimized Setup**: Configures dependencies based on your GPU memory and CPU capabilities
- **Requirements Installation**: Installs all Python packages in the isolated environment
- **Version Compatibility**: Ensures all packages work together correctly

### ✅ **Configuration**

- **`.env` File Creation**: Automatically creates configuration file with sensible defaults
- **Environment Variables**: Pre-configures server settings, file size limits, and processing options
- **OpenAI Integration**: Sets up AI features (just add your API key)

### ✅ **Verification & Testing**

- **Installation Verification**: Tests all critical components (PyTorch, faster-whisper, etc.)
- **Hardware Detection**: Verifies GPU/CUDA availability
- **Import Testing**: Ensures all packages can be imported correctly

### ✅ **Model Management**

- **On-Demand Downloads**: Whisper models download automatically when first used
- **Smart Storage**: Models cached in environment-specific locations
- **Hardware-Aware Selection**: Automatically chooses optimal model size for your hardware

### 📝 **What You Still Need to Do**

- **Install FFmpeg**: System-level dependency (instructions provided)
- **Add OpenAI API Key**: Edit `.env` file if you want AI features
- **Review Configuration**: Optionally customize settings in `.env` file

This automation means you can go from "git clone" to "working application" with just one command!

---

### Advanced Setup Options

#### Option 1: UV (Recommended)

UV provides the best Python management experience:

```bash
# Install UV first
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone <repository-url>
cd Video-to-Transcription
python setup.py
```

**UV Benefits:**

- **Automatic Python 3.11** installation and management
- **Faster dependency resolution** and installation
- **Consistent environments** across different systems
- **Modern workflow** with proper isolation
- **No version conflicts** with system Python

#### Option 2: Traditional Python with Virtual Environment

For compatibility when UV is not available:

```bash
# Clone repository
git clone <repository-url>
cd Video-to-Transcription

# Run setup (will automatically create venv)
python setup.py
```

**Note:** Use `python3` or `python3.11` if you have multiple Python versions installed.

### Manual Installation (Advanced Users)

If you prefer complete manual control:

1. **Install UV and Python 3.11 (Recommended)**

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.11
uv python install 3.11
```

2. **Or install Python 3.11 traditionally**

   - Download from https://python.org

3. **Clone the repository**

```bash
git clone <repository-url>
cd Video-to-Transcription
```

4. **Setup Environment**

```bash
# Option A: UV Environment (Recommended)
uv venv venv -p 3.11
uv pip install -r requirements.txt
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118     # GPU with CUDA
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu       # CPU only

# Option B: Virtual Environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118     # GPU with CUDA
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu       # CPU only
```

5. **Install FFmpeg**

   - **Windows**: Download from https://ffmpeg.org/ or use `winget install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`
   - **macOS**: `brew install ffmpeg`

## 🚀 Usage

## Daily Usage

### With UV (Recommended)

```bash
# Start the application
# Activate venv first to avoid any problems
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
uv run app.py

# Open your browser to http://localhost:8000
```

### With Traditional Python

```bash
# Activate virtual environment first
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Start the application
python app.py

# Open your browser to http://localhost:8000
```

**Note:** Use `python3` or `python3.11` if you have multiple Python versions installed.

### Environment Management

#### UV Environment (Recommended)

```bash
# All commands automatically use the UV-managed venv/
uv run app.py
uv pip install new-package
uv run python -c "import torch; print(torch.__version__)"

# Check Python versions
uv python list
uv python install 3.12  # Install additional versions
```

#### Virtual Environment (Traditional)

```bash
# Windows
venv\Scripts\activate
python app.py
pip install new-package
deactivate

# Linux/macOS
source venv/bin/activate
python app.py
pip install new-package
deactivate
```

**Note:** Use `python3` or `python3.11` if you have multiple Python versions installed.

### Web Interface

#### 1. **Model Management**

- **View Installed Models**: See which models are downloaded in your environment
- **Download New Models**: Browse compatible models for your hardware
- **Real-time Model Info**: View model sizes, requirements, and recommendations

#### 2. **Upload & Process**

- **Select File**: Choose video or audio file
- **Choose Model**: Select from downloaded models or use auto-selection
- **Model Selection**: Choose from available models based on your hardware
- **AI Prompt**: Enter custom prompt for AI processing

#### 3. **Monitor Progress**

- **Real-time Updates**: WebSocket-based progress tracking
- **Model Download Progress**: Live updates during model downloads
- **Processing Status**: Step-by-step transcription and AI processing updates

#### 4. **Manage Results**

- **Download Files**: Access transcriptions and AI responses
- **File Management**: Delete old files to free space
- **Organized Storage**: Separate folders for different file types

## ⚙️ Configuration

### Environment Variables

The setup automatically creates a `.env` file with configuration options:

```env
# =================================================================
# Video to Transcription Service v2.0 - Configuration
# =================================================================

# OpenAI Configuration (optional - for AI text processing features)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1       # gpt-4.1 is the default model

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
```

### Environment Isolation Benefits

#### UV Environment (Recommended)

- **Modern Python management**: Automatic Python 3.11 installation
- **No system pollution**: All packages isolated in UV-managed `venv/`
- **Faster installations**: Optimized dependency resolution
- **Easy cleanup**: `rm -rf venv` removes entire environment
- **Consistent across systems**: Same environment everywhere

#### Virtual Environment (Fallback)

- **System independence**: Dependencies don't affect system Python
- **Reproducible**: Same environment across different systems
- **Easy cleanup**: `rm -rf venv` removes entire environment
- **Standard approach**: Works with any Python 3.11+ installation

## 🏗 Architecture

### Environment-Aware Design

The application automatically adapts to your environment setup:

- **`install_dependencies.py`**: Smart dependency installer with UV/venv detection
- **`setup.py`**: Complete project setup with UV preference
- **Start scripts**: Automatic environment activation and management
- **Model caching**: Environment-specific model storage in `venv/`
- **Configuration**: Environment-aware settings and paths

### Deployment Scenarios

#### Local Development with UV (Recommended)

```bash
# Modern development workflow
uv run app.py
uv pip install new-dependency
uv python list  # Manage Python versions
```

#### Server Deployment with venv (Fallback)

```bash
# Traditional server deployment
source venv/bin/activate
python app.py
pip freeze > requirements.txt  # Save dependencies
```

**Note:** Use `python3` or `python3.11` if you have multiple Python versions installed.

## 🔧 Troubleshooting

### Environment Issues

1. **UV Installation and Setup**

   ```bash
   # Install UV
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install Python 3.11
   uv python install 3.11

   # Check installations
   uv python list
   ```

2. **Virtual environment activation**

   ```bash
   # Windows
   venv\Scripts\activate

   # Linux/macOS
   source venv/bin/activate

   # Check activation
   which python  # Should point to venv
   ```

3. **Mixed environments**

   ```bash
   # Clean start with UV (recommended)
   rm -rf venv
   uv venv venv -p 3.11
   python setup.py

   # Clean start with venv
   rm -rf venv
   python setup.py
   ```

### Dependency Conflicts

1. **Clean environment setup**

   - Remove existing environment (`rm -rf venv`)
   - Run `python setup.py` for fresh installation

2. **Check environment isolation**

   ```bash
   # Verify you're in the right environment
   which python
   python -c "import sys; print(sys.prefix)"
   ```

### Common Issues

1. **Python version problems**

   - Use `python3` or `python3.11` if `python` points to Python 2.x
   - Install Python 3.11 from https://python.org or use UV

2. **FFmpeg not found**

   - **Windows**: Download from https://ffmpeg.org/ or `winget install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`
   - **macOS**: `brew install ffmpeg`

3. **GPU not detected**

   - Install NVIDIA drivers and CUDA toolkit
   - Verify with: `nvidia-smi`
   - Restart after driver installation

4. **Model download issues**

   - Check internet connection
   - Ensure sufficient disk space (models can be 1-3GB)
   - Models download automatically on first use

5. **Port already in use**

   - Change PORT in `.env` file
   - Or stop other applications using port 8000

### Getting Help

1. **Check logs**: Application logs show detailed error information
2. **Environment verification**: Run `python setup.py` to re-verify installation
3. **Clean reinstall**: Remove `venv/` folder and run setup again
4. **Hardware compatibility**: Check if your hardware meets minimum requirements

## 📊 Performance Tips

### Local Development

- **Use UV**: Faster dependency management and installation
- **GPU acceleration**: Ensure CUDA is properly installed for faster processing
- **Model selection**: Use larger models for better accuracy when hardware allows

### Server Deployment

- **CPU optimization**: Use smaller models (tiny/base) for faster processing
- **Memory management**: Monitor RAM usage with larger files
- **File size limits**: Adjust `MAX_FILE_SIZE_MB` based on server capabilities

## 🔐 Security Notes

- **Environment isolation**: All dependencies contained in `venv/` folder
- **No system changes**: Installation doesn't modify system Python or packages
- **Clean removal**: Delete project folder to completely remove all traces
- **API keys**: Store securely in `.env` file (not in version control)

---

**🎉 Ready to get started?** Simply run `python setup.py` and you'll be transcribing in minutes!
