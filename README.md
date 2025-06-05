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
- **UV (Recommended)** or **Python 3.11+**
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

This will automatically:

- ✅ **UV Detection**: Detect UV and recommend installation if not found
- ✅ **Python 3.11 Setup**: Install Python 3.11 via UV if needed
- ✅ **Environment Creation**: Create `venv/` folder with UV or traditional venv
- ✅ **PyTorch Installation**: Install PyTorch with appropriate CUDA support
- ✅ **Interactive Models**: Select models based on your hardware capabilities
- ✅ **Complete Setup**: Configuration files and verification

### Environment Options

#### Option 1: UV (Recommended)

UV provides the best Python management experience:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup project
git clone <repository-url>
cd Video-to-Transcription
python setup.py  # Will detect UV and use it
```

**UV Workflow:**

```bash
# UV will automatically:
uv python install 3.11        # Install Python 3.11
uv venv venv -p 3.11          # Create venv/ with Python 3.11
uv pip install <packages>     # Install packages in venv/
uv run app.py          # Run Python using venv/
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

**Traditional Setup:**

- **Automatic venv creation**: Virtual environment created automatically
- **Environment isolation**: All dependencies installed in `venv/`, not system-wide
- **Server-friendly**: Works on any system with Python 3.11+
- **Manual activation**: Requires venv activation for manual commands

### Manual Installation (Advanced)

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
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Option B: Virtual Environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

5. **Install FFmpeg**

   - **Windows**: Download from https://ffmpeg.org/
   - **Linux**: `sudo apt install ffmpeg`
   - **macOS**: `brew install ffmpeg`

## 🚀 Usage

## Daily Usage

### With UV (Recommended)

```bash
# Start the application
uv run python app.py

# Open your browser to http://localhost:8000
```

### With Traditional Python

```bash
# Activate virtual environment (if using venv)
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Start the application
python app.py

# Open your browser to http://localhost:8000
```

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

Create a `.env` file with your configuration:

```env
# OpenAI Configuration (optional)
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Processing Limits
MAX_FILE_SIZE_MB=500
CHUNK_DURATION=30

# Advanced Options (auto-detection recommended)
# ENVIRONMENT=auto  # auto, local, server
# FORCE_MODEL=large-v3  # Override model selection
# FORCE_DEVICE=auto  # auto, cuda, cpu
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

#### Docker Deployment

```dockerfile
# Can use either UV or venv in containers
FROM python:3.11-slim
# Install UV for better dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# ... setup UV environment
```

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

   # With UV
   uv run python -c "import sys; print(sys.prefix)"
   ```

### Performance Optimization

1. **For Local Use (UV)**:

   - Use UV for fastest dependency management
   - Models cached in UV-managed location
   - Automatic Python version management
   - Use `uv run` for all Python commands

2. **For Server Deployment (venv)**:
   - Use venv for predictable server environments
   - Pin dependencies with `pip freeze`
   - Consider containerization for production

## 🆕 Migration from v1.0

The new version maintains compatibility while adding modern environment management:

1. **Clean migration**:

   ```bash
   # Backup existing files
   mv old-project old-project-backup

   # Fresh installation with UV (recommended)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.11
   git clone <repository-url>
   cd Video-to-Transcription
   python setup.py
   ```

2. **Copy configurations**:
   ```bash
   # Copy your old .env file
   cp ../old-project-backup/.env .
   ```

## 📊 Environment Comparison

| Feature               | UV Environment | Virtual Environment | System Python       |
| --------------------- | -------------- | ------------------- | ------------------- |
| **Isolation**         | ✅ Complete    | ✅ Complete         | ❌ None             |
| **Python Management** | ✅ Automatic   | ⚠️ Manual           | ⚠️ System-dependent |
| **Dependency Speed**  | ✅ Very Fast   | ⚠️ Standard         | ⚠️ Standard         |
| **Server Deployment** | ✅ Excellent   | ✅ Good             | ❌ Not recommended  |
| **Local Development** | ✅ Excellent   | ✅ Good             | ❌ Not recommended  |
| **Cleanup**           | ✅ Easy        | ✅ Easy             | ❌ Impossible       |
| **Version Conflicts** | ✅ None        | ⚠️ Possible         | ❌ Common           |
| **Setup Complexity**  | ✅ Simple      | ⚠️ Manual           | ❌ Complex          |

**Recommendation**: Use UV for both local development and server deployment. Fallback to venv only when UV is not available.

---

**Note**: This v2.0 architecture prioritizes UV for modern Python environment management while maintaining full compatibility with traditional virtual environments. The `venv/` folder approach ensures consistency regardless of the underlying package manager.
