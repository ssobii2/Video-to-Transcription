import os
import asyncio
import time
import logging
import subprocess
from typing import Optional, Dict, List
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uuid
import tempfile
import json
import shutil
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

from config import Config
from utils import allowed_file, clean_folder_contents, get_file_size_mb, safe_filename, validate_environment
from media_processor import MediaProcessor
from transcription import TranscriptionService
from ai_service import AIService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize configuration
config = Config()

# Initialize services
media_processor = MediaProcessor(config)
transcription_service = TranscriptionService(config)
ai_service = AIService(config)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Video to Transcription Service v2.0")
    logger.info(f"Hardware: {config.hardware.cpu_cores} CPU cores, {config.hardware.ram_gb:.1f}GB RAM")
    
    if config.hardware.has_gpu:
        logger.info(f"GPU: {config.hardware.gpu_name} ({config.hardware.gpu_memory_gb:.1f}GB VRAM)")
    else:
        logger.info("GPU: Not available")
    
    logger.info(f"Selected model: {config.model_config.model_size.value} on {config.model_config.device}")
    
    # Validate environment
    requirements = validate_environment()
    if not requirements['ffmpeg']:
        logger.warning("⚠️  FFmpeg not found - video conversion will not work")
        logger.warning("   Please install FFmpeg: https://ffmpeg.org/")
    if not requirements['ffprobe']:
        logger.warning("⚠️  FFprobe not found - duration detection may not work")
    
    # Test AI service if configured
    if ai_service.is_available():
        logger.info("Testing AI service connection...")
        try:
            is_working = await ai_service.test_connection()
            if is_working:
                logger.info("✅ AI service is working")
            else:
                logger.warning("⚠️  AI service test failed - check your API key")
        except Exception as e:
            logger.warning(f"⚠️  AI service test error: {e}")
    else:
        logger.info("AI service not configured (add OPENAI_API_KEY to .env for AI features)")
    
    # Pre-load models for better user experience
    logger.info("Checking model availability...")
    try:
        # Try to load the default model to trigger download if needed
        await transcription_service.transcriber.warm_up_model()
        logger.info(f"✅ {config.model_config.model_size.value} model ready")
    except Exception as e:
        logger.warning(f"⚠️  Model warm-up failed: {e}")
        logger.info("Models will be downloaded automatically when first used")
    
    logger.info("🚀 Service ready at http://localhost:8000")
    
    yield
    
    # Shutdown (optional cleanup)
    logger.info("Shutting down Video to Transcription Service")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Video to Transcription Service",
    description="Intelligent video/audio transcription with adaptive model selection",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store for active chunked uploads
active_uploads: Dict[str, dict] = {}

class ChunkUploadInit(BaseModel):
    filename: str
    filesize: int
    total_chunks: int
    prompt: str = ""
    model: str

class ChunkUploadComplete(BaseModel):
    upload_id: str

class ConnectionManager:
    """Manage WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, message: str, overwrite: bool = False):
        """Send message to all connected clients"""
        if not self.active_connections:
            return
            
        for connection in self.active_connections[:]:  # Use slice copy for safe iteration
            try:
                if overwrite:
                    await connection.send_text(f"\r{message}")
                else:
                    await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                # Remove disconnected client
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

manager = ConnectionManager()

@app.get("/")
async def get_frontend():
    """Serve the main frontend page"""
    return FileResponse("static/index.html")

@app.get("/api/status")
async def get_status():
    """Get system status and configuration"""
    return JSONResponse({
        "status": "running",
        "model_info": config.get_model_info(),
        "ai_available": ai_service.is_available(),
        "supported_formats": media_processor.get_supported_formats()
    })

@app.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    return JSONResponse({
        "models": await transcription_service.get_available_models(),
        "current": config.model_config.model_size.value,
        "installed_models": await get_installed_models(),
        "hardware_info": config.get_model_info()["hardware"]
    })

async def get_installed_models():
    """Get list of models that are already downloaded"""
    try:
        # Try to detect installed models by checking the faster-whisper cache
        import os
        from pathlib import Path
        
        # Common cache locations for faster-whisper
        cache_dirs = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "whisper",
            Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None
        ]
        
        installed = []
        model_names = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
        
        for cache_dir in cache_dirs:
            if cache_dir and cache_dir.exists():
                for model_name in model_names:
                    # Look for model directories
                    pattern_matches = list(cache_dir.glob(f"*whisper*{model_name}*"))
                    if pattern_matches:
                        if model_name not in installed:
                            installed.append(model_name)
        
        return installed
    except Exception as e:
        logger.warning(f"Could not detect installed models: {e}")
        return []

@app.post("/api/models/download")
async def download_model(whisper_model: str = Form(...)):
    """Download a specific model"""
    try:
        # Check for turbo model specifically
        if whisper_model.lower() == "turbo":
            raise HTTPException(
                status_code=400,
                detail="The 'turbo' model is not supported by faster-whisper. Available models: tiny, base, small, medium, large-v2, large-v3"
            )
        
        # Validate model name
        available_models = await transcription_service.get_available_models()
        available_model_sizes = [model["size"] for model in available_models]
        
        if whisper_model not in available_model_sizes:
            raise HTTPException(
                status_code=400, 
                detail=f"Model {whisper_model} not supported. Available models: {', '.join(available_model_sizes)}"
            )
        
        logger.info(f"Starting download for model: {whisper_model}")
        
        # Start download in background
        asyncio.create_task(download_model_background(whisper_model))
        
        return JSONResponse({
            "status": "started",
            "model": whisper_model,
            "message": f"Download started for {whisper_model} model"
        })
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def download_model_background(whisper_model: str):
    """Background task to download model with progress updates"""
    try:
        await manager.send_message(f"📥 Starting download for {whisper_model} model...")
        
        # This will download and cache the model
        from faster_whisper import WhisperModel
        import asyncio
        import time
        
        start_time = time.time()
        await manager.send_message(f"📦 Downloading {whisper_model} model (this may take several minutes)...")
        
        # Create a progress update task
        progress_task = asyncio.create_task(download_progress_updates(whisper_model, start_time))
        
        try:
            # Download model in executor to avoid blocking
            model = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: WhisperModel(whisper_model, device="cpu", compute_type="int8")
            )
            
            # Cancel progress updates
            progress_task.cancel()
            
            elapsed_time = time.time() - start_time
            await manager.send_message(f"✅ {whisper_model} model downloaded successfully in {elapsed_time:.1f} seconds!")
            
            # Clean up
            del model
            
            logger.info(f"Model {whisper_model} downloaded successfully")
            
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            progress_task.cancel()
            await manager.send_message(f"❌ Download of {whisper_model} model was cancelled")
            raise
        except Exception as e:
            # Cancel progress updates on error
            progress_task.cancel()
            raise e
        
    except Exception as e:
        error_msg = f"❌ Failed to download {whisper_model} model: {str(e)}"
        await manager.send_message(error_msg)
        logger.error(f"Model download failed: {e}")

async def download_progress_updates(model_name: str, start_time: float):
    """Send periodic progress updates during model download"""
    try:
        update_interval = 10  # seconds
        counter = 1
        
        while True:
            await asyncio.sleep(update_interval)
            elapsed = time.time() - start_time
            await manager.send_message(
                f"📦 Still downloading {model_name} model... ({elapsed:.0f}s elapsed, please wait)"
            )
            counter += 1
            
            # Increase interval slightly for longer downloads
            if counter > 6:  # After 1 minute
                update_interval = 20
            elif counter > 3:  # After 30 seconds  
                update_interval = 15
                
    except asyncio.CancelledError:
        # Task was cancelled, download completed
        pass

@app.delete("/api/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a downloaded model"""
    try:
        import shutil
        from pathlib import Path
        
        deleted_paths = []
        total_size_freed = 0
        
        # Common cache locations for faster-whisper and HuggingFace models
        cache_dirs = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "whisper"
        ]
        
        # Also check for HF_HOME environment variable
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dirs.append(Path(hf_home) / "hub")
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                # Look for model directories that match the model name
                for item in cache_dir.iterdir():
                    if item.is_dir() and (
                        f"whisper-{model_name}" in item.name.lower() or
                        f"openai--whisper-{model_name}" in item.name.lower() or
                        (model_name in item.name.lower() and "whisper" in item.name.lower())
                    ):
                        try:
                            # Calculate size before deletion
                            size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                            total_size_freed += size
                            
                            # Delete the directory
                            shutil.rmtree(item)
                            deleted_paths.append(str(item))
                            logger.info(f"Deleted model cache: {item}")
                        except Exception as e:
                            logger.warning(f"Could not delete {item}: {e}")
        
        # Also clear the model from memory if it's loaded
        if hasattr(transcription_service, 'transcriber') and hasattr(transcription_service.transcriber, 'model_manager'):
            transcription_service.transcriber.model_manager.clear_cache()
        
        if deleted_paths:
            size_mb = total_size_freed / (1024 * 1024)
            return JSONResponse({
                "message": f"Model '{model_name}' deleted successfully",
                "deleted_paths": deleted_paths,
                "size_freed_mb": round(size_mb, 2)
            })
        else:
            return JSONResponse({
                "message": f"Model '{model_name}' not found in cache",
                "note": "Model may not be downloaded or may be stored in a different location"
            })
        
    except Exception as e:
        logger.error(f"Model deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.get("/api/models/compatible")
async def get_compatible_models():
    """Get models compatible with current hardware"""
    try:
        cuda_version = None
        if config.hardware.has_gpu:
            # Try to detect CUDA version
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    import re
                    cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
                    if cuda_match:
                        cuda_version = cuda_match.group(1)
            except:
                pass
        
        # Show only models that the hardware can realistically handle
        if cuda_version or config.hardware.has_gpu:
            # GPU available - filter by VRAM capacity
            if config.hardware.gpu_memory_gb >= 8:
                # High-end GPU: Can handle all models efficiently
                compatible = [
                    {"name": "Large-v3", "size": "large-v3", "description": "Highest accuracy (recommended)", "recommended": True, "memory_req": "6GB+ VRAM"},
                    {"name": "Large-v2", "size": "large-v2", "description": "Very high accuracy", "recommended": False, "memory_req": "6GB+ VRAM"},
                    {"name": "Medium", "size": "medium", "description": "Good accuracy", "recommended": False, "memory_req": "2GB+ VRAM"},
                    {"name": "Small", "size": "small", "description": "Decent accuracy", "recommended": False, "memory_req": "1GB+ VRAM"},
                    {"name": "Base", "size": "base", "description": "Good balance", "recommended": False, "memory_req": "1GB+ VRAM"},
                    {"name": "Tiny", "size": "tiny", "description": "Basic accuracy", "recommended": False, "memory_req": "512MB+ VRAM"}
                ]
            elif config.hardware.gpu_memory_gb >= 6:
                # Mid-range GPU: Can handle large models but may be slower
                compatible = [
                    {"name": "Large-v3", "size": "large-v3", "description": "Highest accuracy (recommended)", "recommended": True, "memory_req": "6GB+ VRAM"},
                    {"name": "Large-v2", "size": "large-v2", "description": "Very high accuracy", "recommended": False, "memory_req": "6GB+ VRAM"},
                    {"name": "Medium", "size": "medium", "description": "Good accuracy", "recommended": False, "memory_req": "2GB+ VRAM"},
                    {"name": "Small", "size": "small", "description": "Decent accuracy", "recommended": False, "memory_req": "1GB+ VRAM"},
                    {"name": "Base", "size": "base", "description": "Good balance", "recommended": False, "memory_req": "1GB+ VRAM"},
                    {"name": "Tiny", "size": "tiny", "description": "Basic accuracy", "recommended": False, "memory_req": "512MB+ VRAM"}
                ]
            elif config.hardware.gpu_memory_gb >= 4:
                # Lower-end GPU: Large models may cause memory issues
                compatible = [
                    {"name": "Medium", "size": "medium", "description": "Highest practical accuracy (recommended)", "recommended": True, "memory_req": "2GB+ VRAM"},
                    {"name": "Small", "size": "small", "description": "Good accuracy", "recommended": False, "memory_req": "1GB+ VRAM"},
                    {"name": "Base", "size": "base", "description": "Decent accuracy", "recommended": False, "memory_req": "1GB+ VRAM"},
                    {"name": "Tiny", "size": "tiny", "description": "Basic accuracy", "recommended": False, "memory_req": "512MB+ VRAM"}
                ]
            else:
                # Very low VRAM: Only smaller models
                compatible = [
                    {"name": "Small", "size": "small", "description": "Best accuracy for low VRAM (recommended)", "recommended": True, "memory_req": "1GB+ VRAM"},
                    {"name": "Base", "size": "base", "description": "Good balance", "recommended": False, "memory_req": "1GB+ VRAM"},
                    {"name": "Tiny", "size": "tiny", "description": "Basic accuracy", "recommended": False, "memory_req": "512MB+ VRAM"}
                ]
        else:
            # CPU only - filter by RAM and CPU power
            if config.hardware.ram_gb >= 15 and config.hardware.cpu_cores >= 8:
                # High-end CPU: Can handle large models (though slowly)
                compatible = [
                    {"name": "Large-v3", "size": "large-v3", "description": "Highest accuracy (recommended)", "recommended": True, "memory_req": "6GB+ RAM"},
                    {"name": "Large-v2", "size": "large-v2", "description": "Very high accuracy", "recommended": False, "memory_req": "6GB+ RAM"},
                    {"name": "Medium", "size": "medium", "description": "Good accuracy", "recommended": False, "memory_req": "3GB+ RAM"},
                    {"name": "Small", "size": "small", "description": "Decent accuracy", "recommended": False, "memory_req": "2GB+ RAM"},
                    {"name": "Base", "size": "base", "description": "Good balance", "recommended": False, "memory_req": "1GB+ RAM"},
                    {"name": "Tiny", "size": "tiny", "description": "Basic accuracy", "recommended": False, "memory_req": "512MB+ RAM"}
                ]
            elif config.hardware.ram_gb >= 7 and config.hardware.cpu_cores >= 4:
                # Mid-range CPU: Can handle medium models efficiently
                compatible = [
                    {"name": "Medium", "size": "medium", "description": "Highest practical accuracy (recommended)", "recommended": True, "memory_req": "3GB+ RAM"},
                    {"name": "Small", "size": "small", "description": "Good accuracy", "recommended": False, "memory_req": "2GB+ RAM"},
                    {"name": "Base", "size": "base", "description": "Decent accuracy", "recommended": False, "memory_req": "1GB+ RAM"},
                    {"name": "Tiny", "size": "tiny", "description": "Basic accuracy", "recommended": False, "memory_req": "512MB+ RAM"}
                ]
            else:
                # Limited CPU: Only smaller models that won't overwhelm the system
                compatible = [
                    {"name": "Small", "size": "small", "description": "Highest practical accuracy (recommended)", "recommended": True, "memory_req": "2GB+ RAM"},
                    {"name": "Base", "size": "base", "description": "Good accuracy", "recommended": False, "memory_req": "1GB+ RAM"},
                    {"name": "Tiny", "size": "tiny", "description": "Basic accuracy", "recommended": False, "memory_req": "512MB+ RAM"}
                ]
        
        return JSONResponse({
            "compatible_models": compatible,
            "hardware": {
                "has_gpu": config.hardware.has_gpu,
                "gpu_memory_gb": config.hardware.gpu_memory_gb,
                "ram_gb": config.hardware.ram_gb,
                "cuda_version": cuda_version
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting compatible models: {e}")
        raise HTTPException(status_code=500, detail="Error getting compatible models")

@app.get("/api/prompts")
async def get_suggested_prompts():
    """Get suggested AI prompts"""
    if not ai_service.is_available():
        raise HTTPException(status_code=503, detail="AI service not available")
    
    return JSONResponse({
        "prompts": ai_service.get_suggested_prompts()
    })

@app.post("/api/check-duplicate")
async def check_duplicate_transcription(filename: str = Form(...)):
    """Check if transcription or AI response already exists for a filename"""
    try:
        safe_name = safe_filename(filename)
        base_name = os.path.splitext(safe_name)[0]
        
        transcript_filename = f"{base_name}_transcript.txt"
        transcript_path = os.path.join(config.transcription_folder, transcript_filename)
        
        ai_response_filename = f"{base_name}_ai_response.txt"
        ai_response_path = os.path.join(config.ai_responses_folder, ai_response_filename)
        
        existing_files = []
        if os.path.exists(transcript_path):
            existing_files.append(transcript_filename)
        if os.path.exists(ai_response_path):
            existing_files.append(ai_response_filename)
        
        if existing_files:
            if len(existing_files) == 2:
                message = f"Both transcription and AI response already exist for this file ({', '.join(existing_files)}). Please delete them first if you want to create new ones."
            elif transcript_filename in existing_files:
                message = f"A transcription already exists for this file ({transcript_filename}). Please delete it first if you want to create a new one."
            else:
                message = f"An AI response already exists for this file ({ai_response_filename}). Please delete it first if you want to create a new one."
            
            return JSONResponse({
                "exists": True,
                "existing_files": existing_files,
                "message": message
            })
        else:
            return JSONResponse({
                "exists": False,
                "message": "No existing files found. You can proceed with the upload."
            })
    except Exception as e:
        logger.error(f"Error checking duplicate files: {e}")
        raise HTTPException(status_code=500, detail="Error checking for existing files")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), prompt: str = Form(""), model: str = Form("")):
    """Upload and process media file"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        if not allowed_file(file.filename, config.allowed_extensions):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file format. Supported: {', '.join(config.allowed_extensions)}"
            )
        
        # Validate model selection
        if not model:
            raise HTTPException(status_code=400, detail="No model selected")
        
        # Clean input folder and save file using chunked approach (like main_old.py)
        clean_folder_contents(config.input_folder)
        
        safe_name = safe_filename(file.filename)
        file_path = os.path.join(config.input_folder, safe_name)
        
        # Check for duplicate transcriptions before saving file
        transcript_filename = f"{os.path.splitext(safe_name)[0]}_transcript.txt"
        transcript_path = os.path.join(config.transcription_folder, transcript_filename)
        
        if os.path.exists(transcript_path):
            raise HTTPException(
                status_code=400,
                detail=f"Transcription already exists: {transcript_filename}. Please delete the existing transcription file first, then try again."
            )
        
        # Save file using chunked upload to handle large files (adapted from main_old.py)
        file_size_mb = 0
        with open(file_path, 'wb') as f:
            while True:
                chunk = await file.read(1024 * 1024)  # Read 1MB chunks
                if not chunk:
                    break
                f.write(chunk)
                file_size_mb += len(chunk) / (1024 * 1024)
                
                # Check size limit during upload
                if file_size_mb > config.max_file_size_mb:
                    # Clean up partial file
                    f.close()
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {config.max_file_size_mb} MB"
                    )
        
        logger.info(f"File uploaded: {safe_name} ({file_size_mb:.1f} MB)")
        logger.info(f"Model selected: {model}")
        logger.info(f"Prompt received: '{prompt}' (length: {len(prompt.strip())})")
        
        # Process file asynchronously
        asyncio.create_task(process_uploaded_file(file_path, prompt, model))
        
        return JSONResponse({
            "message": "File uploaded successfully. Processing started.",
            "filename": safe_name,
            "size_mb": round(file_size_mb, 1)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_uploaded_file(file_path: str, prompt: str, model: str):
    """Process uploaded file (media conversion + transcription + AI)"""
    try:
        filename = os.path.basename(file_path)
        
        # Step 1: Process media file
        await manager.send_message(f"Processing {filename}...")
        
        audio_path = await media_processor.process_media_file(
            file_path, 
            progress_callback=manager.send_message
        )
        
        # Step 2: Transcribe audio
        await manager.send_message("Starting transcription...")
        
        try:
            transcript_path = await transcription_service.transcribe_file(
                audio_path,
                prompt,
                model,
                progress_callback=manager.send_message
            )
        except FileExistsError as e:
            # Handle duplicate transcription file
            await manager.send_message(f"❌ {str(e)}")
            # Clean up folders even when transcription already exists
            clean_folder_contents(config.input_folder)
            clean_folder_contents(config.output_folder)
            return
        
        # Step 3: Process with AI if available and prompt provided
        if ai_service.is_available() and prompt.strip():
            await manager.send_message("Processing with AI...")
            
            # Read transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_content = f.read()
            
            # Extract just the transcript text (skip header)
            lines = transcript_content.split('\n')
            separator_index = -1
            for i, line in enumerate(lines):
                if line.strip() == '-' * 50:
                    separator_index = i
                    break
            
            if separator_index >= 0:
                transcript_text = '\n'.join(lines[separator_index + 1:]).strip()
            else:
                transcript_text = transcript_content
            
            # Process with AI
            ai_response = await ai_service.process_transcription(
                transcript_text,
                prompt,
                progress_callback=manager.send_message
            )
            
            # Save AI response
            await ai_service.save_ai_response(ai_response, filename, prompt)
        elif prompt.strip() and not ai_service.is_available():
            await manager.send_message("⚠️ AI processing requested but AI service is not available (missing API key)")
        else:
            await manager.send_message("ℹ️ AI processing skipped (no prompt provided)")
        
        await manager.send_message("Processing completed successfully!")
        
        # Clean up input and output folders
        clean_folder_contents(config.input_folder)
        clean_folder_contents(config.output_folder)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        await manager.send_message(f"Processing failed: {str(e)}")
        # Clean up on error
        clean_folder_contents(config.input_folder)
        clean_folder_contents(config.output_folder)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# File management endpoints
@app.get("/transcription/")
async def list_transcriptions():
    """List all transcription files"""
    try:
        files = []
        if os.path.exists(config.transcription_folder):
            files = [f for f in os.listdir(config.transcription_folder) 
                    if f.endswith('.txt')]
        return JSONResponse({"transcriptions": sorted(files, reverse=True)})
    except Exception as e:
        logger.error(f"Error listing transcriptions: {e}")
        raise HTTPException(status_code=500, detail="Error listing transcriptions")

@app.get("/transcription/{filename}")
async def get_transcription(filename: str):
    """Download transcription file"""
    file_path = os.path.join(config.transcription_folder, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Transcription not found")
    return FileResponse(file_path, filename=filename)

@app.delete("/transcription/{filename}")
async def delete_transcription(filename: str):
    """Delete transcription file"""
    try:
        file_path = os.path.join(config.transcription_folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse({"message": "Transcription deleted"})
    except Exception as e:
        logger.error(f"Error deleting transcription: {e}")
        raise HTTPException(status_code=500, detail="Error deleting transcription")

@app.get("/ai/")
async def list_ai_responses():
    """List all AI response files"""
    try:
        files = []
        if os.path.exists(config.ai_responses_folder):
            files = [f for f in os.listdir(config.ai_responses_folder) 
                    if f.endswith('.txt')]
        return JSONResponse({"ai_responses": sorted(files, reverse=True)})
    except Exception as e:
        logger.error(f"Error listing AI responses: {e}")
        raise HTTPException(status_code=500, detail="Error listing AI responses")

@app.get("/ai/{filename}")
async def get_ai_response(filename: str):
    """Download AI response file"""
    file_path = os.path.join(config.ai_responses_folder, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="AI response not found")
    return FileResponse(file_path, filename=filename)

@app.delete("/ai/{filename}")
async def delete_ai_response(filename: str):
    """Delete AI response file"""
    try:
        file_path = os.path.join(config.ai_responses_folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse({"message": "AI response deleted"})
    except Exception as e:
        logger.error(f"Error deleting AI response: {e}")
        raise HTTPException(status_code=500, detail="Error deleting AI response")

@app.post("/upload/init")
async def init_chunked_upload(init_data: ChunkUploadInit):
    """Initialize chunked upload for large files"""
    try:
        # Validate file
        if not init_data.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not allowed_file(init_data.filename, config.allowed_extensions):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file format. Supported: {', '.join(config.allowed_extensions)}"
            )
        
        # Validate model selection
        if not init_data.model:
            raise HTTPException(status_code=400, detail="No model selected")
        
        # Check file size limit
        file_size_mb = init_data.filesize / (1024 * 1024)
        if file_size_mb > config.max_file_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {config.max_file_size_mb} MB"
            )
        
        # Check for duplicate transcriptions
        safe_name = safe_filename(init_data.filename)
        transcript_filename = f"{os.path.splitext(safe_name)[0]}_transcript.txt"
        transcript_path = os.path.join(config.transcription_folder, transcript_filename)
        
        if os.path.exists(transcript_path):
            raise HTTPException(
                status_code=400,
                detail=f"Transcription already exists: {transcript_filename}. Please delete the existing transcription file first, then try again."
            )
        
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Create temporary directory for chunks
        temp_dir = os.path.join(tempfile.gettempdir(), f"upload_{upload_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Store upload info
        active_uploads[upload_id] = {
            "filename": safe_name,
            "filesize": init_data.filesize,
            "total_chunks": init_data.total_chunks,
            "prompt": init_data.prompt,
            "model": init_data.model,
            "temp_dir": temp_dir,
            "chunks_received": set(),
            "created_at": asyncio.get_event_loop().time()
        }
        
        logger.info(f"Initialized chunked upload: {safe_name} ({file_size_mb:.1f} MB, {init_data.total_chunks} chunks)")
        
        return {"upload_id": upload_id, "message": "Upload initialized"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initialize chunked upload: {e}")
        raise HTTPException(status_code=500, detail=f"Upload initialization failed: {str(e)}")

@app.post("/upload/chunk")
async def upload_chunk(
    chunk: UploadFile = File(...),
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...)
):
    """Upload a single chunk of a large file"""
    try:
        # Validate upload session
        if upload_id not in active_uploads:
            raise HTTPException(status_code=400, detail="Invalid upload ID or upload session expired")
        
        upload_info = active_uploads[upload_id]
        
        # Validate chunk
        if chunk_index >= total_chunks or chunk_index < 0:
            raise HTTPException(status_code=400, detail="Invalid chunk index")
        
        if total_chunks != upload_info["total_chunks"]:
            raise HTTPException(status_code=400, detail="Total chunks mismatch")
        
        # Save chunk to temporary directory
        chunk_path = os.path.join(upload_info["temp_dir"], f"chunk_{chunk_index:06d}")
        
        with open(chunk_path, 'wb') as f:
            chunk_data = await chunk.read()
            f.write(chunk_data)
        
        # Track received chunks
        upload_info["chunks_received"].add(chunk_index)
        
        logger.debug(f"Received chunk {chunk_index + 1}/{total_chunks} for upload {upload_id}")
        
        return {
            "message": f"Chunk {chunk_index + 1}/{total_chunks} uploaded successfully",
            "chunks_received": len(upload_info["chunks_received"]),
            "total_chunks": total_chunks
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload chunk: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk upload failed: {str(e)}")

@app.post("/upload/complete")
async def complete_chunked_upload(complete_data: ChunkUploadComplete):
    """Complete chunked upload and start processing"""
    try:
        upload_id = complete_data.upload_id
        
        # Validate upload session
        if upload_id not in active_uploads:
            raise HTTPException(status_code=400, detail="Invalid upload ID or upload session expired")
        
        upload_info = active_uploads[upload_id]
        
        # Verify all chunks received
        expected_chunks = set(range(upload_info["total_chunks"]))
        if upload_info["chunks_received"] != expected_chunks:
            missing_chunks = expected_chunks - upload_info["chunks_received"]
            raise HTTPException(
                status_code=400, 
                detail=f"Missing chunks: {sorted(missing_chunks)}"
            )
        
        # Clean input folder and reconstruct file from chunks
        clean_folder_contents(config.input_folder)
        
        final_file_path = os.path.join(config.input_folder, upload_info["filename"])
        
        with open(final_file_path, 'wb') as final_file:
            for chunk_index in range(upload_info["total_chunks"]):
                chunk_path = os.path.join(upload_info["temp_dir"], f"chunk_{chunk_index:06d}")
                
                if not os.path.exists(chunk_path):
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Chunk file missing: {chunk_index}"
                    )
                
                with open(chunk_path, 'rb') as chunk_file:
                    final_file.write(chunk_file.read())
        
        # Clean up temporary chunks
        try:
            import shutil
            shutil.rmtree(upload_info["temp_dir"])
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")
        
        # Remove from active uploads
        del active_uploads[upload_id]
        
        file_size_mb = upload_info["filesize"] / (1024 * 1024)
        logger.info(f"Chunked upload completed: {upload_info['filename']} ({file_size_mb:.1f} MB)")
        logger.info(f"Model selected: {upload_info['model']}")
        logger.info(f"Prompt received: '{upload_info['prompt']}' (length: {len(upload_info['prompt'].strip())})")
        
        # Process file asynchronously
        asyncio.create_task(process_uploaded_file(
            final_file_path, 
            upload_info["prompt"], 
            upload_info["model"]
        ))
        
        return JSONResponse({
            "message": "Chunked upload completed successfully. Processing started.",
            "filename": upload_info["filename"],
            "size_mb": round(file_size_mb, 1)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to complete chunked upload: {e}")
        
        # Clean up on error
        if upload_id in active_uploads:
            upload_info = active_uploads[upload_id]
            try:
                import shutil
                shutil.rmtree(upload_info["temp_dir"])
            except Exception:
                pass
            del active_uploads[upload_id]
        
        raise HTTPException(status_code=500, detail=f"Upload completion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {config.host}:{config.port}")
    uvicorn.run(
        "app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    ) 