import os
import asyncio
import time
import logging
import subprocess
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv()

from config import Config, Environment
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
    logger.info(f"Environment: {config.environment.value}")
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
        "environment": config.environment.value,
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
        
        if cuda_version or config.hardware.has_gpu:
            # Local environment with GPU
            compatible = [
                {"name": "Large-v3", "size": "large-v3", "description": "Highest accuracy (recommended)", "recommended": True, "memory_req": "6GB+ VRAM"},
                {"name": "Large-v2", "size": "large-v2", "description": "High accuracy", "memory_req": "6GB+ VRAM"},

                {"name": "Medium", "size": "medium", "description": "Balanced speed/accuracy", "memory_req": "2GB+ VRAM"},
                {"name": "Base", "size": "base", "description": "Fast, good accuracy", "memory_req": "1GB+ VRAM"},
                {"name": "Small", "size": "small", "description": "Very fast", "memory_req": "1GB+ VRAM"},
                {"name": "Tiny", "size": "tiny", "description": "Fastest, lower accuracy", "memory_req": "512MB+ VRAM"}
            ]
        else:
            # Server environment - CPU only
            compatible = [
                {"name": "Base", "size": "base", "description": "Best for CPU (recommended)", "recommended": True, "memory_req": "4GB+ RAM"},
                {"name": "Small", "size": "small", "description": "Good accuracy for CPU", "memory_req": "4GB+ RAM"},
                {"name": "Tiny", "size": "tiny", "description": "Fastest for limited resources", "memory_req": "2GB+ RAM"}
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
        
        # Check file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > config.max_file_size_mb:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {config.max_file_size_mb} MB"
            )
        
        # Clean input folder and save file
        clean_folder_contents(config.input_folder)
        
        safe_name = safe_filename(file.filename)
        file_path = os.path.join(config.input_folder, safe_name)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
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