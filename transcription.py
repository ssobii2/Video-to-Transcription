import os
import time
import asyncio
import threading
from typing import Optional, Callable, AsyncGenerator, Iterator
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from config import Config, ModelConfig, ModelSize
from utils import format_time, format_timestamp, ProgressTracker, get_file_size_mb

logger = logging.getLogger(__name__)

class ModelManager:
    """Manage Whisper model instances with caching and optimization"""
    
    def __init__(self):
        self._models = {}
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def get_model(self, config: ModelConfig) -> WhisperModel:
        """Get or create a Whisper model instance"""
        model_key = f"{config.model_size.value}_{config.device}_{config.compute_type}"
        
        if model_key not in self._models:
            logger.info(f"Loading Whisper model: {config.model_size.value} on {config.device}")
            self._models[model_key] = WhisperModel(
                config.model_size.value,
                device=config.device,
                compute_type=config.compute_type,
                num_workers=config.num_workers
            )
            logger.info(f"Model {config.model_size.value} loaded successfully")
        
        return self._models[model_key]
    
    def clear_cache(self):
        """Clear all cached models to free memory"""
        for model in self._models.values():
            del model
        self._models.clear()
        logger.info("Model cache cleared")

class WhisperTranscriber:
    """Enhanced Whisper transcriber with adaptive model selection and performance optimization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_manager = ModelManager()
        self._current_model_config = None
        
    async def transcribe_with_progress(
        self,
        audio_path: str,
        prompt: str = "",
        model_size: str = "",
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Transcribe audio with real-time progress updates
        
        Args:
            audio_path: Path to audio file
            prompt: Initial prompt for transcription
            model_size: Specific model size to use (e.g., "base", "large-v3")
            progress_callback: Callback for progress updates
            
        Yields:
            Transcription text chunks as they become available
        """
        
        # Select optimal model configuration
        model_config = self._select_model_config(audio_path, model_size)
        # Store the model config for later reference
        self._current_model_config = model_config
        
        if progress_callback:
            await progress_callback(f"Loading {model_config.model_size.value} model...")
        
        # Load model in thread pool to avoid blocking
        model = await asyncio.get_event_loop().run_in_executor(
            None, self.model_manager.get_model, model_config
        )
        
        if progress_callback:
            file_size = get_file_size_mb(audio_path)
            await progress_callback(f"Starting transcription ({file_size:.1f} MB)...")
        
        # Perform transcription in thread pool
        transcription_task = asyncio.get_event_loop().run_in_executor(
            None, self._transcribe_blocking, model, audio_path, prompt
        )
        
        # Monitor progress and yield results
        start_time = time.time()
        last_update = 0
        total_time = 0
        segment_count = 0
        
        try:
            # Run transcription and monitor progress
            segments = await transcription_task
            
            full_transcript = ""
            
            for segment in segments:
                segment_count += 1
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Add segment text to full transcript with timestamp range
                start_timestamp = format_timestamp(segment.start)
                end_timestamp = format_timestamp(segment.end)
                segment_text = f"{start_timestamp} --> {end_timestamp}\n{segment.text}\n\n"
                full_transcript += segment_text
                
                # Send progress update
                if progress_callback and (current_time - last_update) >= 1.0:  # Update every second
                    progress_msg = f"Transcribing... {segment_count} segments processed ({format_time(elapsed)} elapsed)"
                    await progress_callback(progress_msg)
                    last_update = current_time
                
                # Yield chunk of transcription
                yield segment_text
            
            # Calculate total time after completion
            total_time = time.time() - start_time
            
            if progress_callback:
                await progress_callback(f"Transcription completed in {format_time(total_time)}")
            
            logger.info(f"Transcription completed: {segment_count} segments in {format_time(total_time)}")
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Transcription failed after {format_time(total_time)}: {e}")
            if progress_callback:
                await progress_callback(f"Transcription failed: {str(e)}")
            raise
    
    async def transcribe(
        self,
        audio_path: str,
        prompt: str = "",
        model_size: str = ""
    ) -> str:
        """
        Transcribe audio file and return complete transcript
        
        Args:
            audio_path: Path to audio file
            prompt: Initial prompt for transcription
            model_size: Specific model size to use (e.g., "base", "large-v3")
            
        Returns:
            Complete transcription text
        """
        full_transcript = ""
        
        async for segment in self.transcribe_with_progress(audio_path, prompt, model_size):
            full_transcript += segment
        
        return full_transcript.strip()
    
    def _select_model_config(self, audio_path: str, model_size: str = "") -> ModelConfig:
        """Select optimal model configuration based on file and preferences"""
        
        # If a specific model is requested, try to use it
        if model_size:
            try:
                # Convert string to ModelSize enum
                from config import ModelSize
                size_mapping = {
                    "tiny": ModelSize.TINY,
                    "base": ModelSize.BASE,
                    "small": ModelSize.SMALL,
                    "medium": ModelSize.MEDIUM,
                    "large-v2": ModelSize.LARGE,
                    "large-v3": ModelSize.LARGE_V3,
                    "turbo": ModelSize.TURBO
                }
                
                if model_size in size_mapping:
                    requested_model = size_mapping[model_size]
                    logger.info(f"Using specifically requested {model_size} model")
                    
                    # Create appropriate config for the requested model
                    if self.config.hardware.has_gpu:
                        device = "cuda"
                        compute_type = "float16"
                        num_workers = 1 if model_size in ["large-v3", "large-v2", "turbo"] else 2
                        batch_size = 4 if model_size in ["large-v3", "large-v2"] else 8
                    else:
                        device = "cpu"
                        compute_type = "int8"
                        num_workers = min(self.config.hardware.cpu_cores, 2)
                        batch_size = 1
                    
                    return ModelConfig(
                        model_size=requested_model,
                        device=device,
                        compute_type=compute_type,
                        num_workers=num_workers,
                        batch_size=batch_size
                    )
                    
            except Exception as e:
                logger.warning(f"Could not use requested model {model_size}: {e}")
        
        # For very large files, consider using a smaller model on limited hardware
        try:
            file_size = get_file_size_mb(audio_path)
            
            # If file is very large and we're on limited hardware, step down model size
            if file_size > 100 and not self.config.hardware.has_gpu:
                if self.config.model_config.model_size in [ModelSize.BASE, ModelSize.SMALL]:
                    logger.info(f"Large file ({file_size:.1f} MB) detected, using tiny model for CPU processing")
                    return ModelConfig(
                        model_size=ModelSize.TINY,
                        device="cpu",
                        compute_type="int8",
                        num_workers=min(self.config.hardware.cpu_cores, 2),
                        batch_size=1
                    )
        except Exception as e:
            logger.warning(f"Could not determine file size: {e}")
        
        # Use default configuration (prioritizes accuracy)
        logger.info(f"Using {self.config.model_config.model_size.value} model for optimal accuracy")
        return self.config.model_config
    
    def _transcribe_blocking(
        self,
        model: WhisperModel,
        audio_path: str,
        prompt: str = ""
    ) -> Iterator[Segment]:
        """Blocking transcription method to run in thread pool"""
        
        transcription_options = {
            'beam_size': 5,
            'best_of': 5,
            'temperature': 0.0,
            'compression_ratio_threshold': 2.4,
            'log_prob_threshold': -1.0,
            'no_speech_threshold': 0.6,
            'condition_on_previous_text': True,
            'word_timestamps': True,
            'prepend_punctuations': "\"'¿([{-",
            'append_punctuations': "\"'.。,，!！?？:：\")]}、"
        }
        
        # Add prompt if provided
        if prompt.strip():
            transcription_options['initial_prompt'] = prompt
        
        try:
            segments, info = model.transcribe(audio_path, **transcription_options)
            
            logger.info(f"Transcription info - Language: {info.language} (probability: {info.language_probability:.2f})")
            logger.info(f"Duration: {format_time(info.duration)}")
            
            return segments
            
        except Exception as e:
            logger.error(f"Model transcription failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about current model configuration"""
        return {
            "config": self.config.get_model_info(),
            "loaded_models": list(self.model_manager._models.keys())
        }
    
    async def warm_up_model(self, model_size: str = "") -> None:
        """Pre-load model to reduce first transcription latency"""
        if model_size:
            model_config = self._select_model_config("", model_size)
        else:
            model_config = self.config.model_config
        
        logger.info(f"Warming up {model_config.model_size.value} model...")
        
        # Load model in background
        await asyncio.get_event_loop().run_in_executor(
            None, self.model_manager.get_model, model_config
        )
        
        logger.info("Model warm-up completed")
    
    def clear_model_cache(self) -> None:
        """Clear model cache to free memory"""
        self.model_manager.clear_cache()

class TranscriptionService:
    """High-level transcription service with file management"""
    
    def __init__(self, config: Config):
        self.config = config
        self.transcriber = WhisperTranscriber(config)
    
    async def transcribe_file(
        self,
        audio_path: str,
        prompt: str = "",
        model_size: str = "",
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Transcribe file and save results
        
        Args:
            audio_path: Path to audio file
            prompt: Transcription prompt
            model_size: Specific model size to use (e.g., "base", "large-v3")
            progress_callback: Progress callback function
            
        Returns:
            Path to saved transcription file
        """
        
        # Generate output filename
        audio_filename = os.path.basename(audio_path)
        transcript_filename = f"{os.path.splitext(audio_filename)[0]}_transcript.txt"
        transcript_path = os.path.join(self.config.transcription_folder, transcript_filename)
        
        try:
            # Perform transcription
            if progress_callback:
                await progress_callback("Starting transcription...")
            
            transcript = await self.transcriber.transcribe(
                audio_path, prompt, model_size
            )
            
            # Save transcript
            if progress_callback:
                await progress_callback("Saving transcription...")
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(f"Transcription of: {audio_filename}\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                model_name = getattr(self.transcriber, '_current_model_config', None)
                if model_name and hasattr(model_name, 'model_size'):
                    f.write(f"Model: {model_name.model_size.value}\n")
                else:
                    f.write(f"Model: {model_size if model_size else self.config.model_config.model_size.value}\n")
                f.write("-" * 50 + "\n\n")
                f.write(transcript)
            
            if progress_callback:
                await progress_callback("Transcription saved successfully!")
            
            logger.info(f"Transcription saved: {transcript_filename}")
            return transcript_path
            
        except Exception as e:
            logger.error(f"Transcription service failed: {e}")
            # Clean up partial file
            if os.path.exists(transcript_path):
                os.remove(transcript_path)
            raise
    
    async def get_available_models(self) -> list:
        """Get list of available models based on hardware"""
        models = []
        
        if self.config.environment.value == "local" and self.config.hardware.has_gpu:
            models.extend([
                {"name": "Large v3", "size": "large-v3", "recommended": True, "description": "Highest accuracy"},
                {"name": "Large v2", "size": "large-v2", "recommended": False, "description": "High accuracy"},
                {"name": "Turbo", "size": "turbo", "recommended": False, "description": "Fastest with good accuracy"},
                {"name": "Medium", "size": "medium", "recommended": False, "description": "Balanced speed/accuracy"},
                {"name": "Base", "size": "base", "recommended": False, "description": "Fast, good accuracy"},
                {"name": "Small", "size": "small", "recommended": False, "description": "Very fast"},
                {"name": "Tiny", "size": "tiny", "recommended": False, "description": "Fastest, lower accuracy"}
            ])
        else:
            # Server/CPU models
            models.extend([
                {"name": "Base", "size": "base", "recommended": True, "description": "Best for CPU"},
                {"name": "Small", "size": "small", "recommended": False, "description": "Faster, good accuracy"},
                {"name": "Tiny", "size": "tiny", "recommended": False, "description": "Fastest for limited resources"}
            ])
        
        return models 