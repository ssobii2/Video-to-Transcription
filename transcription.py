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
from utils import format_time, format_timestamp, ProgressTracker, get_file_size_mb, get_video_duration

logger = logging.getLogger(__name__)

class ModelManager:
    """Manage Whisper model instances with caching and optimization"""
    
    def __init__(self):
        self._models = {}
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def get_model(self, config: ModelConfig) -> WhisperModel:
        """Get or create a Whisper model instance"""
        model_key = f"{config.model_size.value}_{config.device}_{config.compute_type}"
        
        # Validate model size before loading
        if config.model_size.value == "turbo":
            raise ValueError(
                "The 'turbo' model is not supported by faster-whisper. "
                "Please use 'large-v3', 'large-v2', 'medium', 'base', 'small', or 'tiny' instead."
            )
        
        if model_key not in self._models:
            logger.info(f"Loading Whisper model: {config.model_size.value} on {config.device}")
            try:
                self._models[model_key] = WhisperModel(
                    config.model_size.value,
                    device=config.device,
                    compute_type=config.compute_type,
                    num_workers=config.num_workers
                )
                logger.info(f"Model {config.model_size.value} loaded successfully")
            except Exception as e:
                if "Invalid model size 'turbo'" in str(e):
                    raise ValueError(
                        "The 'turbo' model is not supported by faster-whisper. "
                        "Available models: tiny, base, small, medium, large-v1, large-v2, large-v3"
                    ) from e
                else:
                    raise e
        
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
        model_load_start = time.time()
        model = await asyncio.get_event_loop().run_in_executor(
            None, self.model_manager.get_model, model_config
        )
        model_load_time = time.time() - model_load_start
        
        if progress_callback:
            file_size = get_file_size_mb(audio_path)
            await progress_callback(f"Model loaded in {model_load_time:.1f}s. Starting transcription ({file_size:.1f} MB)...")
        
        # Get audio duration for progress calculation
        try:
            total_duration = get_video_duration(audio_path)
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            total_duration = None
        
        # Create progress queue for real-time updates (like main_old.py)
        progress_queue = asyncio.Queue()
        is_transcribing = True
        processing_speed_history = []
        start_time = time.time()
        transcription_done = False
        
        # Progress monitoring task (adapted from main_old.py)
        async def update_progress():
            last_update = time.time()
            last_segment_time = 0
            last_progress = 0
            
            while is_transcribing and not transcription_done:
                try:
                    # Get latest progress update
                    try:
                        segment_end = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                        if segment_end > last_segment_time:  # Only update if time increases
                            last_segment_time = segment_end
                    except asyncio.TimeoutError:
                        pass
                    
                    current_time = time.time()
                    if current_time - last_update >= 0.5 and total_duration and progress_callback:  # Update every 0.5s
                        # Calculate progress based on audio position
                        current_progress = min(0.95, last_segment_time / total_duration)
                        
                        # Only update if progress increases
                        if current_progress >= last_progress:
                            elapsed_time = current_time - start_time
                            
                            # Calculate processing speed (seconds of audio per second of real time)
                            if last_segment_time > 0:
                                current_speed = last_segment_time / elapsed_time
                                
                                # Update moving average of processing speed
                                processing_speed_history.append(current_speed)
                                if len(processing_speed_history) > 5:  # Keep last 5 measurements
                                    processing_speed_history.pop(0)
                                
                                # Calculate average speed
                                avg_speed = sum(processing_speed_history) / len(processing_speed_history)
                                
                                # Calculate remaining time using average speed
                                remaining_audio = total_duration - last_segment_time
                                remaining_seconds = remaining_audio / avg_speed if avg_speed > 0 else 0
                                
                                # Add a small buffer for processing overhead
                                remaining_seconds *= 1.1
                                
                                if remaining_seconds > 1:
                                    remaining_formatted = format_time(max(0, remaining_seconds))
                                    progress_percent = min(95, current_progress * 100)
                                    await progress_callback(f"Transcribing audio... {remaining_formatted} remaining ({progress_percent:.0f}%)")
                            
                            last_progress = current_progress
                        last_update = current_time
                    
                    if transcription_done:
                        break
                        
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.debug(f"Progress update error: {e}")
                    await asyncio.sleep(0.1)
                    continue

        # Start progress update task
        progress_task = asyncio.create_task(update_progress())
        
        try:
            # Run transcription in thread pool (adapted from main_old.py)
            loop = asyncio.get_event_loop()
            def run_transcription():
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
                
                segments, info = model.transcribe(audio_path, **transcription_options)
                
                logger.info(f"Transcription info - Language: {info.language} (probability: {info.language_probability:.2f})")
                logger.info(f"Duration: {format_time(info.duration)}")
                
                # Process segments and track progress (like main_old.py)
                all_segments = []
                for segment in segments:
                    all_segments.append(segment)
                    # Update progress based on segment end time
                    loop.call_soon_threadsafe(
                        progress_queue.put_nowait,
                        segment.end
                    )
                
                return all_segments, info
            
            segments, info = await loop.run_in_executor(None, run_transcription)
            
            # Wait for progress task to complete
            try:
                await asyncio.wait_for(progress_task, timeout=1.0)
            except asyncio.TimeoutError:
                pass
            
            # Show completion and process segments
            if progress_callback:
                total_time = time.time() - start_time
                await progress_callback(f"Transcription completed in {format_time(total_time)}")
            
            # Process and yield transcript segments with timestamps
            segment_count = 0
            for segment in segments:
                segment_count += 1
                
                # Format segment with timestamps
                start_timestamp = format_timestamp(segment.start)
                end_timestamp = format_timestamp(segment.end)
                segment_text = f"{start_timestamp} --> {end_timestamp}\n{segment.text}\n\n"
                
                yield segment_text
            
            logger.info(f"Transcription completed: {segment_count} segments in {format_time(time.time() - start_time)}")
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            if progress_callback:
                await progress_callback(f"Transcription failed: {str(e)}")
            raise
        finally:
            transcription_done = True
            is_transcribing = False
    
    async def transcribe_with_progress_and_segments(
        self,
        audio_path: str,
        prompt: str = "",
        model_size: str = "",
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> tuple[str, list]:
        """
        Transcribe audio with progress updates and return both text and segment objects
        
        Args:
            audio_path: Path to audio file
            prompt: Initial prompt for transcription
            model_size: Specific model size to use (e.g., "base", "large-v3")
            progress_callback: Callback for progress updates
            
        Returns:
            Tuple of (complete_transcript, segments_list)
        """
        # Select optimal model configuration
        model_config = self._select_model_config(audio_path, model_size)
        # Store the model config for later reference
        self._current_model_config = model_config
        
        if progress_callback:
            await progress_callback(f"Loading {model_config.model_size.value} model...")
        
        # Load model in thread pool to avoid blocking
        model_load_start = time.time()
        model = await asyncio.get_event_loop().run_in_executor(
            None, self.model_manager.get_model, model_config
        )
        model_load_time = time.time() - model_load_start
        
        if progress_callback:
            file_size = get_file_size_mb(audio_path)
            await progress_callback(f"Model loaded in {model_load_time:.1f}s. Starting transcription ({file_size:.1f} MB)...")
        
        # Get audio duration for progress calculation
        try:
            total_duration = get_video_duration(audio_path)
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            total_duration = None
        
        # Create progress queue for real-time updates
        progress_queue = asyncio.Queue()
        is_transcribing = True
        processing_speed_history = []
        start_time = time.time()
        transcription_done = False
        
        # Progress monitoring task
        async def update_progress():
            last_update = time.time()
            last_segment_time = 0
            last_progress = 0
            
            while is_transcribing and not transcription_done:
                try:
                    # Get latest progress update
                    try:
                        segment_end = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                        if segment_end > last_segment_time:  # Only update if time increases
                            last_segment_time = segment_end
                    except asyncio.TimeoutError:
                        pass
                    
                    current_time = time.time()
                    if current_time - last_update >= 0.5 and total_duration and progress_callback:  # Update every 0.5s
                        # Calculate progress based on audio position
                        current_progress = min(0.95, last_segment_time / total_duration)
                        
                        # Only update if progress increases
                        if current_progress >= last_progress:
                            elapsed_time = current_time - start_time
                            
                            # Calculate processing speed (seconds of audio per second of real time)
                            if last_segment_time > 0:
                                current_speed = last_segment_time / elapsed_time
                                
                                # Update moving average of processing speed
                                processing_speed_history.append(current_speed)
                                if len(processing_speed_history) > 5:  # Keep last 5 measurements
                                    processing_speed_history.pop(0)
                                
                                # Calculate average speed
                                avg_speed = sum(processing_speed_history) / len(processing_speed_history)
                                
                                # Calculate remaining time using average speed
                                remaining_audio = total_duration - last_segment_time
                                remaining_seconds = remaining_audio / avg_speed if avg_speed > 0 else 0
                                
                                # Add a small buffer for processing overhead
                                remaining_seconds *= 1.1
                                
                                if remaining_seconds > 1:
                                    remaining_formatted = format_time(max(0, remaining_seconds))
                                    progress_percent = min(95, current_progress * 100)
                                    await progress_callback(f"Transcribing audio... {remaining_formatted} remaining ({progress_percent:.0f}%)")
                            
                            last_progress = current_progress
                        last_update = current_time
                    
                    if transcription_done:
                        break
                        
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.debug(f"Progress update error: {e}")
                    await asyncio.sleep(0.1)
                    continue

        # Start progress update task
        progress_task = asyncio.create_task(update_progress())
        
        try:
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            def run_transcription():
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
                
                segments, info = model.transcribe(audio_path, **transcription_options)
                
                logger.info(f"Transcription info - Language: {info.language} (probability: {info.language_probability:.2f})")
                logger.info(f"Duration: {format_time(info.duration)}")
                
                # Process segments and track progress
                all_segments = []
                for segment in segments:
                    all_segments.append(segment)
                    # Update progress based on segment end time
                    loop.call_soon_threadsafe(
                        progress_queue.put_nowait,
                        segment.end
                    )
                
                return all_segments, info
            
            segments, info = await loop.run_in_executor(None, run_transcription)
            
            # Wait for progress task to complete
            try:
                await asyncio.wait_for(progress_task, timeout=1.0)
            except asyncio.TimeoutError:
                pass
            
            # Show completion
            if progress_callback:
                total_time = time.time() - start_time
                await progress_callback(f"Transcription completed in {format_time(total_time)}")
            
            # Generate complete transcript
            complete_transcript = ""
            for segment in segments:
                # Format segment with timestamps
                start_timestamp = format_timestamp(segment.start)
                end_timestamp = format_timestamp(segment.end)
                segment_text = f"{start_timestamp} --> {end_timestamp}\n{segment.text}\n\n"
                complete_transcript += segment_text
            
            logger.info(f"Transcription completed: {len(segments)} segments in {format_time(time.time() - start_time)}")
            
            return complete_transcript.strip(), segments
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            if progress_callback:
                await progress_callback(f"Transcription failed: {str(e)}")
            raise
        finally:
            transcription_done = True
            is_transcribing = False
    
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
            # Check for unsupported models
            if model_size.lower() == "turbo":
                raise ValueError(
                    "The 'turbo' model is not supported by faster-whisper. "
                    "Please use 'large-v3', 'large-v2', 'medium', 'base', 'small', or 'tiny' instead."
                )
            
            try:
                # Convert string to ModelSize enum
                from config import ModelSize
                size_mapping = {
                    "tiny": ModelSize.TINY,
                    "base": ModelSize.BASE,
                    "small": ModelSize.SMALL,
                    "medium": ModelSize.MEDIUM,
                    "large-v2": ModelSize.LARGE,
                    "large-v3": ModelSize.LARGE_V3
                }
                
                if model_size in size_mapping:
                    requested_model = size_mapping[model_size]
                    logger.info(f"Using specifically requested {model_size} model")
                    
                    # Create appropriate config for the requested model
                    if self.config.hardware.has_gpu:
                        device = "cuda"
                        compute_type = "float16"
                        num_workers = 1 if model_size in ["large-v3", "large-v2"] else 2
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
    
    def _get_model_speed_factor(self, model_size) -> float:
        """Get estimated processing speed factor for different models"""
        # These are rough estimates of how many seconds of audio can be processed per second
        speed_factors = {
            "tiny": 8.0,     # Very fast
            "base": 4.0,     # Fast  
            "small": 2.5,    # Medium-fast
            "medium": 1.5,   # Medium
            "large-v2": 0.8, # Slower
            "large-v3": 0.7  # Slower
        }
        
        model_name = model_size.value if hasattr(model_size, 'value') else str(model_size)
        return speed_factors.get(model_name, 1.0)

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
        
        # Check if transcription already exists
        if os.path.exists(transcript_path):
            error_msg = (
                f"Transcription already exists: {transcript_filename}\n"
                f"Please delete the existing transcription file first, then try again."
            )
            logger.warning(error_msg)
            raise FileExistsError(error_msg)
        
        try:
            # Perform transcription with progress updates and get segments in one pass
            if progress_callback:
                await progress_callback("Starting transcription...")
            
            # Get both transcript and segments in a single transcription run
            streaming_transcript, segments = await self.transcriber.transcribe_with_progress_and_segments(
                audio_path, prompt, model_size, progress_callback
            )
            
            # Format with enhanced transcription function
            if progress_callback:
                await progress_callback("Formatting transcription...")
            
            from utils import format_enhanced_transcription
            
            # Get model info for formatting
            model_name = model_size if model_size else self.config.model_config.model_size.value
            
            # Format with enhanced transcription function using the segments we already have
            enhanced_transcript = format_enhanced_transcription(segments, audio_filename, model_name)
            
            # Save transcript
            if progress_callback:
                await progress_callback("Saving transcription...")
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_transcript)
            
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