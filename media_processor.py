import os
import subprocess
import asyncio
import time
import shutil
from typing import Optional, AsyncGenerator, Callable
import logging

from utils import (
    is_audio_file, is_video_file, get_video_duration, 
    generate_output_filename, format_time, ProgressTracker,
    get_file_size_mb
)
from config import Config

logger = logging.getLogger(__name__)

class MediaProcessor:
    """Handle video to audio conversion with GPU acceleration and progress tracking"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def process_media_file(
        self, 
        file_path: str, 
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Process media file (video or audio) and return path to audio file
        
        Args:
            file_path: Path to input media file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to processed audio file
        """
        try:
            if is_audio_file(file_path):
                return await self._process_audio_file(file_path, progress_callback)
            elif is_video_file(file_path):
                return await self._process_video_file(file_path, progress_callback)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
        except Exception as e:
            logger.error(f"Error processing media file {file_path}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    async def _process_audio_file(
        self, 
        file_path: str, 
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Process audio file by copying to output directory"""
        try:
            if progress_callback:
                await progress_callback("Preparing audio file...")
            
            # Small delay to allow upload completion message to be seen
            await asyncio.sleep(0.8)
            
            audio_filename = os.path.basename(file_path)
            audio_file_path = os.path.join(self.config.output_folder, audio_filename)
            
            # Copy audio file to output folder
            shutil.copy2(file_path, audio_file_path)
            
            if progress_callback:
                await progress_callback("Audio file ready for transcription!")
            
            logger.info(f"Audio file copied successfully: {audio_filename}")
            return audio_file_path
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            raise
    
    async def _process_video_file(
        self, 
        file_path: str, 
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Convert video file to audio with progress tracking"""
        try:
            # Generate output filename
            audio_file_path = generate_output_filename(
                file_path, self.config.output_folder, '.mp3'
            )
            
            # Get video duration for progress calculation
            try:
                total_duration = get_video_duration(file_path)
                if total_duration <= 0:
                    raise ValueError("Invalid video duration")
            except Exception as e:
                logger.warning(f"Could not get video duration: {e}")
                total_duration = None
            
            # Build FFmpeg command
            command = self._build_ffmpeg_command(file_path, audio_file_path)
            
            if progress_callback:
                file_size = get_file_size_mb(file_path)
                await progress_callback(f"Converting video to audio... ({file_size:.1f} MB)")
            
            # Execute conversion with progress tracking
            await self._execute_ffmpeg_conversion(
                command, total_duration, progress_callback
            )
            
            if progress_callback:
                await progress_callback("Video conversion completed!")
            
            logger.info(f"Video converted successfully: {os.path.basename(audio_file_path)}")
            return audio_file_path
            
        except Exception as e:
            logger.error(f"Error converting video file: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Clean up partial file if it exists
            if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            raise
    
    def _build_ffmpeg_command(self, input_path: str, output_path: str) -> list:
        """Build FFmpeg command with appropriate settings and fallback strategy"""
        import platform
        
        # On Windows, try to get short path names to avoid space issues
        if platform.system() == "Windows":
            logger.info(f"Running on Windows, checking paths for spaces...")
            logger.info(f"Input path: {input_path}")
            logger.info(f"Output path: {output_path}")
            
            try:
                import ctypes
                from ctypes import wintypes
                
                def get_short_path_name(long_name):
                    """Get Windows short path name (8.3 format) to avoid space issues"""
                    logger.debug(f"Attempting to get short path for: {long_name}")
                    buf_size = ctypes.windll.kernel32.GetShortPathNameW(long_name, None, 0)
                    if buf_size:
                        buf = ctypes.create_unicode_buffer(buf_size)
                        ctypes.windll.kernel32.GetShortPathNameW(long_name, buf, buf_size)
                        short_path = buf.value
                        logger.debug(f"Short path result: {short_path}")
                        return short_path
                    else:
                        logger.debug(f"No short path available for: {long_name}")
                        return long_name
                
                # Try to get short path names for paths with spaces
                if ' ' in input_path:
                    logger.info(f"Input path contains spaces, getting short path...")
                    short_input = get_short_path_name(input_path)
                    if short_input and short_input != input_path:
                        logger.info(f"Using short path for input: {short_input} (was: {input_path})")
                        input_path = short_input
                    else:
                        logger.warning(f"Could not get short path for input: {input_path}")
                
                if ' ' in output_path:
                    logger.info(f"Output path contains spaces, getting short path...")
                    short_output = get_short_path_name(output_path)
                    if short_output and short_output != output_path:
                        logger.info(f"Using short path for output: {short_output} (was: {output_path})")
                        output_path = short_output
                    else:
                        logger.warning(f"Could not get short path for output: {output_path}")
                        
            except Exception as e:
                logger.warning(f"Could not get short path names: {e}")
                logger.warning(f"Exception type: {type(e).__name__}")
                # Fallback: convert backslashes to forward slashes
                input_path = input_path.replace('\\', '/')
                output_path = output_path.replace('\\', '/')
                logger.info(f"Using fallback paths - Input: {input_path}, Output: {output_path}")
        
        command = ['ffmpeg', '-i', input_path]
        
        # Don't use hardware acceleration for audio extraction to avoid conflicts
        # Hardware acceleration is primarily for video processing
        logger.info("Building FFmpeg command for audio extraction (CPU-only for compatibility)")
        
        # Audio extraction settings - no hardware acceleration needed
        command.extend([
            '-vn',  # No video
            '-acodec', 'libmp3lame',  # MP3 codec
            '-q:a', '2',  # High quality
            '-ar', '22050',  # Sample rate
            '-y',  # Overwrite output file
            '-progress', 'pipe:2',  # Send progress to stderr
            '-nostats',  # No statistics
            '-loglevel', 'error',  # Only show errors
            output_path
        ])
        
        logger.info(f"Built FFmpeg command: {' '.join(command)}")
        return command
    
    async def _execute_ffmpeg_conversion(
        self, 
        command: list, 
        total_duration: Optional[float], 
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """Execute FFmpeg conversion with enhanced error handling and retry logic"""
        start_time = time.time()
        max_retries = 2
        retry_count = 0
        
        # First, verify FFmpeg is available using synchronous subprocess for uvicorn compatibility
        try:
            test_result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                cwd=self.config.project_root,
                timeout=10
            )
            if test_result.returncode != 0:
                logger.error(f"FFmpeg not available or not working. Return code: {test_result.returncode}")
                logger.error(f"FFmpeg test stderr: {test_result.stderr.decode('utf-8', errors='ignore')}")
                raise FileNotFoundError("FFmpeg is not available or not working properly")
            else:
                logger.info("FFmpeg availability confirmed")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"FFmpeg not found or not working: {e}")
            raise FileNotFoundError("FFmpeg is not installed or not in PATH")
        
        while retry_count <= max_retries:
            try:
                logger.info(f"Executing FFmpeg command (attempt {retry_count + 1}/{max_retries + 1}): {' '.join(command)}")
                
                # Use synchronous subprocess in thread pool for uvicorn compatibility
                loop = asyncio.get_event_loop()
                process_result = await loop.run_in_executor(
                    None, 
                    self._run_ffmpeg_sync, 
                    command, 
                    total_duration, 
                    start_time, 
                    progress_callback
                )
                
                if process_result['returncode'] == 0:
                    logger.info("FFmpeg conversion completed successfully")
                    if progress_callback:
                        await progress_callback("Video conversion completed!")
                    return
                
                # Handle errors
                stdout = process_result.get('stdout', b'')
                stderr = process_result.get('stderr', b'')
                    
                # Handle specific error codes
                error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown error"
                stdout_msg = stdout.decode('utf-8', errors='ignore') if stdout else ""
                
                # Log detailed error information
                logger.error(f"FFmpeg failed with return code: {process_result['returncode']}")
                logger.error(f"Command was: {' '.join(command)}")
                logger.error(f"stderr output: '{error_msg}'")
                logger.error(f"stdout output: '{stdout_msg}'")
                
                # Check if input file exists and is readable
                input_file = command[2]  # ffmpeg -i <input_file>
                if not os.path.exists(input_file):
                    logger.error(f"Input file does not exist: {input_file}")
                    raise FileNotFoundError(f"Input file not found: {input_file}")
                
                # Check if output directory exists and is writable
                output_file = command[-1]
                output_dir = os.path.dirname(output_file)
                if not os.path.exists(output_dir):
                    logger.error(f"Output directory does not exist: {output_dir}")
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Created output directory: {output_dir}")
                elif not os.access(output_dir, os.W_OK):
                    logger.error(f"Output directory is not writable: {output_dir}")
                    raise PermissionError(f"Cannot write to output directory: {output_dir}")
                
                # Convert return code to handle unsigned int overflow (4294967274 = -22 as signed int)
                return_code_signed = process_result['returncode'] if process_result['returncode'] < 2147483648 else process_result['returncode'] - 4294967296
                
                if (process_result['returncode'] == 4294967274 or return_code_signed == -22 or 
                    "hardware" in error_msg.lower() or "Invalid argument" in error_msg or
                    "Failed to create" in error_msg or "CUDA" in error_msg):
                    logger.warning(f"Hardware/codec issue detected (return code: {process_result['returncode']}, attempt {retry_count + 1}), trying enhanced CPU-only approach")
                    if retry_count < max_retries:
                        # Rebuild command without any hardware acceleration hints
                        command = self._build_cpu_only_command(command[2], command[-1])  # input and output paths
                        retry_count += 1
                        continue
                
                # If all retries failed or other error
                logger.error(f"FFmpeg failed with return code {process_result['returncode']}: {error_msg}")
                raise subprocess.CalledProcessError(process_result['returncode'], command, stderr)
                    
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"All FFmpeg attempts failed: {e}")
                    raise
                
                logger.warning(f"FFmpeg attempt {retry_count} failed: {e}, retrying...")
                await asyncio.sleep(1)
    
    async def _monitor_ffmpeg_with_progress(
        self,
        process: asyncio.subprocess.Process,
        total_duration: Optional[float],
        start_time: float,
        progress_callback: Optional[Callable[[str], None]]
    ) -> tuple:
        """Monitor FFmpeg progress while waiting for completion"""
        stdout_data = b""
        stderr_data = b""
        last_progress = 0
        last_update_time = start_time
        
        try:
            while True:
                # Check if process is done
                if process.returncode is not None:
                    break
                
                # Read available data without blocking
                try:
                    # Read stdout if available
                    if process.stdout:
                        try:
                            stdout_chunk = await asyncio.wait_for(process.stdout.read(8192), timeout=0.1)
                            if stdout_chunk:
                                stdout_data += stdout_chunk
                        except asyncio.TimeoutError:
                            pass
                    
                    # Read stderr if available
                    if process.stderr:
                        try:
                            stderr_chunk = await asyncio.wait_for(process.stderr.read(8192), timeout=0.1)
                            if stderr_chunk:
                                stderr_data += stderr_chunk
                                
                                # Parse progress from stderr
                                if total_duration and progress_callback:
                                    self._parse_and_update_progress(
                                        stderr_chunk, total_duration, start_time, 
                                        last_progress, last_update_time, progress_callback
                                    )
                                    
                        except asyncio.TimeoutError:
                            pass
                            
                except Exception as e:
                    logger.debug(f"Read error: {e}")
                
                # Check if process finished
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.1)
                    break
                except asyncio.TimeoutError:
                    pass
                
                await asyncio.sleep(0.1)
            
            # Read any remaining data
            if process.stdout:
                remaining_stdout = await process.stdout.read()
                stdout_data += remaining_stdout
            
            if process.stderr:
                remaining_stderr = await process.stderr.read()
                stderr_data += remaining_stderr
                
        except Exception as e:
            logger.debug(f"Progress monitoring error: {e}")
            # Fallback to simple wait
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(), timeout=300
                )
            except asyncio.TimeoutError:
                process.terminate()
                await asyncio.sleep(1)
                if process.returncode is None:
                    process.kill()
                raise TimeoutError("FFmpeg conversion timed out")
        
        return stdout_data, stderr_data
    
    def _run_ffmpeg_sync(
        self, 
        command: list, 
        total_duration: Optional[float], 
        start_time: float, 
        progress_callback: Optional[Callable[[str], None]]
    ) -> dict:
        """Run FFmpeg synchronously with progress monitoring (for uvicorn compatibility)"""
        try:
            # Start the process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.config.project_root,
                universal_newlines=False,
                bufsize=0  # Unbuffered for real-time output
            )
            
            # Monitor progress
            stdout_data = b""
            stderr_data = b""
            last_progress_time = start_time
            
            # Read output in chunks with progress monitoring
            while True:
                # Check if process is still running
                poll_result = process.poll()
                if poll_result is not None:
                    # Process finished, read remaining output
                    try:
                        remaining_stdout, remaining_stderr = process.communicate(timeout=5)
                        stdout_data += remaining_stdout
                        stderr_data += remaining_stderr
                    except subprocess.TimeoutExpired:
                        process.kill()
                        remaining_stdout, remaining_stderr = process.communicate()
                        stdout_data += remaining_stdout
                        stderr_data += remaining_stderr
                    break
                
                # Read available data
                import sys
                
                if sys.platform == "win32":
                    # On Windows, use threading to read stderr without blocking
                    time.sleep(0.1)
                    
                    # Try to read stderr for actual FFmpeg progress
                    try:
                        # Use a more robust approach for Windows
                        import threading
                        import queue
                        
                        def read_stderr(process, stderr_queue):
                            """Read stderr in a separate thread"""
                            try:
                                while process.poll() is None:
                                    chunk = process.stderr.read(1024)
                                    if chunk:
                                        stderr_queue.put(chunk)
                                    else:
                                        break
                            except:
                                pass
                        
                        # Start stderr reading thread if not already started
                        if not hasattr(self, '_stderr_queue'):
                            self._stderr_queue = queue.Queue()
                            self._stderr_thread = threading.Thread(target=read_stderr, args=(process, self._stderr_queue))
                            self._stderr_thread.daemon = True
                            self._stderr_thread.start()
                        
                        # Check for new stderr data
                        try:
                            while True:
                                chunk = self._stderr_queue.get_nowait()
                                stderr_data += chunk
                                
                                # Parse progress from actual FFmpeg output
                                if total_duration and progress_callback:
                                    current_time = time.time()
                                    if current_time - last_progress_time >= 0.5:
                                        self._parse_progress_sync(
                                            chunk, total_duration, start_time, progress_callback
                                        )
                                        
                                        # Send the progress update if we have one
                                        if hasattr(self, '_last_progress_message'):
                                            try:
                                                import asyncio
                                                import threading
                                                
                                                def run_callback():
                                                    try:
                                                        loop = asyncio.new_event_loop()
                                                        asyncio.set_event_loop(loop)
                                                        loop.run_until_complete(progress_callback(self._last_progress_message))
                                                        loop.close()
                                                    except:
                                                        pass
                                                
                                                callback_thread = threading.Thread(target=run_callback)
                                                callback_thread.daemon = True
                                                callback_thread.start()
                                                
                                            except Exception as callback_error:
                                                logger.debug(f"Progress callback error: {callback_error}")
                                        
                                        last_progress_time = current_time
                        except queue.Empty:
                            pass
                            
                    except Exception as e:
                        logger.debug(f"Windows stderr reading error: {e}")
                        # Fallback to simple waiting
                        time.sleep(0.1)
                            
                else:
                    # Unix-like systems can use select
                    import select
                    ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
                    for stream in ready:
                        chunk = stream.read(8192)
                        if chunk:
                            if stream == process.stdout:
                                stdout_data += chunk
                            else:
                                stderr_data += chunk
                                if total_duration and progress_callback:
                                    current_time = time.time()
                                    if current_time - last_progress_time >= 0.5:
                                        self._parse_progress_sync(
                                            chunk, total_duration, start_time, progress_callback
                                        )
                                        last_progress_time = current_time
            
            return {
                'returncode': process.returncode,
                'stdout': stdout_data,
                'stderr': stderr_data
            }
            
        except Exception as e:
            logger.error(f"Error in synchronous FFmpeg execution: {e}")
            raise
    
    def _parse_progress_sync(
        self,
        stderr_chunk: bytes,
        total_duration: float,
        start_time: float,
        progress_callback: Callable[[str], None]
    ) -> None:
        """Parse progress from stderr chunk for sync execution with ETA calculation"""
        try:
            stderr_text = stderr_chunk.decode('utf-8', errors='ignore')
            for line in stderr_text.split('\n'):
                if 'out_time_ms=' in line:
                    try:
                        current_time = float(line.strip().split('=')[1]) / 1000000
                        current_time_real = time.time()
                        
                        # Calculate remaining time based on actual FFmpeg progress
                        elapsed_time = current_time_real - start_time
                        if elapsed_time > 0 and current_time > 0:
                            # Speed is how much video time is processed per real time
                            speed = current_time / elapsed_time
                            remaining_video_time = total_duration - current_time
                            remaining_real_time = remaining_video_time / speed if speed > 0 else 0
                            
                            # Only update if we have meaningful progress and reasonable remaining time
                            if remaining_real_time > 1 and current_time_real - getattr(self, '_last_update_time', 0) >= 1:
                                remaining_formatted = format_time(max(0, remaining_real_time))
                                progress_percent = (current_time / total_duration) * 100 if total_duration > 0 else 0
                                
                                # Store the message for later async callback
                                self._last_progress_message = f"Converting video to audio... {remaining_formatted} remaining"
                                self._last_update_time = current_time_real
                            elif remaining_real_time <= 1:
                                # Almost done
                                self._last_progress_message = "Converting video to audio... Almost done!"
                                    
                    except (ValueError, ZeroDivisionError, IndexError):
                        continue
                        
        except Exception as e:
            logger.debug(f"Progress parsing error in sync mode: {e}")
    
    def _parse_and_update_progress(
        self,
        stderr_chunk: bytes,
        total_duration: float,
        start_time: float,
        last_progress: float,
        last_update_time: float,
        progress_callback: Callable[[str], None]
    ) -> None:
        """Parse progress from stderr chunk and update if needed"""
        try:
            stderr_text = stderr_chunk.decode('utf-8', errors='ignore')
            for line in stderr_text.split('\n'):
                if 'out_time_ms=' in line:
                    try:
                        current_time = float(line.strip().split('=')[1]) / 1000000
                        if current_time > last_progress:
                            current_time_real = time.time()
                            
                            # Calculate remaining time
                            elapsed_time = current_time_real - start_time
                            if elapsed_time > 0:
                                speed = current_time / elapsed_time
                                remaining_time = (total_duration - current_time) / speed
                                
                                # Only update if significant progress and reasonable remaining time
                                if remaining_time > 1 and current_time_real - last_update_time >= 1:
                                    remaining_formatted = format_time(max(0, remaining_time))
                                    # Schedule callback to avoid blocking
                                    asyncio.create_task(progress_callback(f"Converting video to audio... {remaining_formatted} remaining"))
                                    last_update_time = current_time_real
                                elif remaining_time <= 1:
                                    break
                                    
                    except (ValueError, ZeroDivisionError, IndexError):
                        continue
                        
        except Exception as e:
            logger.debug(f"Progress parsing error: {e}")
    
    def _build_cpu_only_command(self, input_path: str, output_path: str) -> list:
        """Build a fallback FFmpeg command using only CPU processing with maximum compatibility"""
        logger.info("Building enhanced CPU-only FFmpeg command as fallback")
        
        import platform
        
        # On Windows, try to get short path names to avoid space issues
        if platform.system() == "Windows":
            try:
                import ctypes
                
                def get_short_path_name(long_name):
                    """Get Windows short path name (8.3 format) to avoid space issues"""
                    buf_size = ctypes.windll.kernel32.GetShortPathNameW(long_name, None, 0)
                    if buf_size:
                        buf = ctypes.create_unicode_buffer(buf_size)
                        ctypes.windll.kernel32.GetShortPathNameW(long_name, buf, buf_size)
                        return buf.value
                    return long_name
                
                # Try to get short path names for paths with spaces
                if ' ' in input_path:
                    short_input = get_short_path_name(input_path)
                    if short_input and short_input != input_path:
                        logger.info(f"Using short path for CPU fallback input: {short_input} (was: {input_path})")
                        input_path = short_input
                
                if ' ' in output_path:
                    short_output = get_short_path_name(output_path)
                    if short_output and short_output != output_path:
                        logger.info(f"Using short path for CPU fallback output: {short_output} (was: {output_path})")
                        output_path = short_output
                        
            except Exception as e:
                logger.warning(f"Could not get short path names for CPU fallback: {e}")
                # Fallback: convert backslashes to forward slashes
                input_path = input_path.replace('\\', '/')
                output_path = output_path.replace('\\', '/')
        
        command = [
            'ffmpeg', 
            '-threads', '1',  # Force single thread to avoid issues
            '-i', input_path,
            '-vn',  # No video
            '-acodec', 'libmp3lame',  # MP3 codec
            '-b:a', '96k',  # Lower bitrate for better compatibility
            '-ar', '22050',  # Sample rate
            '-ac', '1',  # Mono audio (smaller file, often fine for transcription)
            '-f', 'mp3',  # Force MP3 format
            '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
            '-y',  # Overwrite output file
            '-nostats',  # No statistics
            '-loglevel', 'info',  # More verbose for debugging
            output_path
        ]
        
        logger.info(f"Built CPU-only FFmpeg command: {' '.join(command)}")
        return command
    
    async def _read_lines(self, stream) -> AsyncGenerator[str, None]:
        """Async generator to read lines from stream"""
        while True:
            line = await stream.readline()
            if not line:
                break
            yield line.decode('utf-8').strip()
    
    def _parse_ffmpeg_progress(self, line: str, total_duration: Optional[float]) -> Optional[dict]:
        """Parse FFmpeg progress line and return progress information"""
        if 'out_time_ms=' not in line:
            return None
        
        try:
            # Extract current time in microseconds
            time_ms_str = line.split('out_time_ms=')[1].split()[0]
            current_time_us = int(time_ms_str)
            current_time_s = current_time_us / 1_000_000
            
            if not total_duration or total_duration <= 0:
                return {'current_time': current_time_s}
            
            # Calculate progress percentage
            progress_percent = min((current_time_s / total_duration) * 100, 100)
            
            # Estimate remaining time
            if current_time_s > 0:
                processing_speed = current_time_s / time.time()  # Approximate
                remaining_content = total_duration - current_time_s
                eta_seconds = remaining_content / processing_speed if processing_speed > 0 else None
            else:
                eta_seconds = None
            
            return {
                'current_time': current_time_s,
                'total_duration': total_duration,
                'percent': progress_percent,
                'eta_seconds': eta_seconds
            }
            
        except (ValueError, IndexError) as e:
            # Ignore parsing errors for invalid lines
            return None
    
    def get_supported_formats(self) -> dict:
        """Get supported input and output formats"""
        return {
            'input': {
                'video': ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'mpeg', 'mpg', '3gp'],
                'audio': ['mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac', 'wma']
            },
            'output': {
                'audio': ['mp3']  # Currently only MP3 output
            }
        }
    
    async def get_media_info(self, file_path: str) -> dict:
        """Get detailed media file information using FFprobe"""
        command = [
            'ffprobe', 
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            file_path
        ]
        
        try:
            # Use synchronous subprocess for uvicorn compatibility
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    command,
                    capture_output=True,
                    cwd=self.config.project_root,
                    timeout=30
                )
            )
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, command, result.stderr.decode('utf-8')
                )
            
            import json
            return json.loads(result.stdout.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error getting media info for {file_path}: {e}")
            raise 