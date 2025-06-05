import os
import subprocess
import time
import shutil
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Enhanced file validation"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in allowed_extensions

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def get_video_duration(video_file_path: str) -> float:
    """Get the duration of the video using ffprobe."""
    ffprobe_command = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_file_path
    ]

    try:
        output = subprocess.check_output(ffprobe_command).strip().decode('utf-8')
        if output == 'N/A' or not output:
            raise ValueError(f"Could not retrieve duration for file: {video_file_path}")
        duration = float(output)
        return duration
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running ffprobe: {e}")
        raise
    except ValueError as e:
        logger.error(f"Error parsing duration: {e}")
        raise

def format_time(seconds: float) -> str:
    """Format seconds into a human-readable time string."""
    if seconds < 60:
        return f"{int(seconds)} second{'s' if seconds != 1 else ''}"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''} and {remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = int(seconds % 60)
        return f"{hours} hour{'s' if hours > 1 else ''}, {minutes} minute{'s' if minutes > 1 else ''}, and {remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"

def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def clean_folder_contents(folder: str) -> None:
    """Delete all contents of a folder."""
    try:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except Exception as e:
        logger.error(f"Error cleaning folder {folder}: {e}")

def is_audio_file(file_path: str) -> bool:
    """Check if file is an audio file based on extension"""
    audio_extensions = {'.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac', '.wma'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in audio_extensions

def is_video_file(file_path: str) -> bool:
    """Check if file is a video file based on extension"""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg', '.mpg', '.3gp'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in video_extensions

def generate_output_filename(input_path: str, output_dir: str, extension: str = '.mp3') -> str:
    """Generate output filename based on input file"""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{base_name}{extension}")

def estimate_processing_time(file_size_mb: float, is_gpu: bool = False) -> float:
    """Estimate processing time based on file size and hardware"""
    # Rough estimates based on typical performance
    if is_gpu:
        # GPU processing is typically 2-4x faster
        time_per_mb = 2.0  # seconds per MB
    else:
        # CPU processing
        time_per_mb = 8.0  # seconds per MB
    
    return file_size_mb * time_per_mb

class ProgressTracker:
    """Track and format progress information"""
    
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        
    def update(self, step: int) -> dict:
        """Update progress and return formatted information"""
        self.current_step = min(step, self.total_steps)
        current_time = time.time()
        
        elapsed_time = current_time - self.start_time
        progress_percent = (self.current_step / self.total_steps) * 100
        
        # Estimate remaining time
        if self.current_step > 0:
            time_per_step = elapsed_time / self.current_step
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = time_per_step * remaining_steps
        else:
            estimated_remaining = 0
        
        self.last_update_time = current_time
        
        return {
            'progress_percent': round(progress_percent, 1),
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'elapsed_time': elapsed_time,
            'estimated_remaining': estimated_remaining,
            'elapsed_formatted': format_time(elapsed_time),
            'remaining_formatted': format_time(estimated_remaining)
        }
    
    def is_complete(self) -> bool:
        """Check if processing is complete"""
        return self.current_step >= self.total_steps

def validate_environment() -> dict:
    """Validate that required tools are available"""
    requirements = {
        'ffmpeg': False,
        'ffprobe': False
    }
    
    # Check for ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
        requirements['ffmpeg'] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check for ffprobe
    try:
        subprocess.run(['ffprobe', '-version'], 
                      capture_output=True, check=True)
        requirements['ffprobe'] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return requirements

def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing problematic characters"""
    import re
    # Remove or replace problematic characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe_name = re.sub(r'\s+', '_', safe_name)  # Replace spaces with underscores
    return safe_name[:255]  # Limit to 255 characters

def format_enhanced_transcription(segments, audio_filename: str, model_name: str) -> str:
    """
    Format transcription with enhanced sentence grouping and professional timestamps
    Based on industry best practices for subtitle/caption formatting
    """
    
    # Header information
    output = f"Transcription of: {audio_filename}\n"
    output += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += f"Model: {model_name}\n"
    output += "-" * 50 + "\n\n"
    
    # Group segments into logical sentences/phrases
    grouped_segments = []
    current_group = []
    current_start = None
    current_text = ""
    
    for segment in segments:
        segment_text = segment.text.strip()
        
        # If this is the start of a new group
        if current_start is None:
            current_start = segment.start
            current_text = segment_text
            current_group = [segment]
            continue
        
        # Check if we should start a new group
        should_break = False
        
        # Break if the pause between segments is longer than 3 seconds (longer pauses)
        if segment.start - current_group[-1].end > 3.0:
            should_break = True
        
        # Break if the current group would be too long (> 200 characters for longer sentences)
        elif len(current_text + " " + segment_text) > 200:
            should_break = True
        
        # Break if we have more than 8 segments in current group (allow longer sentences)
        elif len(current_group) >= 8:
            should_break = True
        
        # Only break on strong sentence endings if the group is already substantial
        elif (len(current_text) > 100 and 
              current_text.rstrip().endswith(('.', '!', '?')) and
              len(current_group) >= 3):
            should_break = True
        
        if should_break:
            # Finalize current group
            if current_group:
                grouped_segments.append({
                    'start': current_start,
                    'end': current_group[-1].end,
                    'text': current_text.strip()
                })
            
            # Start new group
            current_start = segment.start
            current_text = segment_text
            current_group = [segment]
        else:
            # Add to current group
            current_text += " " + segment_text
            current_group.append(segment)
    
    # Don't forget the last group
    if current_group:
        grouped_segments.append({
            'start': current_start,
            'end': current_group[-1].end,
            'text': current_text.strip()
        })
    
    # Format the grouped segments with timestamps
    for group in grouped_segments:
        start_timestamp = format_timestamp_simple(group['start'])
        end_timestamp = format_timestamp_simple(group['end'])
        
        # Simple timestamp and text format (no line numbers)
        output += f"{start_timestamp} --> {end_timestamp}\n"
        output += f"{group['text'].strip()}\n\n"
    
    return output

def format_timestamp_simple(seconds: float) -> str:
    """Format seconds into simple timestamp format (MM:SS)"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    
    return f"{minutes:02d}:{secs:02d}"

 