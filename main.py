import os
import subprocess
import whisper
import torch
import time
import uvicorn
import asyncio
import threading
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, Callable

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure paths
INPUT_FOLDER = 'input/'
OUTPUT_FOLDER = 'output/'
TRANSCRIPTION_FOLDER = 'transcription/'
AI_RESPONSES_FOLDER = 'ai_responses/'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'mpeg', 'mpg', '3gp', 'mp3', 'wav', 'ogg', 'm4a', 'aac'}

# Ensure folders exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)
os.makedirs(AI_RESPONSES_FOLDER, exist_ok=True)

# Static file serving setup
app.mount("/static", StaticFiles(directory="static"), name="static")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, overwrite: bool = False):
        for connection in self.active_connections:
            if overwrite:
                await connection.send_text(f"\r{message}")
            else:
                await connection.send_text(message)

manager = ConnectionManager()

def allowed_file(filename: str) -> bool:
    """Enhanced file validation"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
        
    return True

def is_nvidia_gpu_available() -> bool:
    """Check if an NVIDIA GPU is available using PyTorch's CUDA."""
    return torch.cuda.is_available()

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
         print(f"Error running ffprobe: {e}")
         raise
     except ValueError as e:
         print(f"Error parsing duration: {e}")
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

async def delete_folder_contents(folder: str):
    """Delete all contents of a folder."""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            os.rmdir(file_path)

async def convert_video_to_audio(file_path: str, prompt: str):
    if not allowed_file(file_path):
        print("Error: Invalid file format.")
        await delete_folder_contents(INPUT_FOLDER)
        return

    file_extension = os.path.splitext(file_path)[1].lower()
    audio_extensions = {'.mp3', '.wav', '.ogg', '.m4a', '.aac'}
    
    if file_extension in audio_extensions:
        # If it's already an audio file, copy it to output folder
        audio_filename = os.path.basename(file_path)
        audio_file_path = os.path.join(OUTPUT_FOLDER, audio_filename)
        try:
            import shutil
            shutil.copy2(file_path, audio_file_path)
            print("Audio file copied successfully")
            await manager.send_message(
                "Audio file ready for transcription!",
                overwrite=True
            )
            await asyncio.sleep(1)
            await transcribe_audio_with_whisper(audio_file_path, prompt)
            return
        except Exception as e:
            print(f"Error copying audio file: {e}")
            raise

    # If it's a video file, proceed with conversion
    audio_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}.mp3"
    audio_file_path = os.path.join(OUTPUT_FOLDER, audio_filename)

    command = [
        'ffmpeg',
        '-i', file_path,
        '-vn',
        '-acodec', 'libmp3lame',
        '-q:a', '2',
        '-y',
        '-progress', 'pipe:1',
        '-nostats',
        audio_file_path
    ]

    if is_nvidia_gpu_available():
        command.insert(1, '-hwaccel')
        command.insert(2, 'cuda')
        print("Using NVIDIA GPU for conversion.")
    else:
        print("Using CPU for conversion.")

    try:
        total_duration = get_video_duration(file_path)
        if total_duration <= 0:
            raise ValueError("Invalid video duration")
            
        start_time = time.time()
        last_progress_time = time.time()
        last_progress = 0
        no_progress_count = 0
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        while True:
            if process.poll() is not None:
                break

            line = process.stdout.readline()
            if not line:
                continue
            
            if 'out_time_ms=' in line:
                try:
                    current_time = float(line.strip().split('=')[1]) / 1000000
                    if current_time > last_progress:
                        last_progress = current_time
                        last_progress_time = time.time()
                        no_progress_count = 0
                        
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 0:
                            speed = current_time / elapsed_time
                            remaining_time = (total_duration - current_time) / speed
                            
                            if remaining_time > 1:
                                await manager.send_message(
                                    f"Video to Audio: {format_time(max(0, remaining_time))} remaining",
                                    overwrite=True
                                )
                            else:
                                break
                    else:
                        no_progress_count += 1
                        if no_progress_count > 10:
                            break
                            
                except (ValueError, ZeroDivisionError) as e:
                    print(f"Error parsing progress: {e}")
                    continue

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            print("Had to terminate FFmpeg process")

        if os.path.exists(audio_file_path) and os.path.getsize(audio_file_path) > 0:
            print(f"Conversion completed in {format_time(time.time() - start_time)}")
            await manager.send_message(
                "Video to Audio: Conversion complete!",
                overwrite=True
            )
            await asyncio.sleep(1)
            await transcribe_audio_with_whisper(audio_file_path, prompt)
        else:
            raise Exception("Audio file not created or empty")

    except Exception as e:
        error_msg = f"Error during conversion: {str(e)}"
        print(error_msg)
        await manager.send_message(f"Error: {error_msg}", overwrite=True)
        await delete_folder_contents(INPUT_FOLDER)
        await delete_folder_contents(OUTPUT_FOLDER)

    finally:
        if process.poll() is None:
            process.terminate()
        if os.path.exists(file_path):
            os.remove(file_path)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = "cuda"
else:
    torch.set_num_threads(2)
    device = "cpu"

model = whisper.load_model("base", device=device)

print(f"Using device: {device}")

client = AsyncOpenAI()

class WhisperTranscriber:
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        """Initialize the WhisperTranscriber with specified model and device."""
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = whisper.load_model(model_name, device=self.device)
        self.window_size = 30
        self.current_window = 0
        self.total_windows = 0
        self.start_time = 0
        self.is_transcribing = False

    def _calculate_remaining_time(self) -> float:
        """Calculate remaining time based on actual processing speed of windows."""
        if not self.is_transcribing or self.current_window == 0:
            return 0
            
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0
            
        windows_per_second = self.current_window / elapsed_time
        remaining_windows = self.total_windows - self.current_window
        
        return remaining_windows / windows_per_second if windows_per_second > 0 else 0

    async def transcribe(self, audio_path: str, progress_callback) -> Dict:
        """
        Transcribe audio file with accurate progress tracking.
        Returns dictionary with transcription text or error message.
        """
        try:
            self.start_time = time.time()
            self.is_transcribing = True
            self.current_window = 0
            
            audio = whisper.load_audio(audio_path)
            audio_duration = len(audio) / whisper.audio.SAMPLE_RATE
            self.total_windows = int(audio_duration / self.window_size) + 1
            
            transcription_segments = []
            
            for i in range(self.total_windows):
                start_sample = i * self.window_size * whisper.audio.SAMPLE_RATE
                end_sample = min((i + 1) * self.window_size * whisper.audio.SAMPLE_RATE, len(audio))
                audio_chunk = audio[start_sample:end_sample]
                
                if len(audio_chunk) < whisper.audio.SAMPLE_RATE:
                    break
                    
                mel = whisper.log_mel_spectrogram(audio_chunk).to(self.device)
                result = self.model.transcribe(audio_chunk)
                transcription_segments.append(result["text"])
                
                self.current_window = i + 1
                
                remaining_time = self._calculate_remaining_time()
                if progress_callback:
                    await progress_callback(remaining_time)
            
            total_time = time.time() - self.start_time
            print(f"\nTranscription completed in {format_time(total_time)}")
            print(f"Average processing speed: {audio_duration/total_time:.2f}x real-time")
            await manager.send_message(f"\nTranscription completed in {format_time(total_time)}", overwrite=True)
            
            return {
                "text": " ".join(transcription_segments),
                "segments": transcription_segments,
                "processing_time": total_time,
                "audio_duration": audio_duration,
                "processing_speed": audio_duration/total_time
            }
            
        except Exception as e:
            return {"error": str(e)}
        finally:
            self.is_transcribing = False

async def transcribe_audio_with_whisper(audio_file_path: str, prompt: str):
    print("Starting transcription with Whisper...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transcriber = WhisperTranscriber(model_name="base", device=device)
    
    async def progress_callback(remaining_time: float):
        formatted_time = format_time(max(remaining_time, 0))
        await manager.send_message(
            f"Whisper Transcription: {formatted_time} remaining",
            overwrite=True
        )
    
    try:
        result = await transcriber.transcribe(audio_file_path, progress_callback)
        
        if "error" in result:
            print(f"Transcription error: {result['error']}")
            await manager.send_message("Transcription failed", overwrite=True)
            return
            
        transcription_text = result["text"]
        if not transcription_text:
            print("Transcription failed or returned no text.")
            await manager.send_message("Transcription failed - no text generated", overwrite=True)
            return
            
        transcription_filename = f"{os.path.splitext(os.path.basename(audio_file_path))[0]}.txt"
        transcription_file_path = os.path.join(TRANSCRIPTION_FOLDER, transcription_filename)
        
        with open(transcription_file_path, 'w', encoding='utf-8') as f:
            f.write(transcription_text)
            
        print(f"Transcription saved as {transcription_filename}")
        
        await asyncio.sleep(2)
        
        await manager.send_message(
            "Generating AI response...",
            overwrite=True
        )
        
        ai_response = await use_openai_async(transcription_text, prompt)
        
        if ai_response:
            ai_filename = f"{os.path.splitext(os.path.basename(audio_file_path))[0]}_ai_response.txt"
            ai_file_path = os.path.join(AI_RESPONSES_FOLDER, ai_filename)
            
            with open(ai_file_path, 'w', encoding='utf-8') as f:
                f.write(ai_response)
                
            await manager.send_message(
                "Processing complete! You can now download your files.",
                overwrite=True
            )
            
            print(f"AI response saved as {ai_filename}")
            
    except Exception as e:
        print(f"Error during transcription: {e}")
        await manager.send_message("An error occurred during processing", overwrite=True)
        await delete_folder_contents(INPUT_FOLDER)
        await delete_folder_contents(OUTPUT_FOLDER)
        await delete_folder_contents(TRANSCRIPTION_FOLDER)
        await delete_folder_contents(AI_RESPONSES_FOLDER)
        
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

async def use_openai_async(transcription_text: str, custom_prompt: str) -> str:
    """Use OpenAI's GPT in an asynchronous manner to process the transcription."""
    if not custom_prompt:
        custom_prompt = "Summarize in Points"
    
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": custom_prompt},
                {"role": "user", "content": transcription_text},
            ],
            model="gpt-4o-mini",
        )

        if chat_completion and chat_completion.choices:
            return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"Error occurred while processing with OpenAI: {e}")
        await delete_folder_contents(INPUT_FOLDER)
        await delete_folder_contents(OUTPUT_FOLDER)
        await delete_folder_contents(TRANSCRIPTION_FOLDER)
        await delete_folder_contents(AI_RESPONSES_FOLDER)
    
    return None
    
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), prompt: str = Form(...)):
    file_path = os.path.join(INPUT_FOLDER, file.filename)

    if not allowed_file(file.filename):
        return {"error": "Invalid file format. Please upload a valid video or audio file."}

    transcription_filename = f"{os.path.splitext(file.filename)[0]}.txt"
    ai_response_filename = f"{os.path.splitext(file.filename)[0]}_ai_response.txt"
    transcription_file_path = os.path.join(TRANSCRIPTION_FOLDER, transcription_filename)
    ai_file_path = os.path.join(AI_RESPONSES_FOLDER, ai_response_filename)

    if os.path.exists(transcription_file_path):
        return {"error": "Transcription already exists. Please download it."}
    
    if os.path.exists(ai_file_path):
        return {"error": "Transcription already exists. Please download it."}

    try:
        with open(file_path, 'wb') as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        await convert_video_to_audio(file_path, prompt)
        return {"filename": file.filename, "status": "Processed"}

    except Exception as e:
        print(f"Error during file upload: {e}")
        await delete_folder_contents(INPUT_FOLDER)
        await delete_folder_contents(OUTPUT_FOLDER)
        await delete_folder_contents(TRANSCRIPTION_FOLDER)
        await delete_folder_contents(AI_RESPONSES_FOLDER)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
async def get_frontend():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

@app.get("/transcription/")
async def list_transcriptions():
    transcriptions = os.listdir(TRANSCRIPTION_FOLDER)
    return {"transcriptions": transcriptions}

@app.get("/transcription/{filename}")
async def get_transcription(filename: str):
    transcription_file_path = os.path.join(TRANSCRIPTION_FOLDER, filename)
    if os.path.exists(transcription_file_path):
        return FileResponse(transcription_file_path, media_type='text/plain', filename=filename)
    raise HTTPException(status_code=404, detail="Transcription not found.")

@app.delete("/transcription/{filename}")
async def delete_transcription(filename: str):
    transcription_file_path = os.path.join(TRANSCRIPTION_FOLDER, filename)
    if os.path.exists(transcription_file_path):
        os.remove(transcription_file_path)
        return {"message": f"{filename} deleted"}
    raise HTTPException(status_code=404, detail="Transcription not found.")

@app.get('/ai/')
async def list_ai_responses():
    ai_responses = os.listdir(AI_RESPONSES_FOLDER)
    return {'ai_responses': ai_responses}

@app.get('/ai/{filename}')
async def get_ai_response(filename):
    ai_response_file_path = os.path.join(AI_RESPONSES_FOLDER, filename)
    if os.path.exists(ai_response_file_path):
        return FileResponse(ai_response_file_path, media_type='text/plain', filename=filename)
    raise HTTPException(status_code=404, detail="AI Response not found.")

@app.delete('/ai/{filename}')
async def delete_ai_response(filename):
    ai_response_file_path = os.path.join(AI_RESPONSES_FOLDER, filename)
    if os.path.exists(ai_response_file_path):
        os.remove(ai_response_file_path)
        return {"message": f"{filename} deleted"}
    raise HTTPException(status_code=404, detail="AI Response not found.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
