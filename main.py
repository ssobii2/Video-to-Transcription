import os
import subprocess
import threading
import re
import whisper
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Configure input, output, and transcription folders
INPUT_FOLDER = 'input/'
OUTPUT_FOLDER = 'output/'
TRANSCRIPTION_FOLDER = 'transcription/'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'mpeg', 'mpg', '3gp'}

# Create the folders if they don't exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)

# Static file serving for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global progress variables
progress_audio_conversion = 0
progress_transcription = 0
is_processing = False

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_nvidia_gpu_available() -> bool:
    """Check if an NVIDIA GPU is available using PyTorch's CUDA."""
    return torch.cuda.is_available()

def get_video_duration(video_file_path: str) -> float:
    """Get the duration of the video using ffprobe."""
    ffprobe_command = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_file_path
    ]
    duration = float(subprocess.check_output(ffprobe_command).strip())
    return duration

def update_audio_conversion_progress(process, total_duration: float):
    global progress_audio_conversion
    while True:
        output = process.stderr.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            match = re.search(r'time=(\d+:\d+:\d+\.\d+)', output)
            if match:
                time_str = match.group(1)
                h, m, s = map(float, time_str.split(':'))
                current_time = h * 3600 + m * 60 + s

                # Update global progress variable for audio conversion
                progress_audio_conversion = (current_time / total_duration) * 100
                print(f"Audio Conversion Progress: {progress_audio_conversion:.2f}%")

def convert_video_to_audio(video_file_path: str):
    global progress_audio_conversion, is_processing
    progress_audio_conversion = 0  # Reset progress at the beginning of conversion
    is_processing = True  # Set processing flag

    if not allowed_file(video_file_path):
        print("Error: Invalid file format.")
        is_processing = False
        return

    audio_filename = f"{os.path.splitext(os.path.basename(video_file_path))[0]}.mp3"
    audio_file_path = os.path.join(OUTPUT_FOLDER, audio_filename)

    command = ['ffmpeg', '-i', video_file_path, audio_file_path]
    if is_nvidia_gpu_available():
        command.insert(3, '-c:v')
        command.insert(4, 'h264_nvenc')
        print("Using NVIDIA GPU for conversion.")
    else:
        print("Using CPU for conversion.")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Get video duration for progress calculation
    total_duration = get_video_duration(video_file_path)
    progress_thread = threading.Thread(target=update_audio_conversion_progress, args=(process, total_duration), daemon=True)
    progress_thread.start()

    process.wait()

    if process.returncode != 0:
        print("Error converting video to audio.")
    else:
        print(f"Conversion successful! Audio file saved as: {audio_filename}")
        transcribe_audio_with_whisper(audio_file_path)

    if os.path.exists(video_file_path):
        os.remove(video_file_path)

    is_processing = False  # Reset processing flag
    progress_audio_conversion = 100  # Set progress to 100% after processing

def update_transcription_progress():
    global progress_transcription
    # Simulating transcription progress. You may replace this with actual tracking logic.
    for i in range(1, 101):
        progress_transcription = i
        time.sleep(0.1)  # Simulate processing time
    progress_transcription = 100  # Finalize progress

def transcribe_audio_with_whisper(audio_file_path: str):
    """Transcribe the audio using OpenAI Whisper."""
    global progress_transcription
    print("Starting transcription with Whisper...")

    device = "cuda" if is_nvidia_gpu_available() else "cpu"
    print(f"Using {'GPU' if device == 'cuda' else 'CPU'} for Whisper transcription.")

    model = whisper.load_model("base", device=device)

    # Update transcription progress in a separate thread
    transcription_thread = threading.Thread(target=update_transcription_progress)
    transcription_thread.start()

    transcription = model.transcribe(audio_file_path, verbose=True)

    transcription_filename = f"{os.path.splitext(os.path.basename(audio_file_path))[0]}.txt"
    transcription_file_path = os.path.join(TRANSCRIPTION_FOLDER, transcription_filename)

    with open(transcription_file_path, 'w') as f:
        f.write(transcription['text'])

    print(f"Transcription saved as {transcription_filename}")

    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    video_file_path = os.path.join(INPUT_FOLDER, file.filename)

    with open(video_file_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    convert_video_to_audio(video_file_path)

    return {"filename": file.filename, "status": "Processed"}

@app.get("/")
async def get_frontend():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

@app.get("/progress/")
async def get_progress():
    return {
        "progress_audio_conversion": progress_audio_conversion,
        "progress_transcription": progress_transcription,
        "is_processing": is_processing
    }  # Send processing state

@app.get("/transcription/")
async def list_transcriptions():
    """List all transcription files with download links."""
    transcriptions = os.listdir(TRANSCRIPTION_FOLDER)
    return {"transcriptions": transcriptions}

@app.get("/transcription/{filename}")
async def get_transcription(filename: str):
    transcription_file_path = os.path.join(TRANSCRIPTION_FOLDER, filename)
    if os.path.exists(transcription_file_path):
        return FileResponse(transcription_file_path, media_type='text/plain', filename=filename)  # Serve file directly
    return {"error": "Transcription not found."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
