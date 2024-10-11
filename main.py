import os
import subprocess
import threading
import re
import whisper
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File

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

def update_progress_bar(process, video_file_path):
    """Print the progress and remaining time to the terminal."""
    total_duration = get_video_duration(video_file_path)

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

                remaining_time = total_duration - current_time
                remaining_minutes, remaining_seconds = divmod(remaining_time, 60)

                print(f"Remaining Time: {int(remaining_minutes)}m {int(remaining_seconds)}s")

def convert_video_to_audio(video_file_path: str):
    if not allowed_file(video_file_path):
        print("Error: Invalid file format.")
        return

    audio_filename = f"{os.path.splitext(os.path.basename(video_file_path))[0]}.mp3"
    audio_file_path = os.path.join(OUTPUT_FOLDER, audio_filename)

    if is_nvidia_gpu_available():
        command = ['ffmpeg', '-i', video_file_path, '-c:a', 'mp3', '-c:v', 'h264_nvenc', audio_file_path]
        print("Using NVIDIA GPU for conversion.")
    else:
        command = ['ffmpeg', '-i', video_file_path, audio_file_path]
        print("Using CPU for conversion.")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    progress_thread = threading.Thread(target=update_progress_bar, args=(process, video_file_path), daemon=True)
    progress_thread.start()

    process.wait()

    if process.returncode != 0:
        print("Error converting video to audio.")
    else:
        print(f"Conversion successful! Audio file saved as: {audio_filename}")
        transcribe_audio_with_whisper(audio_file_path)

    if os.path.exists(video_file_path):
        os.remove(video_file_path)

def transcribe_audio_with_whisper(audio_file_path: str):
    """Transcribe the audio using OpenAI Whisper."""
    print("Starting transcription with Whisper...")

    device = "cuda" if is_nvidia_gpu_available() else "cpu"
    print(f"Using {'GPU' if device == 'cuda' else 'CPU'} for Whisper transcription.")

    model = whisper.load_model("base", device=device)
    
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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
