import os
import subprocess
import threading
import re
import whisper
import torch
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'mpeg', 'mpg', '3gp'}

# Ensure folders exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)

# Static file serving setup
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global progress variables
progress_audio_conversion = 0
progress_transcription = 0
is_processing = False

# WebSocket connections
active_connections = []

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

async def send_progress_updates():
    """Send progress updates to all active WebSocket connections."""
    while is_processing:
        for connection in active_connections:
            await connection.send_json({
                "progress_audio_conversion": progress_audio_conversion,
                "progress_transcription": progress_transcription,
                "is_processing": is_processing
            })
        await asyncio.sleep(1)  # Send updates every second

def update_audio_conversion_progress(process, total_duration: float):
    global progress_audio_conversion
    while True:
        output = process.stderr.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            match = re.search(r'time=(\d+:\d+:\d+\.\d+)', output)
            if match:
                time_str = match.group(1)
                h, m, s = map(float, time_str.split(':'))
                current_time = h * 3600 + m * 60 + s

                # Update progress information
                progress_audio_conversion = (current_time / total_duration) * 100
                
                # Log progress for debugging
                print(f"Audio conversion progress: {progress_audio_conversion:.2f}%")

def end_conversion_on_completion(process):
    """Waits for the FFmpeg process to complete."""
    global progress_audio_conversion
    process.wait()
    if process.returncode == 0:
        progress_audio_conversion = 100
    else:
        print("Error: FFmpeg process failed.")
    return process.returncode

async def convert_video_to_audio(video_file_path: str):
    global progress_audio_conversion, is_processing
    is_processing = True
    progress_audio_conversion = 0

    if not allowed_file(video_file_path):
        print("Error: Invalid file format.")
        is_processing = False
        return

    audio_filename = f"{os.path.splitext(os.path.basename(video_file_path))[0]}.mp3"
    audio_file_path = os.path.join(OUTPUT_FOLDER, audio_filename)

    command = [
        'ffmpeg', '-i', video_file_path, '-q:a', '0', '-map', 'a',
        audio_file_path, '-hide_banner', '-loglevel', 'info'
    ]

    if is_nvidia_gpu_available():
        command.insert(3, '-c:v')
        command.insert(4, 'h264_nvenc')
        print("Using NVIDIA GPU for conversion.")
    else:
        print("Using CPU for conversion.")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    total_duration = get_video_duration(video_file_path)
    progress_thread = threading.Thread(
        target=update_audio_conversion_progress,
        args=(process, total_duration),
        daemon=True
    )
    progress_thread.start()

    await asyncio.get_event_loop().run_in_executor(None, end_conversion_on_completion, process)

    if process.returncode == 0:
        print(f"Conversion successful! Audio file saved as: {audio_filename}")
        await transcribe_audio_with_whisper(audio_file_path)

    if os.path.exists(video_file_path):
        os.remove(video_file_path)

    is_processing = False
    await send_progress_updates()

async def transcribe_audio_with_whisper(audio_file_path: str):
    global progress_transcription, is_processing
    print("Starting transcription with Whisper...")

    device = "cuda" if is_nvidia_gpu_available() else "cpu"
    model = whisper.load_model("base", device=device)

    audio_duration = get_video_duration(audio_file_path)
    transcription = model.transcribe(audio_file_path)

    transcription_filename = f"{os.path.splitext(os.path.basename(audio_file_path))[0]}.txt"
    transcription_file_path = os.path.join(TRANSCRIPTION_FOLDER, transcription_filename)

    with open(transcription_file_path, 'w') as f:
        f.write(transcription['text'])

    progress_transcription = 100
    print(f"Transcription saved as {transcription_filename}")

    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

    is_processing = False
    await send_progress_updates()
    print("Transcription complete and processed")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    video_file_path = os.path.join(INPUT_FOLDER, file.filename)

    transcription_filename = f"{os.path.splitext(file.filename)[0]}.txt"
    transcription_file_path = os.path.join(TRANSCRIPTION_FOLDER, transcription_filename)

    if os.path.exists(transcription_file_path):
        return {"error": "Transcription already exists. Please download it."}

    try:
        with open(video_file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        await convert_video_to_audio(video_file_path)
        return {"filename": file.filename, "status": "Processed"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def get_frontend():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws/progress/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while is_processing:
            await asyncio.sleep(1)  # Keep the connection alive
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print("WebSocket connection closed")
    finally:
        await websocket.close()

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
