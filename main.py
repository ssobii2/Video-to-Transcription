import os
import subprocess
import whisper
import torch
import time
import uvicorn
import asyncio
import threading
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
            os.rmdir(file_path)  # For empty directories

async def convert_video_to_audio(video_file_path: str):
    if not allowed_file(video_file_path):
        print("Error: Invalid file format.")
        await delete_folder_contents(INPUT_FOLDER)
        await delete_folder_contents(OUTPUT_FOLDER)
        await delete_folder_contents(TRANSCRIPTION_FOLDER)
        return

    audio_filename = f"{os.path.splitext(os.path.basename(video_file_path))[0]}.mp3"
    audio_file_path = os.path.join(OUTPUT_FOLDER, audio_filename)

    command = [
        'ffmpeg', '-i', video_file_path, '-vn', '-acodec', 'libmp3lame', '-q:a', '2',
        audio_file_path, '-hide_banner', '-loglevel', 'info'
    ]

    if is_nvidia_gpu_available():
        command.insert(1, '-hwaccel')
        command.insert(2, 'cuda')
        print("Using NVIDIA GPU for conversion.")
    else:
        print("Using CPU for conversion.")

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        total_duration = get_video_duration(video_file_path)

        while True:
            output = process.stderr.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Parse the progress from FFmpeg output
                if "time=" in output:
                    time_str = output.split("time=")[1].split(" ")[0]
                    time_parts = list(map(float, time_str.split(':')))
                    current_time_seconds = time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]

                    remaining_time = total_duration - current_time_seconds
                    formatted_remaining_time = format_time(max(remaining_time, 0))
                    await manager.send_message(
                        f"Video to Audio: {formatted_remaining_time} remaining",
                        overwrite=True
                    )

        if process.returncode == 0:
            print(f"Conversion successful! Audio file saved as: {audio_filename}")
            await transcribe_audio_with_whisper(audio_file_path)
        else:
            print("Error: FFmpeg process failed.")
            await delete_folder_contents(INPUT_FOLDER)
            await delete_folder_contents(OUTPUT_FOLDER)
            await delete_folder_contents(TRANSCRIPTION_FOLDER)

    except Exception as e:
        print(f"Error during video conversion: {e}")
        await delete_folder_contents(INPUT_FOLDER)
        await delete_folder_contents(OUTPUT_FOLDER)
        await delete_folder_contents(TRANSCRIPTION_FOLDER)

    if os.path.exists(video_file_path):
        os.remove(video_file_path)

device = "cuda" if is_nvidia_gpu_available() else "cpu"
if is_nvidia_gpu_available():
    print("Using NVIDIA GPU for transcription.")
else:
    print("Using CPU for transcription.")

model = whisper.load_model("base", device=device)

async def transcribe_audio_with_whisper(audio_file_path: str):
    print("Starting transcription with Whisper...")

    audio_duration = get_video_duration(audio_file_path)

    # Estimate total transcription time based on device and audio duration
    if device == "cuda":
        estimated_total_time = audio_duration * 0.2
    else:
        estimated_total_time = audio_duration * 3

    start_time = time.time()

    transcription = {}

    try:
        # Define a function to run transcription in a separate thread
        def run_transcription():
            transcription_result = model.transcribe(audio_file_path, verbose=False)
            transcription.update(transcription_result)

        transcription_thread = threading.Thread(target=run_transcription)
        transcription_thread.start()

        while transcription_thread.is_alive():
            elapsed_time = time.time() - start_time
            estimated_remaining_time = estimated_total_time - elapsed_time
            formatted_remaining_time = format_time(max(estimated_remaining_time, 0))
            await manager.send_message(
                f"Whisper Transcription: Approximately {formatted_remaining_time} remaining (NOT REAL TIME)",
                overwrite=True
            )
            await asyncio.sleep(2)

        transcription_thread.join()

        transcription_filename = f"{os.path.splitext(os.path.basename(audio_file_path))[0]}.txt"
        transcription_file_path = os.path.join(TRANSCRIPTION_FOLDER, transcription_filename)

        with open(transcription_file_path, 'w', encoding='utf-8') as f:
            f.write(transcription['text'])

        total_transcription_time = time.time() - start_time
        formatted_total_time = format_time(total_transcription_time)

        await manager.send_message(
            f"Transcription completed in {formatted_total_time}",
            overwrite=True
        )

        print(f"Transcription saved as {transcription_filename}")

    except Exception as e:
        print(f"Error during transcription: {e}")
        await delete_folder_contents(INPUT_FOLDER)
        await delete_folder_contents(OUTPUT_FOLDER)
        await delete_folder_contents(TRANSCRIPTION_FOLDER)

    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    video_file_path = os.path.join(INPUT_FOLDER, file.filename)

    if not allowed_file(file.filename):
        return {"error": "Invalid file format. Please upload a valid video file."}

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
        print(f"Error during file upload: {e}")
        await delete_folder_contents(INPUT_FOLDER)
        await delete_folder_contents(OUTPUT_FOLDER)
        await delete_folder_contents(TRANSCRIPTION_FOLDER)
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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
