import os
import subprocess
import whisper
import torch
import time
import uvicorn
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

async def convert_video_to_audio(video_file_path: str):
    if not allowed_file(video_file_path):
        print("Error: Invalid file format.")
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
    while True:
        output = process.stderr.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            if "time=" in output:
                time_str = output.split("time=")[1].split(" ")[0]
                time_parts = list(map(float, time_str.split(':')))
                if len(time_parts) == 3:  # hh:mm:ss
                    current_time_seconds = time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
                elif len(time_parts) == 2:  # mm:ss
                    current_time_seconds = time_parts[0] * 60 + time_parts[1]
                else:  # ss
                    current_time_seconds = time_parts[0]
                
                remaining_time = total_duration - current_time_seconds
                await manager.send_message(f"Video to Audio: {format_time(remaining_time)} remaining", overwrite=True)

    if process.returncode == 0:
        print(f"Conversion successful! Audio file saved as: {audio_filename}")
        await transcribe_audio_with_whisper(audio_file_path)
    else:
        print("Error: FFmpeg process failed.")

    if os.path.exists(video_file_path):
        os.remove(video_file_path)

async def transcribe_audio_with_whisper(audio_file_path: str):
    print("Starting transcription with Whisper...")

    device = "cuda" if is_nvidia_gpu_available() else "cpu"
    if is_nvidia_gpu_available():
        print("Using NVIDIA GPU for transcription.")
    else:
        print("Using CPU for transcription.")
    
    model = whisper.load_model("turbo", device=device)

    audio_duration = get_video_duration(audio_file_path)

    if device == "cuda":
        estimated_time = audio_duration  # Real-time estimation for GPU
    else:
        estimated_time = audio_duration * 10  # 1/10th speed on CPU

    # Format estimated time to be user-friendly
    formatted_estimated_time = format_time(estimated_time)

    await manager.send_message(f"Estimated Transcription Time: {formatted_estimated_time} (NOT REAL TIME)", overwrite=True)

    # Start measuring actual transcription time
    start_time = time.time()

    transcription = model.transcribe(audio_file_path, verbose=False)

    total_segments = len(transcription["segments"])
    
    for i, segment in enumerate(transcription["segments"]):
        progress_percentage = (i + 1) / total_segments * 100
        await manager.send_message(f"Whisper: Segment {i + 1}/{total_segments} processed ({progress_percentage:.2f}%)", overwrite=True)

    transcription_filename = f"{os.path.splitext(os.path.basename(audio_file_path))[0]}.txt"
    transcription_file_path = os.path.join(TRANSCRIPTION_FOLDER, transcription_filename)

    with open(transcription_file_path, 'w', encoding='utf-8') as f:
        f.write(transcription['text'])

    actual_transcription_time = time.time() - start_time
    formatted_actual_time = format_time(actual_transcription_time)

    await manager.send_message(f"Actual Transcription Time: {formatted_actual_time}", overwrite=True)

    print(f"Transcription saved as {transcription_filename}")

    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

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
