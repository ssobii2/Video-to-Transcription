# Video to Transcription and AI Response Generator

## Introduction

This project is a web-based application that allows users to upload video files, convert them to audio, and transcribe the audio using OpenAI's Whisper model. Additionally, the application can generate AI responses based on the transcriptions using OpenAI's API. The frontend is built using HTML, CSS, and JavaScript, while the backend is powered by Python with asynchronous support for handling tasks like file uploads, video-to-audio conversion, and transcription.

## Features

- Upload video files and convert them to audio.
- Transcribe audio using OpenAI's Whisper model.
- Generate AI responses based on transcriptions.
- View a list of transcriptions and AI responses.
- Real-time progress updates and error handling.
- Downloading the transcriptions and AI responses as text files.

## Prerequisites

Before running the project locally, ensure you have the following installed/ready:

- **Python 3.11 (Preffered)**: The project uses Python for the backend.
- **FFmpeg**: Required for video-to-audio conversion.
- **OpenAI API Key**: You'll need an API key from OpenAI to use their services (Mandatory for now).

## Installation

Follow these steps to set up the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/ssobii2/Video-to-Transcription.git
cd Video-to-Transcription
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

It's a good practice to use a virtual environment to manage dependencies.

```bash
python -m venv venv
venv\Scripts\activate  # On Linux: source venv/bin/activate
```
On Windows, if running this command in Powershell then make sure it is running in Administrator mode. If using CMD then you can run like this:

```bash
cd venv\Scripts
activate
```

### 3. Install Dependencies

Once inside the Virtual Enviornment then install the required Python packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

Make sure FFmpeg is installed and accessible from your system's PATH. You can download it from [FFmpeg's official website](https://ffmpeg.org/download.html). Check for your OS and install the appropriate version.
For Windows, the easiest way to install it is using Winget because it add's to the PATH automatically. Google it.

To verify FFmpeg is installed correctly, run:

```bash
ffmpeg -version
```

### 5. Set Up Environment Variables

Create a `.env` file in the root directory of the project and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

You can get you API key from [OpenAI's website](https://platform.openai.com/api-keys). You need to create an account and then create a new API key.

### 6. Run the Application

Once everything is set up, you can run the application using the following command:

```bash
uvicorn main:app
```

This will start the backend server which starts the frontend as well because both are handled by FastAPI.

### 7. Access the Application

Open your browser and navigate to:

```
http://localhost:8000
```

You should see the web interface.

## Usage

1. **Upload a Video**: Use the upload button to select a video file. The video will be converted to audio, and the transcription process will begin.
2. **View Transcriptions**: Once the transcription is complete, you can view the list of transcriptions.
3. **Generate AI Responses**: At the same time, whatever prompt you put there, the AI will generate a response based on the transcription. Both files will be saved separately.
4. **Download Transcriptions**: You can download the transcriptions and AI responses as text files.
