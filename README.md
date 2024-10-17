# Video to Transcription and AI Response Generator

## Introduction

This project is a web-based application that allows users to upload video files, convert them to audio, and transcribe the audio using OpenAI's Whisper model. Additionally, the application can generate AI responses based on the transcriptions using OpenAI's API. The frontend is built using HTML, CSS, and JavaScript, while the backend is powered by Python with asynchronous support for handling tasks like file uploads, video-to-audio conversion, and transcription.

## Features

- Upload video files and convert them to audio.
- Transcribe audio using OpenAI's Whisper model.
- Generate AI responses based on transcriptions.
- View a list of transcriptions and AI responses.
- Real-time progress updates and error handling.

## Prerequisites

Before running the project locally, ensure you have the following installed:

- **Python 3.7+**: The project uses Python for the backend.
- **Node.js (optional)**: If you plan to modify or build the frontend assets.
- **FFmpeg**: Required for video-to-audio conversion.
- **OpenAI API Key**: You'll need an API key from OpenAI to use their services.

## Installation

Follow these steps to set up the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

It's a good practice to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

Make sure FFmpeg is installed and accessible from your system's PATH. You can download it from [FFmpeg's official website](https://ffmpeg.org/download.html).

To verify FFmpeg is installed correctly, run:

```bash
ffmpeg -version
```

### 5. Set Up Environment Variables

Create a `.env` file in the root directory of the project and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 6. Run the Application

Once everything is set up, you can run the application using the following command:

```bash
python main.py
```

This will start the backend server.

### 7. Access the Application

Open your browser and navigate to:

```
http://localhost:8000
```

You should see the web interface where you can upload videos, view transcriptions, and interact with AI responses.

## Usage

1. **Upload a Video**: Use the upload form to select a video file. The video will be converted to audio, and the transcription process will begin.
2. **View Transcriptions**: Once the transcription is complete, you can view the list of transcriptions.
3. **Generate AI Responses**: After transcription, you can generate AI responses based on the transcribed text.

## File Structure

Here's a brief overview of the key files and directories in the project:

- `main.py`: The main backend logic, including video-to-audio conversion, transcription, and AI response generation.
- `static/`: Contains static assets like HTML, CSS, and JavaScript files.
  - `index.html`: The main frontend interface.
  - `script.js`: Handles frontend logic, including file uploads, progress updates, and displaying transcriptions/AI responses.
  - `style.css`: Basic styling for the frontend.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Please ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
