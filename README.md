# Google & Zoom Meeting Notes Generator 🎙️📝

An automated system that records Google Meet and Zoom meetings and generates clean, timestamped meeting notes using OpenAI's Whisper speech-to-text model.

## Features
- Auto-detects Zoom / Google Meet sessions
- Records system audio
- Transcribes audio using Whisper
- Generates readable meeting notes
- Fully local & privacy-friendly

## Tech Stack
- Python
- OpenAI Whisper
- FFmpeg
- PyAudio
- SpeechRecognition

## How it Works
1. Join a Zoom or Google Meet call
2. Run the script
3. Audio is recorded automatically
4. Whisper converts speech → text
5. Notes are saved as `.txt` files

## Setup
```bash
pip install -r requirements.txt
python main.py
