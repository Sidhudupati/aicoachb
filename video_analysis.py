import os
import cv2
import google.generativeai as genai
import speech_recognition as sr
import subprocess
from PIL import Image

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_audio_ffmpeg(video_path, audio_path, trim_sec=10):
    # Extract audio from the first N seconds using ffmpeg
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-t", str(trim_sec),
        "-acodec", "aac",
        audio_path
    ]
    subprocess.run(command, check=True)

def convert_to_wav(input_path, output_path, trim_sec=10):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-t", str(trim_sec),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        output_path
    ]
    subprocess.run(command, check=True)

def transcribe_audio(audio_path, max_duration_sec=10):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source, duration=max_duration_sec)
    try:
        transcript = recognizer.recognize_google(audio)
    except Exception as e:
        transcript = ""
        print(f"Transcription error: {e}")
    return transcript

def extract_frames(video_path, interval_sec=10, max_frames=1):
    os.makedirs("frames", exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    frames = []
    while frame_number < max_frames:
        timestamp = frame_number * interval_sec
        frame_id = int(timestamp * fps)
        if frame_id >= frame_count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = f"frames/frame_{frame_number+1}.jpg"
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 60])  # Smaller file!
        frames.append(Image.open(frame_path))
        frame_number += 1
    cap.release()
    return frames

def analyze_video_with_gemini(video_path):
    base_name, _ = os.path.splitext(video_path)
    audio_m4a = f"{base_name}_audio.m4a"
    audio_wav = f"{base_name}_audio.wav"

    # 1. Extract only the first 10 seconds of audio from video
    extract_audio_ffmpeg(video_path, audio_m4a, trim_sec=10)
    convert_to_wav(audio_m4a, audio_wav, trim_sec=10)

    # 2. Transcribe max 10 seconds audio
    transcript = transcribe_audio(audio_wav, max_duration_sec=10)

    # 3. If no meaningful transcript, return quickly
    if not transcript or len(transcript.split()) < 3:
        analysis_text = (
            "No meaningful speech detected. "
            "Please try the interview again for a proper analysis. "
            "The system did not detect spoken input in your video, so no confidence or overall score could be assigned."
        )
        return analysis_text, transcript

    # 4. Extract only 1 frame from the video, compressed
    pil_frames = extract_frames(video_path, interval_sec=10, max_frames=1)

    # 5. FAST Gemini prompt
    prompt = f"""
Audio transcript: {transcript}

Review the candidate's communication and engagement in this interview.
Give:
- 1 sentence summary
- 1 strength
- 1 area to improve
- Confidence Score: (Confidence Score: X/10)
- Overall Score: (Overall Score: X/10)

If no speech is detected, return 'Confidence Score: 0/10' and 'Overall Score: 0/10'.
"""

    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content([prompt, *pil_frames])
    return response.text, transcript
