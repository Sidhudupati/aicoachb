import os
import cv2
import google.generativeai as genai
import speech_recognition as sr
import subprocess
from PIL import Image

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_audio_ffmpeg(video_path, audio_path):
    # Extract audio from video using ffmpeg (.m4a format, AAC codec)
    command = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",  # no video
        "-acodec", "aac",
        audio_path
    ]
    subprocess.run(command, check=True)

def convert_to_wav(input_path, output_path):
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        output_path
    ]
    subprocess.run(command, check=True)

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio)
    except Exception as e:
        transcript = ""
        print(f"Transcription error: {e}")
    return transcript

def extract_frames(video_path, interval_sec=5, max_frames=5):
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
        cv2.imwrite(frame_path, frame)
        frames.append(Image.open(frame_path))  # PIL Image
        frame_number += 1
    cap.release()
    return frames

def analyze_video_with_gemini(video_path):
    base_name, _ = os.path.splitext(video_path)
    audio_m4a = f"{base_name}_audio.m4a"
    audio_wav = f"{base_name}_audio.wav"

    # 1. Extract audio from video using ffmpeg
    extract_audio_ffmpeg(video_path, audio_m4a)

    # 2. Convert M4A → WAV (PCM)
    convert_to_wav(audio_m4a, audio_wav)

    # 3. Transcribe WAV audio
    transcript = transcribe_audio(audio_wav)

    # If no speech or transcript is very short, return custom response
    if not transcript or len(transcript.split()) < 3:
        analysis_text = (
            "No meaningful speech detected. "
            "Please try the interview again for a proper analysis. "
            "The system did not detect spoken input in your video, so no confidence or overall score could be assigned."
        )
        return analysis_text, transcript

    # 4. Extract frames as PIL Images
    pil_frames = extract_frames(video_path)

    # 5. Prepare prompt
    prompt = f"""
Analyze this interview video using both audio and visual information.

Audio (transcription): {transcript}
Visual: facial expressions, posture, eye contact, engagement.
Output:
- Summary
- Strengths
- Improvements
- Confidence Score: (Write only as: Confidence Score: X/10)
- Overall Score: (Write only as: Overall Score: X/10)

If no speech is detected, return 'Confidence Score: 0/10' and 'Overall Score: 0/10'.
"""
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content([prompt, *pil_frames])
    return response.text, transcript
