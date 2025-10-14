# video_analysis.py

import os
import cv2
import moviepy.editor as mp
import google.generativeai as genai

# Remove this line! It's not needed here:
# app = Flask(__name__)
# CORS(app)  # also remove

def analyze_video_with_gemini(video_path):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Extract audio
    base_name, _ = os.path.splitext(video_path)
    audio_path = f"{base_name}_audio.m4a"
    video_clip = mp.VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path, codec='aac')

    # Extract frames every 5 seconds
    os.makedirs("frames", exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = 5
    frame_paths = []
    frame_number = 0
    while True:
        timestamp = frame_number * interval
        frame_id = int(timestamp * fps)
        if frame_id >= frame_count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = f"frames/frame_{frame_number+1}.jpg"
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_number += 1
    cap.release()

    # Upload files to Gemini
    uploaded_audio = genai.upload_file(path=audio_path)
    uploaded_frames = [genai.upload_file(path=path) for path in frame_paths]

    # Multimodal analysis
    prompt = """Analyze this interview video using both audio and visual information.
Audio: transcription, tone, pace, confidence
Visual: facial expressions, posture, eye contact, engagement
Output: summary, strengths, improvements, score 1-10"""

    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content([prompt, uploaded_audio, *uploaded_frames])
    return response.text
