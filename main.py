from flask import Flask, request, jsonify
from flask_cors import CORS
from video_analysis import analyze_video_with_gemini
import os, tempfile, traceback

app = Flask(__name__)
CORS(app)

@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file"}), 400

        video_file = request.files["video"]
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "interview_video.mp4")
        video_file.save(video_path)

        print(f"📹 Received video: {video_path}")

        analysis, transcript = analyze_video_with_gemini(video_path)
        return jsonify({"analysis": analysis, "transcription": transcript})

    except Exception as e:
        print("❌ Error occurred:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000)) # 10000 or any fallback value
    app.run(host="0.0.0.0", port=port, debug=True)
