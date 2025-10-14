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

        result = analyze_video_with_gemini(video_path)
        return jsonify({"analysis": result})
    except Exception as e:
        print("❌ Error occurred:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5050, debug=True)
