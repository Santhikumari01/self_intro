from flask import Flask, request, jsonify, render_template
import cv2, numpy as np, time, os, tempfile, re

from mtcnn import MTCNN
from ultralytics import YOLO

# AUDIO
import librosa
import ffmpeg
import whisper

app = Flask(__name__)

# ---------------- CONFIG ----------------
ABSENCE_THRESHOLD = 5  # seconds
FORBIDDEN_OBJECTS = ["cell phone", "book", "backpack"]

# ---------------- MODELS (LOADED ONCE) ----------------
face_detector = MTCNN()
yolo_model = YOLO("yolov8n.pt")
whisper_model = whisper.load_model("base")

# ---------------- GLOBAL STATE ----------------
proctoring = False
last_face_time = None

leave_count = 0
multi_face_count = 0
object_count = 0

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start")
def start():
    global proctoring, last_face_time
    global leave_count, multi_face_count, object_count

    proctoring = True
    last_face_time = time.time()

    leave_count = 0
    multi_face_count = 0
    object_count = 0

    return jsonify({"status": "started"})

@app.route("/stop")
def stop():
    global proctoring
    proctoring = False
    return jsonify({"status": "stopped"})

# ---------------- VIDEO ANALYSIS (UNCHANGED LOGIC) ----------------
@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    global last_face_time, leave_count, multi_face_count, object_count

    if not proctoring:
        return jsonify({"video_score": 0})

    frame = cv2.imdecode(
        np.frombuffer(request.files["frame"].read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    alerts = []
    score = 100

    faces = face_detector.detect_faces(frame)
    face_count = len(faces)

    # Candidate left frame
    if face_count == 0:
        if time.time() - last_face_time > ABSENCE_THRESHOLD:
            leave_count += 1
            alerts.append("Candidate left frame")
            score -= min(leave_count * 10, 30)
    else:
        last_face_time = time.time()

    # Multiple faces
    if face_count > 1:
        multi_face_count += 1
        alerts.append("Multiple faces detected")
        score -= min(multi_face_count * 10, 30)

    # Object detection
    results = yolo_model(frame, conf=0.3, verbose=False)
    detected = []

    for r in results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls)]
            if label in FORBIDDEN_OBJECTS:
                detected.append(label)

    if detected:
        object_count += 1
        alerts.append(f"Unauthorized object detected: {list(set(detected))}")
        score -= min(object_count * 10, 30)

    score = max(0, score)

    return jsonify({
        "face_count": face_count,
        "video_alerts": alerts,
        "plagiarism": {
            "left_frame": leave_count,
            "multiple_faces": multi_face_count,
            "objects": object_count
        },
        "video_score": score
    })

# ---------------- AUDIO ANALYSIS (WHISPER – ACCURATE) ----------------
@app.route("/analyze_audio", methods=["POST"])
def analyze_audio():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No audio"}), 400

    # Save temporary webm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
        audio.save(f.name)
        webm_path = f.name

    wav_path = webm_path.replace(".webm", ".wav")

    # Convert to wav (16kHz mono)
    ffmpeg.input(webm_path).output(
        wav_path, ac=1, ar=16000
    ).run(overwrite_output=True, quiet=True)

    # Load audio
    y, sr = librosa.load(wav_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    # Speech-to-text
    result = whisper_model.transcribe(wav_path)
    text = result.get("text", "").strip()
    segments = result.get("segments", [])

    # No / very low speech
    if duration < 3 or len(text.split()) < 5:
        cleanup_files(webm_path, wav_path)
        return jsonify({
            "transcript": text,
            "fluency_score": 20,
            "pauses": 0,
            "fillers": 0,
            "wpm": 0
        })

    # Words per minute
    words = len(text.split())
    wpm = (words / duration) * 60

    # Pause detection
    pauses = sum(
        1 for i in range(1, len(segments))
        if segments[i]["start"] - segments[i - 1]["end"] > 0.6
    )

    # Filler detection
    fillers = len(
        re.findall(r"\b(um|uh|ah|like|you know)\b", text.lower())
    )

    # Fluency score (balanced, realistic)
    fluency_score = (
        max(0, 100 - abs(140 - wpm)) * 0.4 +
        max(0, 100 - pauses * 8) * 0.3 +
        max(0, 100 - fillers * 12) * 0.3
    )

    cleanup_files(webm_path, wav_path)

    return jsonify({
        "transcript": text,
        "wpm": round(wpm, 1),
        "pauses": pauses,
        "fillers": fillers,
        "fluency_score": round(fluency_score, 2)
    })

# ---------------- CLEANUP ----------------
def cleanup_files(*paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
