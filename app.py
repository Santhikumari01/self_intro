from flask import Flask, request, jsonify, render_template
import cv2, numpy as np, time, os, tempfile, re

from mtcnn import MTCNN
from ultralytics import YOLO
from deepface import DeepFace

# AUDIO
import librosa
import ffmpeg
import whisper

app = Flask(__name__)

# ---------------- CONFIG ----------------
ABSENCE_THRESHOLD = 5
FORBIDDEN_OBJECTS = ["cell phone", "book", "backpack"]

UPLOAD_FOLDER = "uploads"
REFERENCE_IMAGE = os.path.join(UPLOAD_FOLDER, "reference.jpg")
FACE_MATCH_THRESHOLD = 60

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- MODELS ----------------
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

# ---------- UPLOAD REFERENCE PHOTO ----------
@app.route("/upload_photo", methods=["POST"])
def upload_photo():
    photo = request.files.get("photo")
    if not photo:
        return jsonify({"error": "No photo uploaded"}), 400

    photo.save(REFERENCE_IMAGE)
    return jsonify({"message": "Reference photo uploaded successfully"})

@app.route("/start")
def start():
    global proctoring, last_face_time
    global leave_count, multi_face_count, object_count

    if not os.path.exists(REFERENCE_IMAGE):
        return jsonify({"error": "Upload reference photo first"}), 400

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

# ---------------- VIDEO + FACE VERIFICATION ----------------
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
    face_match = None

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

    # -------- FACE VERIFICATION (ONLY IF 1 FACE) --------
    if face_count == 1 and os.path.exists(REFERENCE_IMAGE):
        try:
            result = DeepFace.verify(
                img1_path=REFERENCE_IMAGE,
                img2_path=frame,
                model_name="Facenet",
                enforce_detection=False
            )
            face_match = round(result.get("confidence", 0), 2)
            if face_match < FACE_MATCH_THRESHOLD:
                alerts.append("Face mismatch")
                score -= 20
        except:
            face_match = None

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
        alerts.append("Unauthorized object detected")
        score -= min(object_count * 10, 30)

    score = max(0, score)

    return jsonify({
        "face_count": face_count,
        "face_match": face_match,
        "video_alerts": alerts,
        "plagiarism": {
            "left_frame": leave_count,
            "multiple_faces": multi_face_count,
            "objects": object_count
        },
        "video_score": score
    })

# ---------------- AUDIO ANALYSIS ----------------
@app.route("/analyze_audio", methods=["POST"])
def analyze_audio():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No audio"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
        audio.save(f.name)
        webm_path = f.name

    wav_path = webm_path.replace(".webm", ".wav")

    ffmpeg.input(webm_path).output(
        wav_path, ac=1, ar=16000
    ).run(overwrite_output=True, quiet=True)

    y, sr = librosa.load(wav_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    result = whisper_model.transcribe(wav_path)
    text = result.get("text", "").strip()
    segments = result.get("segments", [])

    words = len(text.split())
    wpm = (words / duration) * 60 if duration else 0

    pauses = sum(
        1 for i in range(1, len(segments))
        if segments[i]["start"] - segments[i - 1]["end"] > 0.6
    )

    fillers = len(re.findall(r"\b(um|uh|ah|like|you know)\b", text.lower()))

    fluency_score = (
        max(0, 100 - abs(140 - wpm)) * 0.4 +
        max(0, 100 - pauses * 8) * 0.3 +
        max(0, 100 - fillers * 12) * 0.3
    )

    os.remove(webm_path)
    os.remove(wav_path)

    return jsonify({
        "transcript": text,
        "wpm": round(wpm, 1),
        "pauses": pauses,
        "fillers": fillers,
        "fluency_score": round(fluency_score, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
