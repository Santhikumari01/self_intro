from flask import Flask, render_template, request, jsonify
import cv2, time, os
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
REFERENCE_IMAGE = os.path.join(UPLOAD_FOLDER, "reference.jpg")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- CONFIG ----------------
FACE_MATCH_THRESHOLD = 60
ABSENCE_THRESHOLD = 5   # seconds
FORBIDDEN_OBJECTS = ["cell phone", "book", "backpack"]

# ---------------- MODELS (LOAD ONCE) ----------------
face_detector = MTCNN()
yolo_model = YOLO("yolov8n.pt")

# ---------------- GLOBAL STATE ----------------
proctoring_active = False
last_face_seen_time = None

leave_count = 0
multi_face_count = 0
object_count = 0

# ---------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_photo", methods=["POST"])
def upload_photo():
    photo = request.files.get("photo")
    if not photo:
        return jsonify({"error": "No photo"}), 400
    photo.save(REFERENCE_IMAGE)
    return jsonify({"message": "Reference image uploaded"})

@app.route("/start", methods=["GET"])
def start():
    global proctoring_active, last_face_seen_time
    global leave_count, multi_face_count, object_count

    proctoring_active = True
    last_face_seen_time = time.time()

    leave_count = 0
    multi_face_count = 0
    object_count = 0

    return jsonify({"message": "Proctoring started"})

@app.route("/stop", methods=["GET"])
def stop():
    global proctoring_active
    proctoring_active = False
    return jsonify({"message": "Proctoring stopped"})

@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    global last_face_seen_time
    global leave_count, multi_face_count, object_count

    if not proctoring_active:
        return jsonify({
            "video_score": None,
            "video_alerts": [],
            "plagiarism": {}
        })

    file = request.files.get("frame")
    if not file:
        return jsonify({"error": "No frame"}), 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    alerts = []

    # ---------------- FACE DETECTION ----------------
    faces = face_detector.detect_faces(img)
    face_count = len(faces)

    if face_count > 1:
        multi_face_count += 1
        alerts.append("Multiple faces detected")

    # ---------------- PRESENCE ----------------
    if face_count >= 1:
        last_face_seen_time = time.time()
    else:
        if time.time() - last_face_seen_time > ABSENCE_THRESHOLD:
            leave_count += 1
            last_face_seen_time = time.time()
            alerts.append("Candidate left frame")

    # ---------------- FACE VERIFICATION ----------------
    face_match = None
    if face_count == 1 and os.path.exists(REFERENCE_IMAGE):
        try:
            result = DeepFace.verify(
                REFERENCE_IMAGE,
                img,
                enforce_detection=False,
                model_name="Facenet"
            )
            face_match = round(result["confidence"], 2)
            if face_match < FACE_MATCH_THRESHOLD:
                alerts.append("Face mismatch")
        except:
            pass

    # ---------------- OBJECT DETECTION ----------------
    results = yolo_model(img, conf=0.3, verbose=False)
    detected = set()

    for r in results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls)]
            if label in FORBIDDEN_OBJECTS:
                detected.add(label)

    if detected:
        object_count += 1
        alerts.append("Unauthorized object detected")

    # ---------------- FINAL VIDEO SCORE ----------------
    video_score = 100
    video_score -= leave_count * 10
    video_score -= multi_face_count * 15
    video_score -= object_count * 20
    video_score = max(video_score, 0)

    return jsonify({
        "face_count": face_count,
        "face_match": face_match,
        "video_alerts": alerts,
        "plagiarism": {
            "left_frame": leave_count,
            "multiple_faces": multi_face_count,
            "objects": object_count
        },
        "video_score": video_score
    })

if __name__ == "__main__":
    app.run(debug=True)
