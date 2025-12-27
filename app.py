from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from ultralytics import YOLO
import time, os

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
REFERENCE_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "temp_reference.jpg")

FACE_MATCH_THRESHOLD = 60       # %
ABSENCE_THRESHOLD = 5           # seconds

# Forbidden objects (expanded slightly)
FORBIDDEN_OBJECTS = ["cell phone", "book", "backpack"]

# ---------------------------------------

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Models
face_detector = MTCNN()
yolo_model = YOLO("yolov8n.pt")   # auto-download

# Proctoring state
proctoring_active = False
last_face_seen_time = None

# Object memory for multi-frame confirmation
object_memory = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_photo", methods=["POST"])
def upload_photo():
    photo = request.files.get("photo")
    if not photo:
        return jsonify({"message": "No photo uploaded"}), 400

    photo.save(REFERENCE_IMAGE_PATH)
    return jsonify({"message": "Reference photo uploaded successfully"})

@app.route("/start_proctoring", methods=["POST"])
def start_proctoring():
    global proctoring_active, last_face_seen_time, object_memory
    proctoring_active = True
    last_face_seen_time = time.time()
    object_memory = {}
    return jsonify({"message": "Proctoring started"})

@app.route("/stop_proctoring", methods=["POST"])
def stop_proctoring():
    global proctoring_active
    proctoring_active = False
    return jsonify({"message": "Proctoring stopped"})

@app.route("/analyze", methods=["POST"])
def analyze():
    global last_face_seen_time, object_memory

    # 🚫 Do nothing if proctoring not started
    if not proctoring_active:
        return jsonify({
            "face_count": None,
            "face_match_score": None,
            "detected_objects": [],
            "video_alerts": []
        })

    frame_file = request.files.get("frame")
    if not frame_file:
        return jsonify({"error": "No frame received"}), 400

    np_img = np.frombuffer(frame_file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    alerts = []
    detected_objects = []
    match_score = None

    # ---------- 1️⃣ FACE DETECTION ----------
    faces = face_detector.detect_faces(frame)
    face_count = len(faces)

    if face_count > 1:
        alerts.append("Multiple persons detected")

# Select largest face as candidate
    if faces:
        faces = sorted(faces, key=lambda x: x['box'][2]*x['box'][3], reverse=True)


    # ---------- 4️⃣ PRESENCE MONITORING ----------
    if face_count >= 1:
        last_face_seen_time = time.time()
    elif last_face_seen_time is not None:
        if time.time() - last_face_seen_time > ABSENCE_THRESHOLD:
            alerts.append("Candidate left frame")

    # ---------- 2️⃣ FACE VERIFICATION ----------
    if face_count == 1 and os.path.exists(REFERENCE_IMAGE_PATH):
        try:
            result = DeepFace.verify(
                img1_path=REFERENCE_IMAGE_PATH,
                img2_path=frame,
                enforce_detection=False,
                model_name="Facenet"
            )
            match_score = round(result["confidence"], 2)
            if match_score < FACE_MATCH_THRESHOLD:
                alerts.append("Face mismatch")
        except:
            pass

    # ---------- 3️⃣ OBJECT DETECTION (YOLOv8n – FIXED) ----------
    results = yolo_model(frame, conf=0.25, verbose=False)

    current_detected = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            label = yolo_model.names[cls_id]

            if label in FORBIDDEN_OBJECTS:
                current_detected.append(label)

    # Multi-frame confirmation (2 frames)
    for obj in current_detected:
        object_memory[obj] = object_memory.get(obj, 0) + 1

    # Reset memory for objects not seen
    for key in list(object_memory.keys()):
        if key not in current_detected:
            object_memory[key] = 0

    confirmed_objects = [k for k, v in object_memory.items() if v >= 2]

    if confirmed_objects:
        detected_objects.extend(confirmed_objects)
        if "Unauthorized object detected" not in alerts:
            alerts.append("Unauthorized object detected")


    return jsonify({
        "face_count": face_count,
        "face_match_score": match_score,
        "detected_objects": list(set(detected_objects)),
        "video_alerts": list(set(alerts))
    })

if __name__ == "__main__":
    app.run(debug=True)
