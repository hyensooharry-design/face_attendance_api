from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "ai"

ARC_PATH = MODEL_DIR / "arcface.onnx"
RETINA_PATH = MODEL_DIR / "retinaface.onnx"
DNN_PROTO = MODEL_DIR / "deploy.prototxt"
DNN_MODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

print("🔍 [face_models] Import started")
print("📁 ArcFace:", ARC_PATH)
print("📁 RetinaFace:", RETINA_PATH)
print("📁 DNN proto:", DNN_PROTO)
print("📁 DNN model:", DNN_MODEL)

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
if not ARC_PATH.exists():
    raise FileNotFoundError(f"Missing ArcFace model: {ARC_PATH}")

if not DNN_PROTO.exists() or not DNN_MODEL.exists():
    raise RuntimeError("❌ OpenCV DNN face detector files missing")

# ------------------------------------------------------------
# ONNX Runtime
# ------------------------------------------------------------
providers = ["CPUExecutionProvider"]
arc_sess = ort.InferenceSession(str(ARC_PATH), providers=providers)

try:
    retina_sess = ort.InferenceSession(str(RETINA_PATH), providers=providers)
    retina_available = True
except Exception:
    retina_sess = None
    retina_available = False

print("✅ [face_models] ArcFace loaded")
print(f"ℹ️  RetinaFace available: {retina_available}")

# ------------------------------------------------------------
# OpenCV DNN Detector
# ------------------------------------------------------------
dnn_net = cv2.dnn.readNetFromCaffe(str(DNN_PROTO), str(DNN_MODEL))
print("✅ [face_models] OpenCV DNN detector loaded")

# ------------------------------------------------------------
# SAFE FACE CROP (🔥 CRITICAL FIX)
# ------------------------------------------------------------
def safe_crop(img, box):
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


# ------------------------------------------------------------
# FACE DETECTION
# ------------------------------------------------------------
def detect_faces(frame_bgr: np.ndarray, conf_thresh: float = 0.3):
    orig_h, orig_w = frame_bgr.shape[:2]

    # -------- RetinaFace (best effort) --------
    if retina_available:
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)

            input_name = retina_sess.get_inputs()[0].name
            outputs = retina_sess.run(None, {input_name: img})

            if len(outputs) >= 2 and outputs[1].shape[-1] == 4:
                scores, boxes = outputs[:2]
                faces = []

                for i in range(scores.shape[1]):
                    if scores[0, i, 1] < conf_thresh:
                        continue

                    b = boxes[0, i]
                    x1 = int(b[0] * orig_w / 640)
                    y1 = int(b[1] * orig_h / 640)
                    x2 = int(b[2] * orig_w / 640)
                    y2 = int(b[3] * orig_h / 640)

                    if x2 > x1 and y2 > y1:
                        faces.append([x1, y1, x2, y2])

                if faces:
                    return faces
        except Exception:
            pass

    # -------- OpenCV DNN fallback --------
    resized = cv2.resize(frame_bgr, (640, 480))
    blob = cv2.dnn.blobFromImage(
        resized, 1.0, (300, 300),
        (104.0, 177.0, 123.0),
        False, False
    )

    dnn_net.setInput(blob)
    detections = dnn_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_thresh:
            continue

        box = detections[0, 0, i, 3:7]
        x1 = int(box[0] * orig_w)
        y1 = int(box[1] * orig_h)
        x2 = int(box[2] * orig_w)
        y2 = int(box[3] * orig_h)

        if x2 > x1 and y2 > y1:
            faces.append([x1, y1, x2, y2])

    return faces

# ------------------------------------------------------------
# ArcFace Embedding
# ------------------------------------------------------------
def get_embedding(face_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(rgb, (112, 112))
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    input_name = arc_sess.get_inputs()[0].name
    emb = arc_sess.run(None, {input_name: face})[0][0]
    emb = emb / np.linalg.norm(emb)

    return emb



