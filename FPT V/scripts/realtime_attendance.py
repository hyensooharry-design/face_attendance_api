"""Terminal-based real-time attendance (OpenCV).

This script:
- Loads MTCNN + FaceNet + FAISS index
- Runs webcam loop
- Logs IN/OUT toggled events to data/attendance.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
import time

import cv2
import numpy as np
import torch

# Ensure project root imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.face_models import load_mtcnn, load_facenet
from utils.faiss_utils import load_faiss_assets, search_face
from utils.attendance_utils import ensure_attendance_csv, log_attendance


def get_embedding(mtcnn, facenet, frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)
    if face is None:
        return None
    face = face.unsqueeze(0)
    with torch.no_grad():
        emb = facenet(face).cpu().numpy().reshape(-1)
    return emb


def main():
    ensure_attendance_csv()

    mtcnn = load_mtcnn(device="cpu")
    facenet = load_facenet(device="cpu")
    facenet.eval()

    index, names = load_faiss_assets()

    threshold = 0.8
    cam_index = 0

    attendance_cache = {}

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    print("▶️ Press 'q' to quit.")
    last_log_ts = 0.0
    cooldown_sec = 1.5  # avoid spamming logs every frame

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        emb = get_embedding(mtcnn, facenet, frame)
        name = None
        dist = None
        if emb is not None:
            name, dist = search_face(index, names, emb, threshold=threshold)

        label = "No face"
        if emb is not None and name is None:
            label = f"Unknown (dist={dist:.3f})"
        if name is not None:
            label = f"{name} (dist={dist:.3f})"

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Real-time Attendance", frame)

        now = time.time()
        if name is not None and (now - last_log_ts) >= cooldown_sec:
            last_status = attendance_cache.get(name, "OUT")
            new_status = "IN" if last_status == "OUT" else "OUT"
            attendance_cache[name] = new_status
            log_attendance(name, new_status, confidence=float(1.0 - min(dist, 1.0)))
            print(f"✅ Logged: {name} - {new_status}")
            last_log_ts = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
