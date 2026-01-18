from __future__ import annotations

import numpy as np
import cv2

# ------------------------------------------------------------
# SINGLE SOURCE OF TRUTH (NO FALLBACKS)
# ------------------------------------------------------------
try:
    from api.models.face_models import detect_faces, get_embedding
except Exception as e:
    raise RuntimeError(
        "❌ Face models failed to load.\n"
        "Check api/models/face_models.py\n"
        f"Original error: {e}"
    )


def get_embedding_from_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode image → detect face → crop → ArcFace embedding

    GUARANTEES:
    - Same pipeline as recognition
    - Correct face crop
    - L2-normalized 512-d vector
    - Compatible with Supabase pgvector
    """

    # --------------------------------------------------------
    # 1️⃣ Decode image
    # --------------------------------------------------------
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("❌ Failed to decode image bytes")

    img_h, img_w = img.shape[:2]

    # --------------------------------------------------------
    # 2️⃣ Detect faces (CPU-safe threshold)
    # --------------------------------------------------------
    faces = detect_faces(img, conf_thresh=0.3)

    if not faces:
        raise ValueError("❌ No face detected in image")

    # --------------------------------------------------------
    # 3️⃣ Pick BEST face (largest area)
    # --------------------------------------------------------
    best_box = None
    best_area = 0

    for (x1, y1, x2, y2) in faces:
        # Clamp coordinates (CRITICAL)
        x1 = max(0, min(int(x1), img_w - 1))
        y1 = max(0, min(int(y1), img_h - 1))
        x2 = max(0, min(int(x2), img_w))
        y2 = max(0, min(int(y2), img_h))

        w = x2 - x1
        h = y2 - y1
        area = w * h

        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        raise ValueError("❌ No valid face bounding box")

    x1, y1, x2, y2 = best_box
    face_crop = img[y1:y2, x1:x2]

    if face_crop.size == 0:
        raise ValueError("❌ Face crop is empty")

    # --------------------------------------------------------
    # 4️⃣ ArcFace embedding
    # --------------------------------------------------------
    emb = get_embedding(face_crop)

    if emb is None:
        raise ValueError("❌ ArcFace returned None")

    emb = np.asarray(emb, dtype=np.float32)

    if emb.shape != (512,):
        raise ValueError(f"❌ Invalid embedding shape: {emb.shape}")

    # --------------------------------------------------------
    # 5️⃣ FINAL NORMALIZATION (MANDATORY)
    # --------------------------------------------------------
    norm = np.linalg.norm(emb)
    if norm <= 1e-6:
        raise ValueError("❌ Zero-norm embedding")

    emb = emb / norm

    return emb
