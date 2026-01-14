from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import cv2

# -----------------------------------------------------------------------------
# Optional model imports (graceful fallback when models/ is missing)
# -----------------------------------------------------------------------------
_MODEL_AVAILABLE = True
try:
    from api.models.face_models import load_retinaface, load_arcface  # type: ignore
except Exception:
    _MODEL_AVAILABLE = False
    load_retinaface = None  # type: ignore
    load_arcface = None  # type: ignore

# If models are missing, enable dummy mode by default (can be overridden)
# - DUMMY_MODE=1 : always dummy (even if models exist)
# - DUMMY_MODE=0 : force real (will crash if models missing)
_env_dummy = os.getenv("DUMMY_MODE", "").strip()
if _env_dummy == "":
    # default behavior: dummy if models missing
    DUMMY_MODE = not _MODEL_AVAILABLE
else:
    DUMMY_MODE = _env_dummy == "1"


# Lazy-loaded models
_RETINA = None
_ARCFACE = None


def _ensure_models() -> None:
    """
    Load models lazily. If dummy mode, skip.
    """
    global _RETINA, _ARCFACE

    if DUMMY_MODE:
        return

    if not _MODEL_AVAILABLE:
        raise RuntimeError(
            "Model modules not available (models/face_models.py missing). "
            "Set DUMMY_MODE=1 to run without models."
        )

    if _RETINA is None:
        _RETINA = load_retinaface()  # type: ignore
    if _ARCFACE is None:
        _ARCFACE = load_arcface()  # type: ignore


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes.")
    return img


def _crop_face(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detect face using RetinaFace and crop the most confident face.
    """
    _ensure_models()

    # If dummy mode, just return center crop-ish region to avoid crashing.
    if DUMMY_MODE:
        h, w = img_bgr.shape[:2]
        y1 = max(0, int(h * 0.15))
        y2 = min(h, int(h * 0.85))
        x1 = max(0, int(w * 0.15))
        x2 = min(w, int(w * 0.85))
        face = img_bgr[y1:y2, x1:x2]
        if face.size == 0:
            raise ValueError("Invalid dummy crop.")
        return face

    bboxes, landmarks = _RETINA.detect(img_bgr)  # type: ignore
    if bboxes is None or len(bboxes) == 0:
        raise ValueError("No face detected.")

    best = max(bboxes, key=lambda x: float(x[4]) if len(x) > 4 else 0.0)
    x1, y1, x2, y2 = [int(v) for v in best[:4]]

    h, w = img_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid face bbox.")

    face = img_bgr[y1:y2, x1:x2]
    return face


def _preprocess_for_arcface(face_bgr: np.ndarray) -> np.ndarray:
    """
    ArcFace commonly expects 112x112 RGB, normalized.
    """
    face = cv2.resize(face_bgr, (112, 112))
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_rgb = face_rgb.astype(np.float32) / 255.0
    return face_rgb


def _dummy_embedding(seed: int = 42, dim: int = 512) -> np.ndarray:
    """
    Deterministic dummy embedding for pipeline testing.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.random(dim, dtype=np.float32)


def get_embedding_from_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode image -> detect+crop face -> ArcFace embedding
    If models are missing (dummy mode), returns a deterministic 512-dim vector.
    """
    if DUMMY_MODE:
        # You can change seed based on image content for variation if you want.
        return _dummy_embedding(seed=42, dim=512)

    _ensure_models()
    img = _decode_image(image_bytes)
    face = _crop_face(img)
    x = _preprocess_for_arcface(face)

    emb = _ARCFACE.get_embedding(x)  # type: ignore
    if emb is None:
        raise ValueError("Failed to get embedding.")
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))
