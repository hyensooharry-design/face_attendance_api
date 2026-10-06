from __future__ import annotations

import hashlib
import os

import cv2
import numpy as np

from api.models.face_models import load_arcface, load_face_detector

DUMMY_MODE = os.getenv("DUMMY_MODE", "0").strip() == "1"

_DETECTOR = None
_ARCFACE = None


def _ensure_models() -> None:
    global _DETECTOR, _ARCFACE
    if DUMMY_MODE:
        return

    if _DETECTOR is None:
        _DETECTOR = load_face_detector()
    if _ARCFACE is None:
        device = os.getenv("FACE_DEVICE", "cpu").strip().lower() or "cpu"
        _ARCFACE = load_arcface(device=device)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes.")
    return image


def _largest_face_crop(image_bgr: np.ndarray) -> np.ndarray:
    _ensure_models()
    boxes = _DETECTOR.detect(image_bgr)
    if not boxes:
        raise ValueError("No face detected.")

    x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])

    # Add a small margin so the recognition model receives the whole face.
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image_bgr.shape[1], x + w + margin_x)
    y2 = min(image_bgr.shape[0], y + h + margin_y)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Detected face crop is empty.")
    return crop


def _dummy_embedding(image_bytes: bytes, dim: int = 512) -> np.ndarray:
    """Deterministic test embedding derived from image content."""
    digest = hashlib.sha256(image_bytes).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    embedding = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(embedding))
    return embedding / norm if norm > 0 else embedding


def get_embedding_from_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Empty image bytes.")

    if DUMMY_MODE:
        return _dummy_embedding(image_bytes)

    _ensure_models()
    image = _decode_image(image_bytes)
    face = _largest_face_crop(image)
    return _ARCFACE.get_embedding(face)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Embedding shape mismatch: {a.shape} vs {b.shape}")

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)
