from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from api.model_assets import get_models_dir


class HaarFaceDetector:
    """Small, dependency-free frontal-face detector bundled with OpenCV."""

    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV Haar cascade: {cascade_path}")

    def detect(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        return [tuple(int(v) for v in box) for box in boxes]


class ArcFaceEncoder:
    """Generic ArcFace-style ONNX encoder using ONNX Runtime."""

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ArcFace model not found: {path}")

        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_meta = self.session.get_inputs()[0]
        self.output_meta = self.session.get_outputs()[0]

        shape = list(self.input_meta.shape)
        self.channels_last = len(shape) == 4 and shape[-1] == 3

        if len(shape) != 4:
            raise RuntimeError(f"Unsupported ArcFace input shape: {shape}")

        if self.channels_last:
            self.height = int(shape[1]) if isinstance(shape[1], int) else 112
            self.width = int(shape[2]) if isinstance(shape[2], int) else 112
        else:
            self.height = int(shape[2]) if isinstance(shape[2], int) else 112
            self.width = int(shape[3]) if isinstance(shape[3], int) else 112

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Standard ArcFace normalization.
        face = (face - 127.5) / 127.5

        if self.channels_last:
            batch = np.expand_dims(face, axis=0)
        else:
            batch = np.transpose(face, (2, 0, 1))[None, ...]

        return np.ascontiguousarray(batch, dtype=np.float32)

    def get_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        batch = self._preprocess(face_bgr)
        outputs = self.session.run([self.output_meta.name], {self.input_meta.name: batch})
        embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if embedding.size == 0:
            raise RuntimeError("ArcFace model returned an empty embedding.")

        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        return embedding


def load_face_detector() -> HaarFaceDetector:
    return HaarFaceDetector()


def load_arcface(device: str = "cpu") -> ArcFaceEncoder:
    model_path = get_models_dir() / "arcface.onnx"
    return ArcFaceEncoder(model_path, device=device)


# Backwards-compatible alias retained for older imports. It now returns the
# reliable OpenCV detector rather than a raw ONNX Runtime session.
def load_retinaface(device: str = "cpu") -> HaarFaceDetector:
    _ = device
    return load_face_detector()
