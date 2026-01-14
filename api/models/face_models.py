"""
Face detection and recognition model loaders.
Uses RetinaFace for face detection and ArcFace for embeddings (ONNX).
"""

import os
import onnxruntime as ort

# -------------------------------------------------
# Model paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "ai")

RETINAFACE_MODEL_PATH = os.path.join(MODEL_DIR, "retinaface.onnx")
ARCFACE_MODEL_PATH = os.path.join(MODEL_DIR, "arcface.onnx")


# -------------------------------------------------
# RetinaFace Loader (Face Detection)
# -------------------------------------------------
def load_retinaface(device: str = "cpu"):
    """
    Load RetinaFace ONNX model for face detection.

    Args:
        device: 'cpu' or 'cuda'

    Returns:
        ONNX Runtime InferenceSession
    """
    print(f"Loading RetinaFace on {device}...")

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )

    session = ort.InferenceSession(
        RETINAFACE_MODEL_PATH,
        providers=providers
    )

    print("✅ RetinaFace loaded successfully")
    return session


# -------------------------------------------------
# ArcFace Loader (Face Embedding)
# -------------------------------------------------
def load_arcface(device: str = "cpu"):
    """
    Load ArcFace ONNX model for face embeddings.

    Args:
        device: 'cpu' or 'cuda'

    Returns:
        ONNX Runtime InferenceSession
    """
    print(f"Loading ArcFace on {device}...")

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )

    session = ort.InferenceSession(
        ARCFACE_MODEL_PATH,
        providers=providers
    )

    print("✅ ArcFace loaded successfully")
    return session


# -------------------------------------------------
# Test loader
# -------------------------------------------------
if __name__ == "__main__":
    print("Testing model loading...")

    retinaface = load_retinaface()
    arcface = load_arcface()

    print("\n✅ All models loaded successfully!")
