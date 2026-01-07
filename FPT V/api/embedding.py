import numpy as np
import torch
from PIL import Image
import io

from models.face_models import load_mtcnn, load_facenet

# 모델은 서버 시작 시 1번만 로딩 (매 요청마다 로딩하면 느림)
DEVICE = "cpu"  # 가능하면 "cuda"로 바꾸면 빨라짐(환경 되면)
_mtcnn = load_mtcnn(device=DEVICE)
_facenet = load_facenet(device=DEVICE)
_facenet.eval()

def get_embedding_from_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Returns:
        np.ndarray shape (D,) float32  (보통 D=512)
    Raises:
        ValueError: 얼굴 미검출/이미지 읽기 실패 등
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image bytes: {e}")

    face = _mtcnn(img)
    if face is None:
        raise ValueError("No face detected")

    face = face.unsqueeze(0)  # (1,3,160,160)

    with torch.no_grad():
        emb = _facenet(face).cpu().numpy().reshape(-1).astype("float32")
    
    print("🔥 embedding dim =", emb.shape[0])

    return emb
