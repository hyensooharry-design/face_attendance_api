"""
Face detection and recognition model loaders.
Uses MTCNN for face detection and FaceNet (InceptionResnetV1) for embeddings.
"""
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


def load_mtcnn(device="cpu", image_size=160, margin=20):
    """
    Load MTCNN face detection model.
    
    Args:
        device: 'cpu' or 'cuda'
        image_size: Output image size (default 160 for FaceNet)
        margin: Margin around detected face
        
    Returns:
        MTCNN model instance
    """
    print(f"Loading MTCNN on {device}...")
    mtcnn = MTCNN(
        image_size=image_size,
        margin=margin,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],  # Detection thresholds
        factor=0.709,
        post_process=True,
        device=device,
        keep_all=False,  # Only keep the most prominent face
        selection_method='largest'  # Select largest face if multiple detected
    )
    print("✅ MTCNN loaded successfully")
    return mtcnn


def load_facenet(device="cpu", pretrained='vggface2'):
    """
    Load FaceNet (InceptionResnetV1) embedding model.
    
    Args:
        device: 'cpu' or 'cuda'
        pretrained: 'vggface2' or 'casia-webface'
        
    Returns:
        InceptionResnetV1 model instance
    """
    print(f"Loading FaceNet ({pretrained}) on {device}...")
    facenet = InceptionResnetV1(pretrained=pretrained).eval().to(device)
    print("✅ FaceNet loaded successfully")
    return facenet


if __name__ == "__main__":
    # Test model loading
    print("Testing model loading...")
    mtcnn = load_mtcnn()
    facenet = load_facenet()
    print("\n✅ All models loaded successfully!")
