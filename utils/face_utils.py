import cv2
import torch

def get_embedding(frame, mtcnn, facenet):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)

    if face is None:
        return None

    face = face.unsqueeze(0)

    with torch.no_grad():
        embedding = facenet(face)

    return embedding.squeeze().cpu().numpy()
