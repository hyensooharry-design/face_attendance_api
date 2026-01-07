"""Vectorize dataset/ faces into data/face_db.npy.

For each person folder under dataset/<Name>/, this script:
- loads images
- detects face with MTCNN
- computes embedding with FaceNet
- averages embeddings per identity
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from models.face_models import load_mtcnn, load_facenet
from utils.project_paths import DATASET_DIR, DATA_DIR, ensure_dirs


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _iter_images(person_dir: Path) -> List[Path]:
    paths = []
    for p in person_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return sorted(paths)


def main() -> None:
    ensure_dirs()
    dataset_dir = DATASET_DIR
    out_path = DATA_DIR / "face_db.npy"

    mtcnn = load_mtcnn(device="cpu")
    facenet = load_facenet(device="cpu")
    facenet.eval()

    face_db: Dict[str, np.ndarray] = {}
    skipped_total = 0

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    people = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if not people:
        raise ValueError(f"No person folders found under {dataset_dir}")

    for person in people:
        name = person.name
        imgs = _iter_images(person)
        if not imgs:
            print(f"⚠️  No images for {name}, skipping")
            continue

        embs = []
        skipped = 0
        for img_path in imgs:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                skipped += 1
                continue

            face = mtcnn(img)
            if face is None:
                skipped += 1
                continue

            face = face.unsqueeze(0)  # (1,3,160,160)
            with torch.no_grad():
                emb = facenet(face).cpu().numpy().reshape(-1)
            embs.append(emb)

        if skipped:
            print(f"ℹ️  {name}: skipped {skipped}/{len(imgs)} images (no face / read error)")
            skipped_total += skipped

        if len(embs) < 1:
            print(f"⚠️  {name}: no valid face found in any image, skipping")
            continue

        avg = np.mean(np.stack(embs, axis=0), axis=0).astype("float32")
        face_db[name] = avg
        print(f"✅ {name}: {len(embs)} embeddings -> saved")

    np.save(out_path, face_db)
    print(f"\n✅ Saved face DB with {len(face_db)} identities to: {out_path}")
    if skipped_total:
        print(f"ℹ️  Total skipped images: {skipped_total}")


if __name__ == "__main__":
    main()
