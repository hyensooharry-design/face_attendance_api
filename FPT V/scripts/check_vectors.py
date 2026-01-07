"""Sanity checks for face DB and FAISS assets."""
from __future__ import annotations

import numpy as np

from utils.faiss_utils import load_face_db, load_faiss_assets, build_faiss_index


def main() -> None:
    print("=== Checking face_db.npy ===")
    face_db = load_face_db()
    print(f"✅ Loaded face DB with {len(face_db)} identities")

    # Check embeddings
    dims = set()
    bad = []
    for name, vec in face_db.items():
        arr = np.asarray(vec)
        if arr.ndim != 1:
            bad.append((name, f"ndim={arr.ndim}"))
            continue
        dims.add(arr.shape[0])
        if arr.dtype not in (np.float32, np.float64):
            bad.append((name, f"dtype={arr.dtype}"))
    if bad:
        print("⚠️  Found potential issues in embeddings:")
        for n, why in bad[:20]:
            print(f" - {n}: {why}")
        if len(bad) > 20:
            print(f" ... and {len(bad)-20} more")

    if len(dims) != 1:
        print(f"⚠️  Embedding dimension inconsistent: {sorted(dims)}")
    else:
        d = list(dims)[0] if dims else None
        print(f"✅ Embedding dimension: {d}")

    print("\n=== Checking FAISS assets ===")
    try:
        index, names = load_faiss_assets()
        print(f"✅ Loaded FAISS index (ntotal={index.ntotal}) and names (len={len(names)})")
    except Exception as e:
        print(f"⚠️  FAISS assets not valid: {e}")
        print("→ Rebuilding index...")
        index_path, names_path, n = build_faiss_index()
        print(f"✅ Rebuilt FAISS index with {n} vectors")
        print(f" - index: {index_path}")
        print(f" - names: {names_path}")


if __name__ == "__main__":
    main()
