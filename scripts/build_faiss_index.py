"""Build FAISS index from data/face_db.npy."""
from __future__ import annotations

from utils.faiss_utils import build_faiss_index


def main() -> None:
    index_path, names_path, n = build_faiss_index()
    print(f"✅ Built FAISS index with {n} vectors")
    print(f" - index: {index_path}")
    print(f" - names: {names_path}")


if __name__ == "__main__":
    main()
