"""FAISS utilities for building and querying face embedding indices."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import faiss

from utils.project_paths import FAISS_DIR, DATA_DIR, ensure_dirs


def get_default_db_path() -> Path:
    ensure_dirs()
    return DATA_DIR / "face_db.npy"


def get_default_index_path() -> Path:
    ensure_dirs()
    return FAISS_DIR / "face.index"


def get_default_names_path() -> Path:
    ensure_dirs()
    return FAISS_DIR / "names.npy"


def load_face_db(db_path: Optional[str] = None) -> dict:
    path = Path(db_path) if db_path else get_default_db_path()
    if not path.exists():
        raise FileNotFoundError(f"Face DB not found: {path}")
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError("face_db.npy must contain a dict: {name: embedding}")
    return data


def build_faiss_index(
    db_path: Optional[str] = None,
    index_path: Optional[str] = None,
    names_path: Optional[str] = None,
) -> Tuple[Path, Path, int]:
    """Build FAISS index from face_db.npy.

    Returns:
        (index_path, names_path, n_vectors)
    """
    ensure_dirs()
    face_db = load_face_db(db_path)

    names = sorted(face_db.keys())
    if not names:
        raise ValueError("Face DB is empty. Add dataset images and run vectorize first.")

    vectors = np.stack([np.asarray(face_db[n], dtype="float32") for n in names], axis=0)

    if vectors.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array (N, D), got shape {vectors.shape}")

    n, d = vectors.shape
    index = faiss.IndexFlatL2(d)
    index.add(vectors)

    ip = Path(index_path) if index_path else get_default_index_path()
    npy = Path(names_path) if names_path else get_default_names_path()

    ip.parent.mkdir(parents=True, exist_ok=True)
    npy.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(ip))
    np.save(str(npy), np.array(names, dtype=object))

    return ip, npy, n


def load_faiss_assets(
    index_path: Optional[str] = None,
    names_path: Optional[str] = None,
) -> Tuple[faiss.Index, np.ndarray]:
    ip = Path(index_path) if index_path else get_default_index_path()
    npy = Path(names_path) if names_path else get_default_names_path()

    if not ip.exists():
        raise FileNotFoundError(f"FAISS index not found: {ip}")
    if not npy.exists():
        raise FileNotFoundError(f"Names file not found: {npy}")

    index = faiss.read_index(str(ip))
    names = np.load(str(npy), allow_pickle=True)

    if index.ntotal != len(names):
        raise ValueError(
            f"Index/name mismatch: index.ntotal={index.ntotal} vs len(names)={len(names)}. "
            "Rebuild the index."
        )

    return index, names


def search_face(
    index: faiss.Index,
    names: np.ndarray,
    query_vec: np.ndarray,
    threshold: float = 0.8,
) -> Tuple[Optional[str], float]:
    """Return (name, distance) if under threshold else (None, distance)."""
    q = np.asarray(query_vec, dtype="float32")
    if q.ndim == 1:
        q = q.reshape(1, -1)

    distances, indices = index.search(q, 1)
    dist = float(distances[0][0])
    idx = int(indices[0][0])

    if idx < 0 or idx >= len(names):
        return None, dist

    name = str(names[idx])
    if dist <= float(threshold):
        return name, dist
    return None, dist


# -------------------------------------------------------------------
# Compatibility wrapper (for Streamlit/WebRTC app)
# -------------------------------------------------------------------
def load_index(
    index_path: Optional[str] = None,
    names_path: Optional[str] = None,
) -> Tuple[faiss.Index, np.ndarray]:
    """
    Backward-compatible wrapper.

    Streamlit/WebRTC app expects:
        index, names = load_index()

    This project already provides load_faiss_assets(), so we wrap it.
    """
    return load_faiss_assets(index_path=index_path, names_path=names_path)
