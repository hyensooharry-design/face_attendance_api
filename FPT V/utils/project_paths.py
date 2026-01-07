"""Project path helpers.

Keep all filesystem paths consistent regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    # utils/ is directly under the project root
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT: Path = get_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
FAISS_DIR: Path = PROJECT_ROOT / "faiss_index"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_from_root(*parts: str) -> Path:
    """Resolve a path under the project root."""
    return PROJECT_ROOT.joinpath(*parts)
