# api/model_assets.py
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
import tempfile


def _get_urls():
    arc_url = (os.getenv("ARC_MODEL_URL", "") or os.getenv("ARC_URL", "")).strip()
    retina_url = (os.getenv("RETINA_MODEL_URL", "") or os.getenv("RETINA_URL", "")).strip()
    return arc_url, retina_url


def _get_models_dir():
    default_dir = Path(tempfile.gettempdir()) / "face_attendance_models"
    return Path(os.getenv("MODELS_DIR", str(default_dir))).resolve()


def _download(url: str, out_path: Path) -> None:
    if not url:
        raise RuntimeError(f"Missing URL for {out_path.name}")
    tmp = out_path.with_suffix(out_path.suffix + ".download")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(out_path)


def ensure_models() -> dict:
    arc_url, retina_url = _get_urls()
    models_dir = _get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    arc_path = models_dir / "arcface.onnx"
    retina_path = models_dir / "retinaface.onnx"

    print("[model_assets] MODELS_DIR =", models_dir)
    print("[model_assets] ARC_URL set?   =", bool(arc_url))
    print("[model_assets] RETINA_URL set?=", bool(retina_url))

    if not arc_path.exists():
        _download(arc_url, arc_path)
    if not retina_path.exists():
        _download(retina_url, retina_path)

    return {"arcface": str(arc_path), "retinaface": str(retina_path)}
