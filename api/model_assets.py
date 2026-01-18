# api/model_assets.py
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
import tempfile


# ✅ 너가 이미 올려둔 GitHub Release URL로 fallback
DEFAULT_ARC_URL = "https://github.com/hyensooharry-design/face_attendance_api/releases/download/models-v1/arcface.onnx"
DEFAULT_RETINA_URL = "https://github.com/hyensooharry-design/face_attendance_api/releases/download/models-v1/retinaface.onnx"


def _get_urls():
    arc_url = (os.getenv("ARC_MODEL_URL", "") or os.getenv("ARC_URL", "")).strip()
    retina_url = (os.getenv("RETINA_MODEL_URL", "") or os.getenv("RETINA_URL", "")).strip()

    # ✅ env 없으면 기본값 사용
    if not arc_url:
        arc_url = DEFAULT_ARC_URL
    if not retina_url:
        retina_url = DEFAULT_RETINA_URL

    return arc_url, retina_url


def _get_models_dir():
    default_dir = Path(tempfile.gettempdir()) / "face_attendance_models"
    return Path(os.getenv("MODELS_DIR", str(default_dir))).resolve()


def _download(url: str, out_path: Path) -> None:
    # 여기서 url 비면 죽는 구조였는데, 이제는 사실상 비지 않게 됨
    if not url:
        raise RuntimeError(f"Missing URL for {out_path.name}")

    tmp = out_path.with_suffix(out_path.suffix + ".download")
    print(f"[models] downloading: {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(out_path)
    print(f"[models] saved: {out_path} ({out_path.stat().st_size/1024/1024:.2f} MB)")


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

    print("[models] done.")
    return {"arcface": str(arc_path), "retinaface": str(retina_path)}
