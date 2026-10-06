from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "ai"

DEFAULT_ARC_URL = (
    "https://github.com/hyensooharry-design/face_attendance_api/"
    "releases/download/models-v1/arcface.onnx"
)
DEFAULT_ARC_SHA256 = "f3a6bc281e72f88862f5748b53be3d76b3b48f8f1ab1f4a537941bdc4e1b01da"

# Kept as an optional asset for backwards compatibility. The current
# inference path uses OpenCV for face detection and ArcFace ONNX for embeddings.
DEFAULT_RETINA_URL = (
    "https://github.com/hyensooharry-design/face_attendance_api/"
    "releases/download/models-v1/retinaface.onnx"
)
DEFAULT_RETINA_SHA256 = "40f825cf7dd0a88b26fb61db9a3aaedc2cad35162091113f4017b3c26a4f792d"


def get_models_dir() -> Path:
    raw = os.getenv("MODELS_DIR", "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_MODELS_DIR.resolve()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, out_path: Path, expected_sha256: str = "") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        if expected_sha256:
            if _sha256(out_path).lower() == expected_sha256.lower():
                print(f"[models] cached: {out_path.name}")
                return
            print(f"[models] checksum mismatch; re-downloading {out_path.name}")
        else:
            print(f"[models] cached: {out_path.name}")
            return

    fd, temp_name = tempfile.mkstemp(prefix=out_path.name + ".", suffix=".download", dir=str(out_path.parent))
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        print(f"[models] downloading: {url}")
        urllib.request.urlretrieve(url, temp_path)

        if expected_sha256:
            got = _sha256(temp_path)
            if got.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"SHA256 mismatch for {out_path.name}: got={got}, expected={expected_sha256}"
                )

        temp_path.replace(out_path)
        print(f"[models] saved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def ensure_models(download_retinaface: bool | None = None) -> dict[str, str]:
    models_dir = get_models_dir()

    arc_url = (
        os.getenv("ARC_MODEL_URL", "").strip()
        or os.getenv("ARCFACE_URL", "").strip()
        or os.getenv("ARC_URL", "").strip()
        or DEFAULT_ARC_URL
    )
    arc_sha = (
        os.getenv("ARC_MODEL_SHA256", "").strip()
        or os.getenv("ARCFACE_SHA256", "").strip()
        or DEFAULT_ARC_SHA256
    )

    arc_path = models_dir / "arcface.onnx"
    _download(arc_url, arc_path, arc_sha)

    result = {"arcface": str(arc_path)}

    if download_retinaface is None:
        download_retinaface = os.getenv("DOWNLOAD_RETINAFACE", "0").strip() == "1"

    if download_retinaface:
        retina_url = (
            os.getenv("RETINA_MODEL_URL", "").strip()
            or os.getenv("RETINAFACE_URL", "").strip()
            or os.getenv("RETINA_URL", "").strip()
            or DEFAULT_RETINA_URL
        )
        retina_sha = (
            os.getenv("RETINA_MODEL_SHA256", "").strip()
            or os.getenv("RETINAFACE_SHA256", "").strip()
            or DEFAULT_RETINA_SHA256
        )
        retina_path = models_dir / "retinaface.onnx"
        _download(retina_url, retina_path, retina_sha)
        result["retinaface"] = str(retina_path)

    return result
