#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from api.model_assets import ensure_models


def main() -> int:
    models = ensure_models()
    for name, path in models.items():
        print(f"[models] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
