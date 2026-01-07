# api/supabase_client.py
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# ✅ 프로젝트 루트의 .env를 강제로 로드
ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ROOT_ENV)

def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise RuntimeError(f"Missing env vars. Loaded env from: {ROOT_ENV}")

    return create_client(url, key)
