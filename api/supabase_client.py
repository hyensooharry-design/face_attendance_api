# api/supabase_client.py
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

_supabase: Optional[Client] = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is not None:
        return _supabase

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()

    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment (.env)")

    _supabase = create_client(url, key)
    return _supabase
