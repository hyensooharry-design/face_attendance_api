# api/common.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
from fastapi import HTTPException


def execute_or_500(fn: Callable[[], Any], msg: str) -> Any:
    """
    supabase-py / postgrest-py는 실패 시 보통 예외를 던진다.
    (APIResponse.error 같은 속성에 의존하면 안 됨)
    """
    try:
        return fn()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{msg}: {e}")


def get_data(resp: Any) -> List[Dict[str, Any]]:
    data = getattr(resp, "data", None)
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def get_one_or_404(resp: Any, not_found_msg: str) -> Dict[str, Any]:
    rows = get_data(resp)
    if not rows:
        raise HTTPException(status_code=404, detail=not_found_msg)
    return rows[0]
