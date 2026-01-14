from typing import Any, Dict, List, Optional
from fastapi import HTTPException

def raise_if_error(resp: Any, msg: str) -> None:
    err = getattr(resp, "error", None)
    if err:
        raise HTTPException(status_code=500, detail=f"{msg}: {err}")

def ensure_one(resp: Any, msg: str) -> Dict[str, Any]:
    raise_if_error(resp, msg)
    if not resp.data:
        raise HTTPException(status_code=404, detail=f"{msg}: not found")
    # supabase-py의 single()은 dict, 아니면 list일 수 있음
    if isinstance(resp.data, list):
        if len(resp.data) == 0:
            raise HTTPException(status_code=404, detail=f"{msg}: not found")
        return resp.data[0]
    return resp.data
