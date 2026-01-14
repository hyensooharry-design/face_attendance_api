# UI/api_client.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "").strip()  # 있으면 사용, 없으면 빈값


class ApiError(RuntimeError):
    pass


def _headers() -> Dict[str, str]:
    h: Dict[str, str] = {}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def _base(api_base: Optional[str]) -> str:
    b = (api_base or DEFAULT_API_BASE).rstrip("/")
    if not b:
        b = "http://127.0.0.1:8000"
    return b


def _try_urls(method: str, urls: List[str], *, timeout: float = 30, **kwargs) -> Any:
    last_err: Optional[str] = None
    for url in urls:
        try:
            r = requests.request(method, url, headers=_headers(), timeout=timeout, **kwargs)
            if r.status_code < 400:
                ctype = (r.headers.get("content-type") or "").lower()
                return r.json() if "application/json" in ctype else r.text
            last_err = f"{r.status_code} {r.text[:300]}"
        except Exception as e:
            last_err = str(e)
    raise ApiError(f"API call failed. Tried: {urls}\nLast error: {last_err}")


def _wrap_recognize_response(res: Any) -> Dict[str, Any]:
    if not isinstance(res, dict):
        return {"recognized": False, "message": str(res)}

    recognized = bool(res.get("recognized", res.get("matched", False)))
    similarity = res.get("similarity", res.get("score", None))

    out = dict(res)
    out["recognized"] = recognized
    out["similarity"] = similarity
    return out


# -------------------------
# Employees
# -------------------------
def list_employees(query: str = "", limit: int = 50, api_base: str = "") -> List[Dict[str, Any]]:
    b = _base(api_base)
    url = f"{b}/employees"
    # 백엔드가 q를 쓰든 query를 쓰든 둘 다 보내서 호환
    params = {"limit": limit}
    if query is not None:
        params["q"] = query
        params["query"] = query
    res = _try_urls("GET", [url], params=params)
    if isinstance(res, list):
        return res
    if isinstance(res, dict) and isinstance(res.get("data"), list):
        return res["data"]
    return []


def create_employee(name: str, employee_code: Optional[str] = None, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/employees"
    payload: Dict[str, Any] = {"name": name}
    if employee_code:
        payload["employee_code"] = employee_code
    res = _try_urls("POST", [url], json=payload)
    return res if isinstance(res, dict) else {"result": res}


def update_employee(employee_id: int, *, name: Optional[str] = None, employee_code: Optional[str] = None,
                    is_active: Optional[bool] = None, role: Optional[str] = None, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/employees/{employee_id}"
    payload: Dict[str, Any] = {}
    if name is not None:
        payload["name"] = name
    if employee_code is not None:
        payload["employee_code"] = employee_code
    if is_active is not None:
        payload["is_active"] = is_active
    if role is not None:
        payload["role"] = role
    res = _try_urls("PATCH", [url], json=payload)
    return res if isinstance(res, dict) else {"result": res}


def delete_employee(employee_id: int, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/employees/{employee_id}"
    res = _try_urls("DELETE", [url])
    return res if isinstance(res, dict) else {"result": res}


# -------------------------
# Faces (face_embeddings)
# -------------------------
def list_faces(limit: int = 200, api_base: str = "") -> List[Dict[str, Any]]:
    b = _base(api_base)
    url = f"{b}/faces"
    res = _try_urls("GET", [url], params={"limit": limit})
    if isinstance(res, list):
        return res
    if isinstance(res, dict) and isinstance(res.get("data"), list):
        return res["data"]
    return []


def enroll_face(employee_id: int, image_bytes: bytes, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    files = {"file": ("face.jpg", image_bytes, "image/jpeg")}
    urls = [
        f"{b}/faces/enroll/{employee_id}",
        f"{b}/employees/{employee_id}/enroll-face",
    ]
    res = _try_urls("POST", urls, files=files, timeout=60)
    return res if isinstance(res, dict) else {"result": res}


def delete_face(employee_id: int, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/faces/{employee_id}"
    res = _try_urls("DELETE", [url])
    return res if isinstance(res, dict) else {"result": res}


# -------------------------
# Logs (attendance_logs)
# -------------------------
def fetch_logs(limit: int = 200, api_base: str = "") -> List[Dict[str, Any]]:
    b = _base(api_base)
    url = f"{b}/logs"
    # 백엔드가 limit를 query로 받는다고 가정
    res = _try_urls("GET", [url], params={"limit": limit})
    if isinstance(res, list):
        return res
    if isinstance(res, dict) and isinstance(res.get("data"), list):
        return res["data"]
    return []


# -------------------------
# Cameras
# -------------------------
def list_cameras(limit: int = 200, api_base: str = "") -> List[Dict[str, Any]]:
    b = _base(api_base)
    url = f"{b}/cameras"
    res = _try_urls("GET", [url], params={"limit": limit})
    if isinstance(res, list):
        return res
    if isinstance(res, dict) and isinstance(res.get("data"), list):
        return res["data"]
    return []


def create_camera(camera_id: str, is_active: bool = True, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/cameras"
    res = _try_urls("POST", [url], json={"camera_id": camera_id, "is_active": is_active})
    return res if isinstance(res, dict) else {"result": res}


def toggle_camera(camera_id: str, is_active: bool, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/cameras/{camera_id}"
    res = _try_urls("PATCH", [url], json={"is_active": is_active})
    return res if isinstance(res, dict) else {"result": res}


def delete_camera(camera_id: str, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/cameras/{camera_id}"
    res = _try_urls("DELETE", [url])
    return res if isinstance(res, dict) else {"result": res}


# -------------------------
# Schedules
# -------------------------
def list_schedules(limit: int = 200, api_base: str = "") -> List[Dict[str, Any]]:
    b = _base(api_base)
    url = f"{b}/schedules"
    res = _try_urls("GET", [url], params={"limit": limit})
    if isinstance(res, list):
        return res
    if isinstance(res, dict) and isinstance(res.get("data"), list):
        return res["data"]
    return []


def create_schedule(employee_id: int, schedule: str, start_time: str, end_time: str, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/schedules"
    payload = {
        "employee_id": employee_id,
        "schedule": schedule,
        "start_time": start_time,
        "end_time": end_time,
    }
    res = _try_urls("POST", [url], json=payload)
    return res if isinstance(res, dict) else {"result": res}


def delete_schedule(schedule_id: int, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/schedules/{schedule_id}"
    res = _try_urls("DELETE", [url])
    return res if isinstance(res, dict) else {"result": res}


# -------------------------
# Recognize
# -------------------------
def recognize(image_bytes: bytes, event_type: str, camera_id: str, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/recognize"
    files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
    data = {"event_type": event_type, "camera_id": camera_id}
    res = _try_urls("POST", [url], files=files, data=data, timeout=60)
    return _wrap_recognize_response(res)
