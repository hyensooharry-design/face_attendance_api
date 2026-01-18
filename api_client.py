# UI/api_client.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

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
    """
    여러 URL fallback 시도.
    - 성공(2xx/3xx): JSON이면 JSON, 아니면 text 반환
    - 실패(4xx/5xx): 가능한 한 서버가 준 JSON(detail/trace)을 포함해 에러 메시지에 담음
    """
    last_err: Optional[str] = None

    for url in urls:
        try:
            r = requests.request(method, url, headers=_headers(), timeout=timeout, **kwargs)
            ctype = (r.headers.get("content-type") or "").lower()

            if r.status_code < 400:
                return r.json() if "application/json" in ctype else r.text

            # 실패: JSON이면 JSON을 최대한 살려서 노출
            if "application/json" in ctype:
                try:
                    j = r.json()
                    last_err = f"{r.status_code} {str(j)[:1200]}"
                except Exception:
                    last_err = f"{r.status_code} {r.text[:1200]}"
            else:
                last_err = f"{r.status_code} {r.text[:1200]}"

        except Exception as e:
            last_err = str(e)

    raise ApiError(f"API call failed. Tried: {urls}\nLast error: {last_err}")


def _as_list(res: Any) -> List[Dict[str, Any]]:
    if isinstance(res, list):
        return [x for x in res if isinstance(x, dict)]
    if isinstance(res, dict) and isinstance(res.get("data"), list):
        return [x for x in res["data"] if isinstance(x, dict)]
    return []


def _as_dict(res: Any) -> Dict[str, Any]:
    return res if isinstance(res, dict) else {"result": res}


def _wrap_recognize_response(res: Any) -> Dict[str, Any]:
    if not isinstance(res, dict):
        return {"recognized": False, "message": str(res)}

    recognized = bool(res.get("recognized", res.get("matched", False)))
    similarity = res.get("similarity", res.get("score", None))

    out = dict(res)
    out["recognized"] = recognized
    out["similarity"] = similarity
    return out


# =========================
# Employees
# =========================
def list_employees(query: str = "", limit: int = 50, api_base: str = "") -> List[Dict[str, Any]]:
    b = _base(api_base)
    url = f"{b}/employees"

    # 백엔드가 q를 쓰든 query를 쓰든 둘 다 보내서 호환
    params: Dict[str, Any] = {"limit": limit}
    if query is not None:
        params["q"] = query
        params["query"] = query

    res = _try_urls("GET", [url], params=params)
    return _as_list(res)


def create_employee(name: str, employee_code: Optional[str] = None, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/employees"
    payload: Dict[str, Any] = {"name": name}
    if employee_code:
        payload["employee_code"] = employee_code
    res = _try_urls("POST", [url], json=payload)
    return _as_dict(res)


def update_employee(
    employee_id: int,
    *,
    name: Optional[str] = None,
    employee_code: Optional[str] = None,
    is_active: Optional[bool] = None,
    role: Optional[str] = None,
    api_base: str = "",
) -> Dict[str, Any]:
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
    return _as_dict(res)


def delete_employee(employee_id: int, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/employees/{employee_id}"
    res = _try_urls("DELETE", [url])
    return _as_dict(res)


# =========================
# Faces (persons/face_embeddings)
# =========================
def list_faces(limit: int = 200, api_base: str = "") -> List[Dict[str, Any]]:
    """
    주의: 백엔드 구현에 따라 반환 필드가 다를 수 있음.
    (예: employee_id, person_id, created_at, model_name, model_version 등)
    """
    b = _base(api_base)
    url = f"{b}/faces"
    res = _try_urls("GET", [url], params={"limit": limit})
    return _as_list(res)


def enroll_face(employee_id: int, image_bytes: bytes, api_base: str = "") -> Dict[str, Any]:
    """
    얼굴 등록:
    - 1순위: /faces/enroll/{employee_id}
    - 2순위: /employees/{employee_id}/enroll-face (레거시 호환)
    """
    b = _base(api_base)
    files = {"file": ("face.jpg", image_bytes, "image/jpeg")}
    urls = [
        f"{b}/faces/enroll/{employee_id}",
        f"{b}/employees/{employee_id}/enroll-face",
    ]
    res = _try_urls("POST", urls, files=files, timeout=60)
    return _as_dict(res)


def delete_face(employee_id: int, api_base: str = "") -> Dict[str, Any]:
    """
    얼굴 삭제:
    백엔드에 따라 엔드포인트가 다를 수 있어 fallback 제공.
    - /faces/{employee_id} (레거시/단순)
    - /faces/by-employee/{employee_id} (패치 버전에 있을 수 있음)
    - /employees/{employee_id}/delete-face (있을 수 있는 레거시)
    """
    b = _base(api_base)
    urls = [
        f"{b}/faces/{employee_id}",
        f"{b}/faces/by-employee/{employee_id}",
        f"{b}/employees/{employee_id}/delete-face",
    ]
    res = _try_urls("DELETE", urls)
    return _as_dict(res)


# =========================
# Logs (attendance_logs)
# =========================
def fetch_logs(limit: int = 200, api_base: str = "") -> List[Dict[str, Any]]:
    b = _base(api_base)
    url = f"{b}/logs"
    res = _try_urls("GET", [url], params={"limit": limit})
    return _as_list(res)


# =========================
# Cameras
# =========================
def list_cameras(limit: int = 200, api_base: str = "") -> List[Dict[str, Any]]:
    b = _base(api_base)
    url = f"{b}/cameras"
    res = _try_urls("GET", [url], params={"limit": limit})
    return _as_list(res)


def create_camera(
    camera_id: str,
    *,
    name: Optional[str] = None,
    location: Optional[str] = None,
    api_base: str = "",
) -> Dict[str, Any]:
    """
    스키마 기준 cameras 테이블에는 is_active가 없음.
    따라서 camera_id (+ optional name/location)만 전송.
    """
    b = _base(api_base)
    url = f"{b}/cameras"

    payload: Dict[str, Any] = {"camera_id": camera_id}
    if name is not None:
        payload["name"] = name
    if location is not None:
        payload["location"] = location

    res = _try_urls("POST", [url], json=payload)
    return _as_dict(res)


def update_camera(
    camera_id: str,
    *,
    name: Optional[str] = None,
    location: Optional[str] = None,
    api_base: str = "",
) -> Dict[str, Any]:
    """
    카메라 상태 토글(is_active)은 스키마에 없어서 지원하지 않음.
    필요한 경우 스키마에 컬럼을 추가하고 백엔드도 함께 맞춰야 함.
    """
    b = _base(api_base)
    url = f"{b}/cameras/{camera_id}"

    payload: Dict[str, Any] = {}
    if name is not None:
        payload["name"] = name
    if location is not None:
        payload["location"] = location

    # 빈 payload면 의미없는 요청이라 에러 대신 no-op 반환
    if not payload:
        return {"result": "no-op (no fields to update)", "camera_id": camera_id}

    res = _try_urls("PATCH", [url], json=payload)
    return _as_dict(res)


def delete_camera(camera_id: str, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/cameras/{camera_id}"
    res = _try_urls("DELETE", [url])
    return _as_dict(res)


# =========================
# Schedules
# =========================
def list_schedules(limit: int = 200, api_base: str = "") -> List[Dict[str, Any]]:
    b = _base(api_base)
    url = f"{b}/schedules"
    res = _try_urls("GET", [url], params={"limit": limit})
    return _as_list(res)


def create_schedule(
    employee_id: int,
    schedule: str,
    start_time: str,
    end_time: str,
    api_base: str = "",
) -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/schedules"
    payload = {
        "employee_id": employee_id,
        "schedule": schedule,
        "start_time": start_time,
        "end_time": end_time,
    }
    res = _try_urls("POST", [url], json=payload)
    return _as_dict(res)


def delete_schedule(schedule_id: int, api_base: str = "") -> Dict[str, Any]:
    b = _base(api_base)
    url = f"{b}/schedules/{schedule_id}"
    res = _try_urls("DELETE", [url])
    return _as_dict(res)


# =========================
# Recognize
# =========================
def recognize(image_bytes: bytes, event_type: str, camera_id: str, api_base: str = "") -> Dict[str, Any]:
    """
    event_type: 보통 "CHECK_IN" | "CHECK_OUT" 권장
    camera_id: 스키마 상 cameras.camera_id (TEXT)
    """
    b = _base(api_base)
    url = f"{b}/recognize"
    files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
    data = {"event_type": event_type, "camera_id": camera_id}
    res = _try_urls("POST", [url], files=files, data=data, timeout=60)
    return _wrap_recognize_response(res)
