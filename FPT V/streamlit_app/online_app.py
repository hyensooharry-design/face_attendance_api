import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
    RTCConfiguration,
)
import av

# -----------------------------
# Config
# -----------------------------
API_BASE = os.getenv("API_BASE", "https://face-attendance-api-6pqw.onrender.com").rstrip("/")
DEFAULT_INTERVAL = float(os.getenv("RECOG_INTERVAL_SEC", "1.0"))  # 서버 부담 줄이기
REQ_TIMEOUT = int(os.getenv("REQ_TIMEOUT_SEC", "60"))

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -----------------------------
# Helpers: API calls
# -----------------------------
def api_get(path: str) -> Any:
    r = requests.get(f"{API_BASE}{path}", timeout=REQ_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text[:500]}")
    return r.json()

def api_post_json(path: str, payload: dict) -> Any:
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=REQ_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:500]}")
    return r.json()

def api_post_file(path: str, file_bytes: bytes, filename="image.jpg", fields: Optional[dict]=None) -> Any:
    files = {"file": (filename, file_bytes, "image/jpeg")}
    data = fields or {}
    r = requests.post(f"{API_BASE}{path}", files=files, data=data, timeout=REQ_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:500]}")
    return r.json()

def jpeg_bytes_from_bgr(img_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

# -----------------------------
# WebRTC Video Processor
# -----------------------------
@dataclass
class LastResult:
    ts: float = 0.0
    ok: bool = False
    recognized: bool = False
    name: Optional[str] = None
    employee_id: Optional[int] = None
    similarity: Optional[float] = None
    raw: Optional[dict] = None
    error: Optional[str] = None

class LiveRecognizeProcessor(VideoProcessorBase):
    """
    - 브라우저에서 프레임 수신
    - interval_sec마다 /recognize 호출
    - 결과/에러를 last_result에 저장
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.last_result = LastResult()
        self.camera_id = "CAM_MAIN_01"
        self.event_type = "CHECK_IN"
        self.interval_sec = DEFAULT_INTERVAL
        self._last_sent = 0.0
        self.enabled = True

    def update_params(self, camera_id: str, event_type: str, interval_sec: float, enabled: bool):
        with self._lock:
            self.camera_id = camera_id
            self.event_type = event_type
            self.interval_sec = float(interval_sec)
            self.enabled = bool(enabled)

    def get_last(self) -> LastResult:
        with self._lock:
            return LastResult(**self.last_result.__dict__)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        with self._lock:
            enabled = self.enabled
            interval = self.interval_sec
            camera_id = self.camera_id
            event_type = self.event_type
            can_send = enabled and (now - self._last_sent >= interval)

        if can_send:
            with self._lock:
                self._last_sent = now

            try:
                jpg = jpeg_bytes_from_bgr(img)
                j = api_post_file(
                    "/recognize",
                    jpg,
                    filename="frame.jpg",
                    fields={"camera_id": camera_id, "event_type": event_type},
                )
                lr = LastResult(
                    ts=now,
                    ok=True,
                    recognized=bool(j.get("recognized", False)),
                    name=j.get("name"),
                    employee_id=j.get("employee_id"),
                    similarity=j.get("similarity"),
                    raw=j,
                    error=None,
                )
            except Exception as e:
                lr = LastResult(ts=now, ok=False, error=str(e), raw=None)

            with self._lock:
                self.last_result = lr

        # Overlay
        lr2 = self.get_last()
        if lr2.error:
            cv2.putText(img, f"API ERR: {lr2.error[:70]}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            if lr2.recognized:
                txt = f"OK {lr2.name or lr2.employee_id} sim={lr2.similarity}"
                cv2.putText(img, txt, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                cv2.putText(img, "Recognizing...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Face Attendance Online (Streamlit)", layout="wide")
st.title("🧩 Face Attendance Online (Streamlit + WebRTC + FastAPI + Supabase)")
st.caption(f"API_BASE = {API_BASE}")

# Sidebar menu
with st.sidebar:
    st.header("Menu")
    menu = st.radio(
        "Select",
        ["Live Attendance", "Employees", "Enroll Face", "Cameras", "Logs", "API Health"],
        label_visibility="collapsed"
    )

def page_api_health():
    st.subheader("API Health Check")
    cols = st.columns(3)
    with cols[0]:
        if st.button("GET /health"):
            try:
                st.json(api_get("/health"))
            except Exception as e:
                st.error(str(e))
    with cols[1]:
        if st.button("Open /docs URL"):
            st.write(f"{API_BASE}/docs")
    with cols[2]:
        if st.button("GET / (root)"):
            try:
                st.json(api_get("/"))
            except Exception as e:
                st.error(str(e))

def page_cameras():
    st.subheader("Cameras")
    col1, col2 = st.columns([1.0, 1.2])

    with col1:
        st.markdown("### Add Camera")
        camera_id = st.text_input("camera_id", placeholder="CAM_ENT_01")
        label = st.text_input("label", placeholder="Entrance")
        location = st.text_input("location", placeholder="Building A")
        is_active = st.checkbox("is_active", value=True)
        if st.button("Create camera"):
            try:
                # 백엔드에 POST /cameras가 없으면 여기서 에러가 뜸 -> 그게 바로 원인
                st.json(api_post_json("/cameras", {
                    "camera_id": camera_id.strip(),
                    "label": label.strip() or None,
                    "location": location.strip() or None,
                    "is_active": is_active,
                }))
                st.success("Created.")
            except Exception as e:
                st.error(str(e))

    with col2:
        st.markdown("### Camera List")
        try:
            cams = api_get("/cameras")
            st.json(cams)
        except Exception as e:
            st.error(str(e))

def page_employees():
    st.subheader("Employees")
    col1, col2 = st.columns([1.0, 1.2])

    with col1:
        st.markdown("### Create Employee")
        name = st.text_input("name", placeholder="홍길동")
        code = st.text_input("employee_code (optional)", placeholder="EMP001")
        is_active = st.checkbox("active", value=True)
        if st.button("Create employee"):
            try:
                st.json(api_post_json("/employees", {
                    "name": name.strip(),
                    "employee_code": code.strip() or None,
                    "is_active": is_active,
                }))
                st.success("Created.")
            except Exception as e:
                st.error(str(e))

    with col2:
        st.markdown("### Employee List")
        q = st.text_input("search (name/code)", placeholder="홍 / EMP")
        try:
            path = f"/employees?limit=200&query={requests.utils.quote(q)}" if q else "/employees?limit=200"
            rows = api_get(path)
            st.dataframe(rows, use_container_width=True, height=360)
        except Exception as e:
            st.error(str(e))

def page_enroll_face():
    st.subheader("Enroll Face (Employee)")
    st.caption("직원 선택 → 사진 업로드 → 서버에서 임베딩 생성/저장")

    try:
        emps = api_get("/employees?limit=200")
    except Exception as e:
        st.error(f"Failed to load employees: {e}")
        return

    if not emps:
        st.warning("No employees. Create one first.")
        return

    options = {f'{r.get("employee_id")} - {r.get("name")}': r.get("employee_id") for r in emps}
    sel = st.selectbox("Select employee", list(options.keys()))
    emp_id = options[sel]

    img = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
    if st.button("Enroll") and img is not None:
        try:
            data = img.read()
            # 백엔드 엔드포인트가 다르면 여기서 에러가 뜸 -> 그게 현재 미구현/불일치의 증거
            j = api_post_file(f"/employees/{emp_id}/faces", data, filename=img.name, fields={})
            st.success("Enrolled.")
            st.json(j)
        except Exception as e:
            st.error(str(e))

def page_logs():
    st.subheader("Attendance Logs")
    limit = st.slider("limit", 50, 1000, 200, 50)
    try:
        rows = api_get(f"/logs?limit={limit}")
        st.dataframe(rows, use_container_width=True, height=520)
    except Exception as e:
        st.error(str(e))

def page_live_attendance():
    st.subheader("Live Attendance (WebRTC)")
    st.caption("브라우저 웹캠 → 일정 주기마다 서버 /recognize 호출 → DB에 로그 저장")

    top = st.columns([1.2, 1.0])
    with top[0]:
        # camera list
        try:
            cams = api_get("/cameras")
        except Exception as e:
            st.error(f"/cameras failed: {e}")
            st.info("카메라가 안 뜨면: 1) DB에 cameras가 비었거나 2) 백엔드에 /cameras 라우터가 없거나 3) 권한/환경변수 문제")
            cams = []

        cam_ids = [c.get("camera_id") for c in cams if c.get("camera_id")] or ["CAM_MAIN_01"]
        camera_id = st.selectbox("camera_id", cam_ids)

        event_type = st.radio("event_type", ["CHECK_IN", "CHECK_OUT"], horizontal=True)
        enabled = st.toggle("Enable API recognize calls", value=True)
        interval = st.slider("recognize interval (sec)", 0.4, 3.0, DEFAULT_INTERVAL, 0.1)
        st.caption("서버가 느리면 interval을 1.0~2.0으로 올려. (Render 콜드스타트/추론 때문에)")

    with top[1]:
        st.markdown("### Last Result / Error")
        box = st.empty()
        st.markdown("### Recent Logs (auto refresh)")
        log_box = st.empty()

    ctx = webrtc_streamer(
        key="webrtc-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=LiveRecognizeProcessor,
        async_processing=True,
    )

    proc = ctx.video_processor if ctx and ctx.video_processor else None
    if proc:
        proc.update_params(camera_id=camera_id, event_type=event_type, interval_sec=interval, enabled=enabled)
        lr = proc.get_last()

        if lr.error:
            box.error(lr.error)
        else:
            box.json(lr.raw or {})

        # logs refresh (light polling)
        try:
            rows = api_get("/logs?limit=50")
            log_box.dataframe(rows, use_container_width=True, height=260)
        except Exception as e:
            log_box.error(str(e))
    else:
        st.info("Start 버튼 누르고 브라우저에서 카메라 권한을 허용해줘.")

# Router
if menu == "API Health":
    page_api_health()
elif menu == "Cameras":
    page_cameras()
elif menu == "Employees":
    page_employees()
elif menu == "Enroll Face":
    page_enroll_face()
elif menu == "Logs":
    page_logs()
else:
    page_live_attendance()
