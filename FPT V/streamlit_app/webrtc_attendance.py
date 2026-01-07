import os
import time
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

API_BASE = os.getenv("API_BASE", "https://face-attendance-api-6pqw.onrender.com")

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@dataclass
class LastResult:
    ts: float = 0.0
    recognized: bool = False
    employee_id: Optional[int] = None
    similarity: Optional[float] = None
    name: Optional[str] = None
    raw: Optional[dict] = None
    err: Optional[str] = None

class RecognizeVideoProcessor(VideoProcessorBase):
    """
    - 브라우저에서 들어오는 프레임을 받음
    - 일정 주기마다 FastAPI /recognize로 업로드
    - 결과는 self.last_result에 저장
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.last_result = LastResult()
        self.camera_id = "CAM_MAIN_01"
        self.event_type = "CHECK_IN"
        self.interval_sec = 0.7  # 0.7초마다 1번 업로드
        self._last_sent = 0.0

    def update_params(self, camera_id: str, event_type: str, interval_sec: float):
        with self._lock:
            self.camera_id = camera_id
            self.event_type = event_type
            self.interval_sec = float(interval_sec)

    def get_last_result(self) -> LastResult:
        with self._lock:
            return LastResult(**self.last_result.__dict__)

    def _call_recognize_api(self, frame_bgr: np.ndarray, camera_id: str, event_type: str):
        # JPEG 인코딩
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return {"err": "jpeg encode failed"}

        files = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}
        data = {"camera_id": camera_id, "event_type": event_type}

        try:
            r = requests.post(f"{API_BASE}/recognize", files=files, data=data, timeout=20)
            if r.status_code != 200:
                return {"err": f"HTTP {r.status_code}: {r.text[:400]}"}
            return {"json": r.json()}
        except Exception as e:
            return {"err": str(e)}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        with self._lock:
            camera_id = self.camera_id
            event_type = self.event_type
            interval = self.interval_sec
            can_send = (now - self._last_sent) >= interval

        # 일정 주기마다 API 호출
        if can_send:
            with self._lock:
                self._last_sent = now

            resp = self._call_recognize_api(img, camera_id, event_type)

            with self._lock:
                if "err" in resp:
                    self.last_result = LastResult(ts=now, err=resp["err"])
                else:
                    j = resp["json"]
                    self.last_result = LastResult(
                        ts=now,
                        recognized=bool(j.get("recognized", False)),
                        employee_id=j.get("employee_id"),
                        similarity=j.get("similarity"),
                        name=j.get("name"),
                        raw=j,
                        err=None,
                    )

        # 화면 오버레이 (결과 표시)
        lr = self.get_last_result()
        if lr.err:
            cv2.putText(img, f"ERR: {lr.err[:60]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            if lr.recognized:
                txt = f"OK {lr.name or lr.employee_id} sim={lr.similarity}"
                cv2.putText(img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                cv2.putText(img, "Recognizing...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="WebRTC Attendance (Streamlit)", layout="wide")
st.title("🎥 Real-time Face Attendance (Streamlit + WebRTC)")

# 카메라 목록은 우선 하드코딩/환경변수로 시작해도 되고,
# 나중에 GET /cameras 붙이면 됨.
camera_id = st.selectbox("camera_id", ["CAM_MAIN_01", "CAM_ENT_01", "CAM_LAB_02"])
event_type = st.radio("event_type", ["CHECK_IN", "CHECK_OUT"], horizontal=True)
interval = st.slider("recognize interval (sec)", 0.3, 2.0, 0.7, 0.1)

col1, col2 = st.columns([1.4, 1.0])

with col1:
    ctx = webrtc_streamer(
        key="webrtc-attendance",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=RecognizeVideoProcessor,
        async_processing=True,
    )

with col2:
    st.subheader("Last recognize result")
    box = st.empty()

    proc = ctx.video_processor if ctx and ctx.video_processor else None
    if proc:
        proc.update_params(camera_id=camera_id, event_type=event_type, interval_sec=interval)
        lr = proc.get_last_result()
        if lr.err:
            box.error(lr.err)
        else:
            box.json(lr.raw or {})
    else:
        st.info("카메라 Start 후 권한을 허용하세요.")
