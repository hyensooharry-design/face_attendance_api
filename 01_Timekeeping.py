from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any

import cv2
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

import api_client as api_service
from styles import theme
from ui import header, overlays, sidebar

load_dotenv()

# ------------------------------------
# Config
# ------------------------------------
API_BASE_DEFAULT = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
SCAN_INTERVAL_SEC = float(os.getenv("SCAN_INTERVAL_SEC", "1.5"))  # scan every N seconds
AUTO_REFRESH_MS = int(os.getenv("AUTO_REFRESH_MS", "500"))        # rerun UI every N ms when camera is on

# ------------------------------------
# Setup
# ------------------------------------
st.set_page_config(page_title="Timekeeping", page_icon="📷", layout="wide")
theme.apply()
sidebar.render_sidebar()

api_base = (st.session_state.get("api_base") or API_BASE_DEFAULT).rstrip("/")

# session defaults
st.session_state.setdefault("last_scan_ts", 0.0)
st.session_state.setdefault("last_scan_result", None)   # type: Optional[Dict[str, Any]]
st.session_state.setdefault("last_scan_error", None)    # type: Optional[str]
st.session_state.setdefault("scan_enabled", True)

# UI
header.render_header("Timekeeping Area", "Please place your face in the frame.")

col_cam, col_info = st.columns([2, 1])

# ------------------------------------
# Video processor
# ------------------------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_bgr = None

    def recv(self, frame):
        self.latest_bgr = frame.to_ndarray(format="bgr24")
        return frame

# ------------------------------------
# Camera column
# ------------------------------------
with col_cam:
    st.markdown(
        '<div class="viewfinder-container">'
        '<div class="corner tl"></div><div class="corner tr"></div>'
        '<div class="corner bl"></div><div class="corner br"></div>'
        '<div class="scanline"></div>',
        unsafe_allow_html=True
    )

    ctx = webrtc_streamer(
        key="timekeeping",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
        async_processing=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------
# Result column
# ------------------------------------
with col_info:
    st.caption(f"API Base: `{api_base}`")

    scan_enabled = st.toggle("Enable scanning", value=st.session_state.scan_enabled)
    st.session_state.scan_enabled = scan_enabled

    res = st.session_state.get("last_scan_result")
    err = st.session_state.get("last_scan_error")

    if err:
        st.error(err)

    if res:
        if res.get("recognized"):
            overlays.render_success_message(res.get("name"), res.get("employee_code"), res.get("similarity"))
        else:
            overlays.render_denied_message()
    else:
        st.info("Waiting for scan...")

    with st.expander("Debug", expanded=False):
        st.write(
            {
                "playing": bool(getattr(ctx.state, "playing", False)),
                "has_video_processor": bool(ctx.video_processor),
                "last_scan_ts": st.session_state.get("last_scan_ts"),
                "now": time.time(),
            }
        )

# ------------------------------------
# Helper: encode frame smaller (reduce latency)
# ------------------------------------
def _encode_jpg(bgr, max_w: int = 640, quality: int = 85) -> Optional[bytes]:
    if bgr is None:
        return None

    h, w = bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return buf.tobytes()

# ------------------------------------
# Auto refresh (so scanning continues even if API fails)
# ------------------------------------
def _autorefresh_when_playing():
    if getattr(ctx.state, "playing", False):
        # Streamlit의 experimental rerun 루프를 안정화하기 위한 트릭:
        # 특정 시간 간격마다 query param을 바꿔 rerun 유도
        st.query_params["_t"] = str(int(time.time() * 1000) // AUTO_REFRESH_MS)

_autorefresh_when_playing()

# ------------------------------------
# Scan step
# ------------------------------------
def _should_scan() -> bool:
    if not st.session_state.scan_enabled:
        return False
    if not getattr(ctx.state, "playing", False):
        return False
    if not ctx.video_processor:
        return False
    last = float(st.session_state.get("last_scan_ts") or 0.0)
    return (time.time() - last) >= SCAN_INTERVAL_SEC

if _should_scan():
    frame = ctx.video_processor.latest_bgr if ctx.video_processor else None
    jpg = _encode_jpg(frame)

    if jpg is None:
        st.session_state.last_scan_error = "Camera frame is not ready yet."
        st.session_state.last_scan_ts = time.time()
        st.rerun()
    else:
        try:
            # Default is CHECK_IN, Camera Default
            resp = api_service.recognize(jpg, "CHECK_IN", "CAM_MAIN", api_base)

            st.session_state.last_scan_result = resp
            st.session_state.last_scan_error = None
            st.session_state.last_scan_ts = time.time()
            st.rerun()

        except Exception as e:
            st.session_state.last_scan_error = f"Recognize call failed: {type(e).__name__}: {e}"
            st.session_state.last_scan_ts = time.time()
            st.rerun()
