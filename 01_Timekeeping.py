from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import cv2
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

import api_client as api_service
from styles import theme
from ui import header, overlays, sidebar

load_dotenv()

API_BASE_DEFAULT = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
SCAN_INTERVAL_SEC = float(os.getenv("SCAN_INTERVAL_SEC", "1.5"))
AUTO_REFRESH_MS = int(os.getenv("AUTO_REFRESH_MS", "500"))

st.set_page_config(page_title="Timekeeping", page_icon="📷", layout="wide")
theme.apply()
sidebar.render_sidebar()

api_base = (st.session_state.get("api_base") or API_BASE_DEFAULT).rstrip("/")
st.session_state.setdefault("last_scan_ts", 0.0)
st.session_state.setdefault("last_scan_result", None)
st.session_state.setdefault("last_scan_error", None)
st.session_state.setdefault("scan_enabled", True)

header.render_header("Timekeeping Area", "Place your face in the frame to record attendance.")

settings_left, settings_mid, settings_right = st.columns([2, 2, 3])
with settings_left:
    event_type = st.selectbox("Attendance event", ["CHECK_IN", "CHECK_OUT"], index=0)
with settings_mid:
    camera_id = st.text_input("Camera ID", value="CAM_MAIN").strip() or "CAM_MAIN"
with settings_right:
    api_base = st.text_input("API Base", value=api_base).rstrip("/")
    st.session_state["api_base"] = api_base

col_cam, col_info = st.columns([2, 1])


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_bgr = None

    def recv(self, frame):
        self.latest_bgr = frame.to_ndarray(format="bgr24")
        return frame


with col_cam:
    overlays.render_viewfinder()
    ctx = webrtc_streamer(
        key="timekeeping",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
        async_processing=True,
    )

with col_info:
    scan_enabled = st.toggle("Enable scanning", value=st.session_state.scan_enabled)
    st.session_state.scan_enabled = scan_enabled

    res = st.session_state.get("last_scan_result")
    err = st.session_state.get("last_scan_error")

    if err:
        st.error(err)

    if res:
        if res.get("recognized"):
            overlays.render_success_message(
                res.get("name") or "Employee",
                res.get("employee_code") or "",
                float(res.get("similarity") or 0.0),
            )
        else:
            overlays.render_denied_message()
    else:
        st.info("Waiting for scan...")

    with st.expander("Debug", expanded=False):
        st.write(
            {
                "api_base": api_base,
                "camera_id": camera_id,
                "event_type": event_type,
                "playing": bool(getattr(ctx.state, "playing", False)),
                "has_video_processor": bool(ctx.video_processor),
                "last_scan_ts": st.session_state.get("last_scan_ts"),
            }
        )


def _encode_jpg(bgr, max_w: int = 640, quality: int = 85) -> Optional[bytes]:
    if bgr is None:
        return None

    h, w = bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


def _autorefresh_when_playing() -> None:
    if getattr(ctx.state, "playing", False):
        st.query_params["_t"] = str(int(time.time() * 1000) // AUTO_REFRESH_MS)


def _should_scan() -> bool:
    if not st.session_state.scan_enabled:
        return False
    if not getattr(ctx.state, "playing", False):
        return False
    if not ctx.video_processor:
        return False
    last = float(st.session_state.get("last_scan_ts") or 0.0)
    return (time.time() - last) >= SCAN_INTERVAL_SEC


_autorefresh_when_playing()

if _should_scan():
    frame = ctx.video_processor.latest_bgr if ctx.video_processor else None
    jpg = _encode_jpg(frame)

    if jpg is None:
        st.session_state.last_scan_error = "Camera frame is not ready yet."
        st.session_state.last_scan_ts = time.time()
        st.rerun()

    try:
        resp: Dict[str, Any] = api_service.recognize(
            jpg,
            event_type,
            camera_id,
            api_base,
        )
        st.session_state.last_scan_result = resp
        st.session_state.last_scan_error = None
    except Exception as exc:
        st.session_state.last_scan_error = (
            f"Recognize call failed: {type(exc).__name__}: {exc}"
        )
    finally:
        st.session_state.last_scan_ts = time.time()
        st.rerun()
