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


# ---------------------------------------------------------------------
# Env / Config
# ---------------------------------------------------------------------
load_dotenv()

API_BASE_DEFAULT = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
SCAN_INTERVAL_SEC = float(os.getenv("SCAN_INTERVAL_SEC", "1.5"))  # default 1.5s
CAMERA_ID_DEFAULT = os.getenv("CAMERA_ID_DEFAULT", "CAM_MAIN")
EVENT_TYPE_DEFAULT = os.getenv("EVENT_TYPE_DEFAULT", "CHECK_IN")


# ---------------------------------------------------------------------
# Streamlit Setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Timekeeping", page_icon="📷", layout="wide")
theme.apply()

# Sidebar may update st.session_state["api_base"] etc.
sidebar.render_sidebar()

# Resolve api_base (session > env > default)
api_base = (st.session_state.get("api_base") or API_BASE_DEFAULT).rstrip("/")

# Session defaults
st.session_state.setdefault("last_scan_ts", 0.0)
st.session_state.setdefault("last_scan_result", None)  # type: ignore[assignment]
st.session_state.setdefault("last_scan_error", None)
st.session_state.setdefault("scan_enabled", True)  # you can toggle if you want later


# ---------------------------------------------------------------------
# UI Header
# ---------------------------------------------------------------------
header.render_header("Timekeeping Area", "Please place your face in the frame.")

col_cam, col_info = st.columns([2, 1])


# ---------------------------------------------------------------------
# Video Processor
# ---------------------------------------------------------------------
class VideoProcessor(VideoProcessorBase):
    """Keep the latest frame for periodic scanning."""
    def __init__(self) -> None:
        self.latest_bgr = None

    def recv(self, frame):
        self.latest_bgr = frame.to_ndarray(format="bgr24")
        return frame


# ---------------------------------------------------------------------
# LEFT COLUMN: Camera View
# ---------------------------------------------------------------------
with col_cam:
    st.markdown(
        '<div class="viewfinder-container">'
        '<div class="corner tl"></div><div class="corner tr"></div>'
        '<div class="corner bl"></div><div class="corner br"></div>'
        '<div class="scanline"></div>',
        unsafe_allow_html=True,
    )

    ctx = webrtc_streamer(
        key="timekeeping",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
        async_processing=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# RIGHT COLUMN: Results
# ---------------------------------------------------------------------
with col_info:
    # Optional: show which API is being used (helps debugging)
    with st.expander("Debug", expanded=False):
        st.write("API Base:", api_base)
        st.write("Event Type:", EVENT_TYPE_DEFAULT)
        st.write("Camera ID:", CAMERA_ID_DEFAULT)
        if st.session_state.get("last_scan_error"):
            st.error(st.session_state["last_scan_error"])

    res = st.session_state.get("last_scan_result")

    if isinstance(res, dict):
        if res.get("recognized") is True:
            overlays.render_success_message(
                res.get("name"),
                res.get("employee_code"),
                res.get("similarity"),
            )
        else:
            overlays.render_denied_message()
    else:
        st.info("Waiting for scan...")


# ---------------------------------------------------------------------
# Helper: Safe recognize call
# ---------------------------------------------------------------------
def _recognize_once(
    frame_bgr,
    *,
    event_type: str,
    camera_id: str,
    api_base_url: str,
) -> Optional[Dict[str, Any]]:
    """Encode frame -> call /recognize -> normalize response to dict."""
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPG.")

    resp = api_service.recognize(
        buf.tobytes(),
        event_type,
        camera_id,
        api_base_url,
    )

    # Normalize/validate
    if resp is None:
        raise RuntimeError("API returned empty response (None).")

    if not isinstance(resp, dict):
        raise RuntimeError(f"API returned non-dict response: {type(resp)}")

    # Ensure keys exist at least
    resp.setdefault("recognized", False)
    return resp


# ---------------------------------------------------------------------
# BACKGROUND PROCESS: periodic scan (every SCAN_INTERVAL_SEC)
# ---------------------------------------------------------------------
should_scan = (
    st.session_state.get("scan_enabled", True)
    and ctx is not None
    and ctx.state.playing
    and ctx.video_processor is not None
)

if should_scan:
    now = time.time()
    last_ts = float(st.session_state.get("last_scan_ts", 0.0))
    if (now - last_ts) >= SCAN_INTERVAL_SEC:
        frame = getattr(ctx.video_processor, "latest_bgr", None)

        # Advance the timer even if frame is None to prevent tight rerun loops
        st.session_state["last_scan_ts"] = now

        if frame is not None:
            try:
                st.session_state["last_scan_error"] = None
                result = _recognize_once(
                    frame,
                    event_type=EVENT_TYPE_DEFAULT,
                    camera_id=CAMERA_ID_DEFAULT,
                    api_base_url=api_base,
                )
                st.session_state["last_scan_result"] = result
            except Exception as e:
                # Don’t hide errors — this is what makes “model conflict?” confusion.
                st.session_state["last_scan_result"] = {"recognized": False}
                st.session_state["last_scan_error"] = f"{type(e).__name__}: {e}"

            # Trigger UI update once per scan attempt
            st.rerun()
