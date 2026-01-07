from __future__ import annotations

# Load environment variables from .env (project root)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass

import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
    RTCConfiguration,
)
import av
import cv2
import torch

from utils.db_utils import (
    is_db_enabled,
    ensure_camera,
    get_or_create_employee_id,
    insert_attendance_log,
    fetch_attendance_logs,
)
from utils.faiss_utils import load_index
from models.face_models import load_mtcnn, load_facenet


# -----------------------------
# WebRTC config
# -----------------------------
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def now_ts() -> float:
    return time.time()


@dataclass
class RecogState:
    name: str = "Unknown"
    dist: float = 999.0
    recognized: bool = False
    last_seen_ts: float = 0.0
    # person-specific lock
    locked_name: Optional[str] = None
    locked_last_seen_ts: float = 0.0


@st.cache_resource(show_spinner=False)
def _load_models_and_index():
    """
    Heavy resources: load once per Streamlit process.
    """
    mtcnn = load_mtcnn()
    facenet = load_facenet()
    index, names = load_index()
    names = list(names)
    return mtcnn, facenet, index, names


def _extract_embedding(frame_bgr: np.ndarray, mtcnn, facenet) -> Optional[np.ndarray]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)
    if face is None:
        return None
    with torch.no_grad():
        emb = facenet(face.unsqueeze(0)).cpu().numpy().reshape(-1)
    return emb.astype("float32")


def _recognize_one(frame_bgr: np.ndarray, mtcnn, facenet, index, names: list[str]) -> Optional[Tuple[str, float]]:
    emb = _extract_embedding(frame_bgr, mtcnn, facenet)
    if emb is None:
        return None
    # FAISS search: smaller distance = closer
    D, I = index.search(emb.reshape(1, -1), 1)
    dist = float(D[0][0])
    idx = int(I[0][0])
    if idx < 0 or idx >= len(names):
        return None
    return names[idx], dist


# -----------------------------
# Video Processor (NO rerun for video)
# -----------------------------
class AttendanceVideoProcessor(VideoProcessorBase):
    """
    Runs in a separate thread. Do NOT call streamlit APIs here.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.state = RecogState()
        self.threshold = 0.8
        self.unlock_sec = 5.0
        self.process_every_n = 2  # process 1/2 frames for speed
        self._frame_count = 0

        self.mtcnn, self.facenet, self.index, self.names = _load_models_and_index()

    def update_params(self, threshold: float, unlock_sec: float):
        with self._lock:
            self.threshold = float(threshold)
            self.unlock_sec = float(unlock_sec)

    def get_state(self) -> RecogState:
        with self._lock:
            return RecogState(
                name=self.state.name,
                dist=self.state.dist,
                recognized=self.state.recognized,
                last_seen_ts=self.state.last_seen_ts,
                locked_name=self.state.locked_name,
                locked_last_seen_ts=self.state.locked_last_seen_ts,
            )

    def can_commit(self, name: str) -> bool:
        """
        Person-specific lock:
        - If locked_name == name, block until that person is absent for unlock_sec.
        - Other persons are allowed.
        """
        with self._lock:
            locked = self.state.locked_name
            if locked is None:
                return True
            return name != locked

    def mark_committed(self, name: str):
        with self._lock:
            self.state.locked_name = name
            self.state.locked_last_seen_ts = self.state.last_seen_ts

    def _update_lock_release(self):
        """
        Release lock when locked person is not seen for unlock_sec.
        """
        locked = self.state.locked_name
        if locked is None:
            return

        # If current frame recognized locked person, refresh lock last seen
        if self.state.recognized and self.state.name == locked:
            self.state.locked_last_seen_ts = self.state.last_seen_ts
            return

        # If locked person not seen, release after unlock_sec
        if now_ts() - self.state.locked_last_seen_ts >= self.unlock_sec:
            self.state.locked_name = None
            self.state.locked_last_seen_ts = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        self._frame_count += 1
        do_process = (self._frame_count % self.process_every_n == 0)

        if do_process:
            result = _recognize_one(img, self.mtcnn, self.facenet, self.index, self.names)
            with self._lock:
                if result is None:
                    self.state.name = "Unknown"
                    self.state.dist = 999.0
                    self.state.recognized = False
                    # last_seen_ts 유지 (락 타이머용)
                else:
                    name, dist = result
                    self.state.name = name
                    self.state.dist = dist
                    self.state.recognized = (dist <= self.threshold)
                    self.state.last_seen_ts = now_ts()

                self._update_lock_release()

        # Overlay
        with self._lock:
            if self.state.recognized:
                txt = f"{self.state.name} (dist={self.state.dist:.3f})"
                cv2.putText(img, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Waiting for face...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

            if self.state.locked_name:
                cv2.putText(
                    img,
                    f"LOCK: {self.state.locked_name}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 200, 0),
                    2,
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("🎓 Face Recognition Attendance")

DB_ON = is_db_enabled()
camera_id = os.getenv("CAMERA_ID", "CAM_MAIN_01")
camera_label = os.getenv("CAMERA_LABEL", "Terminal")
camera_location = os.getenv("CAMERA_LOCATION", "")

# Session state init
if "mode" not in st.session_state:
    st.session_state.mode = "CHECK_IN"  # default
if "last_auto_commit_ts" not in st.session_state:
    st.session_state.last_auto_commit_ts = 0.0
if "last_auto_commit_name" not in st.session_state:
    st.session_state.last_auto_commit_name = ""
if "last_ui_poll_ts" not in st.session_state:
    st.session_state.last_ui_poll_ts = 0.0

# Header status
if DB_ON:
    st.success(f"DB mode: ON · camera_id={camera_id}")
else:
    st.warning("DB mode: OFF (CSV fallback). Set DATABASE_URL in .env to enable DB.")

st.caption("✅ 운영 UX: **먼저 IN/OUT 모드를 선택 → 카메라 시작 → 인식되면 자동 기록(사람별 락 적용)**")

# Sidebar controls
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Recognition threshold (L2 dist)", 0.3, 2.0, 0.8, 0.05)
unlock_sec = st.sidebar.slider("Unlock seconds (no face)", 1.0, 10.0, 5.0, 0.5)

# Mode selection BEFORE start
st.sidebar.header("Attendance Mode")
st.session_state.mode = st.sidebar.radio(
    "Select mode (before starting camera)",
    options=["CHECK_IN", "CHECK_OUT"],
    index=0 if st.session_state.mode == "CHECK_IN" else 1,
)

st.sidebar.caption("※ WebRTC: 카메라 선택은 브라우저 권한 팝업에서 합니다(카메라 index 없음).")

left, right = st.columns([1.35, 1.0], vertical_alignment="top")

with left:
    ctx = webrtc_streamer(
        key="attendance-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=AttendanceVideoProcessor,
        async_processing=True,
    )

with right:
    st.subheader("Recognition / Auto Logging")

    processor: Optional[AttendanceVideoProcessor] = ctx.video_processor if ctx and ctx.video_processor else None

    status_box = st.empty()
    action_box = st.empty()
    mode_box = st.empty()

    mode_box.info(f"Current mode: **{st.session_state.mode}**  (카메라 시작 전 선택)")

    if processor:
        # Update live params
        processor.update_params(threshold=threshold, unlock_sec=unlock_sec)
        s = processor.get_state()

        # Show recognition state
        if s.recognized:
            status_box.success(f"Detected: **{s.name}** (dist={s.dist:.3f})")
        else:
            status_box.warning("Waiting for face...")

        if s.locked_name:
            action_box.info(f"LOCK active for **{s.locked_name}** (unlock when absent for {unlock_sec:.1f}s)")

        # Auto commit rules (run on each UI run)
        if s.recognized and s.name != "Unknown":
            fresh = (now_ts() - s.last_seen_ts) <= 2.0
            can_commit = processor.can_commit(s.name)
            not_spam = (now_ts() - st.session_state.last_auto_commit_ts) >= 1.0
            not_same_repeat = not (
                st.session_state.last_auto_commit_name == s.name
                and (now_ts() - st.session_state.last_auto_commit_ts) < unlock_sec
            )

            if fresh and can_commit and not_spam and not_same_repeat:
                if not DB_ON:
                    action_box.error("DB mode OFF. Set DATABASE_URL in .env to enable DB insert.")
                else:
                    ok = True
                    err = ""
                    try:
                        ensure_camera(camera_id=camera_id, label=camera_label, location=camera_location)
                        emp_id = get_or_create_employee_id(s.name)
                        score = float(max(0.0, 1.0 - min(1.0, s.dist)))

                        insert_attendance_log(
                            event_type=st.session_state.mode,
                            camera_id=camera_id,
                            recognized=True,
                            similarity=score,
                            employee_id=emp_id,
                        )
                    except Exception as e:
                        ok = False
                        err = str(e)

                    if ok:
                        processor.mark_committed(s.name)  # 사람별 락
                        st.session_state.last_auto_commit_ts = now_ts()
                        st.session_state.last_auto_commit_name = s.name
                        action_box.success(f"✅ AUTO SAVED: {s.name} · {st.session_state.mode} · {camera_id}")
                    else:
                        action_box.error(f"Failed to save: {err}")

        # ✅ Poll UI periodically WITHOUT external package
        # This rerun refreshes only UI and log; WebRTC video stream stays stable.
        if now_ts() - st.session_state.last_ui_poll_ts >= 0.5:
            st.session_state.last_ui_poll_ts = now_ts()
            st.rerun()

    else:
        status_box.warning("Camera not started. 브라우저에서 카메라 권한을 허용한 뒤 Start를 눌러주세요.")
        action_box.info("모드는 먼저 선택해두고(Start 전), 카메라가 시작되면 자동 기록이 됩니다.")

# Attendance Log
st.divider()
st.subheader("Attendance Log (DB)")

if DB_ON:
    try:
        rows = fetch_attendance_logs(limit=200)
        df = pd.DataFrame(rows)

        if not df.empty:
            rename_map = {
                "name": "Name",
                "event_date": "Date",
                "event_time": "Time",
                "event_type": "Status",
                "similarity": "Confidence",
                "camera_id": "Camera",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        st.dataframe(df, use_container_width=True, height=300)
    except Exception as e:
        st.error(f"Failed to read logs from DB: {e}")
else:
    st.info("DB mode is OFF.")
