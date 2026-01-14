from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

import api_client as api_service  # 네 api_client.py 기준 :contentReference[oaicite:1]{index=1}

# (있으면 쓰고, 없으면 무시) - 기존 프로젝트 호환
try:
    from styles import theme
except Exception:
    theme = None

try:
    from ui import header, overlays, sidebar
except Exception:
    header = overlays = sidebar = None


# -----------------------------
# Config
# -----------------------------
load_dotenv()
DEFAULT_API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
SCAN_INTERVAL_SEC = 1.5

# 최대 5개 카메라 슬롯 (요구사항 1)
CAM_POOL = [f"CAM_{i:02d}" for i in range(1, 6)]


# -----------------------------
# Utils
# -----------------------------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def _ensure_session_ids() -> None:
    # 세션 고유 ID (재접속하면 바뀔 수 있음)
    st.session_state.setdefault("client_uuid", uuid.uuid4().hex)

    # CAM ID 자동 부여: uuid 기반으로 0~4 슬롯에 매핑
    if "cam_id" not in st.session_state:
        u = st.session_state["client_uuid"]
        idx = int(u[-2:], 16) % len(CAM_POOL)
        st.session_state["cam_id"] = CAM_POOL[idx]

    # 인식 횟수(요구사항 2: 홀수=CHECK_IN, 짝수=CHECK_OUT)
    st.session_state.setdefault("scan_count", 0)

    # 마지막 인식 결과/시간
    st.session_state.setdefault("last_scan_ts", 0.0)
    st.session_state.setdefault("last_scan_result", None)
    st.session_state.setdefault("last_event_message", "")

    # 화면에 표시할 로그/스케줄 캐시
    st.session_state.setdefault("camera_logs", [])
    st.session_state.setdefault("user_schedules", [])


def _next_event_type_by_count() -> str:
    # 요구사항 2: 홀수번째 인식 => CHECK_IN, 짝수번째 인식 => CHECK_OUT
    # scan_count는 "성공 인식" 기준으로만 증가시킴
    next_n = st.session_state.get("scan_count", 0) + 1
    return "CHECK_IN" if (next_n % 2 == 1) else "CHECK_OUT"


def _format_schedule_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 표시용 정리
    out = []
    for r in rows:
        out.append(
            {
                "schedule": r.get("schedule"),
                "start_time": r.get("start_time"),
                "end_time": r.get("end_time"),
            }
        )
    return out


def _refresh_logs(api_base: str, cam_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    # api_client.fetch_logs(limit)만 있으니, 받아서 cam_id로 필터링
    try:
        rows = api_service.fetch_logs(limit=limit, api_base=api_base)  # :contentReference[oaicite:2]{index=2}
        rows = [r for r in rows if _safe_str(r.get("camera_id")) == cam_id]
        # 최신순 가정(아니면 event_time 기준 정렬)
        rows.sort(key=lambda x: _safe_str(x.get("event_time") or x.get("created_at")), reverse=True)
        return rows[:20]
    except Exception:
        return []


def _refresh_schedules(api_base: str, employee_id: int) -> List[Dict[str, Any]]:
    # api_client.list_schedules(limit)만 있으니, 받아서 employee_id로 필터링
    try:
        rows = api_service.list_schedules(limit=200, api_base=api_base)  # :contentReference[oaicite:3]{index=3}
        rows = [r for r in rows if int(r.get("employee_id")) == int(employee_id)]
        # start_time 오름차순
        rows.sort(key=lambda x: _safe_str(x.get("start_time")))
        return rows
    except Exception:
        return []


# -----------------------------
# WebRTC Video Processor
# -----------------------------
class CamProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.latest_bgr: Optional[Any] = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_bgr = img
        return frame


# -----------------------------
# UI
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Timekeeping", layout="wide")

    if theme and hasattr(theme, "apply"):
        try:
            theme.apply()
        except Exception:
            pass

    _ensure_session_ids()

    # Header
    if header and hasattr(header, "page_header"):
        try:
            header.page_header("Timekeeping", "Realtime face recognition timekeeping")
        except Exception:
            st.title("Timekeeping")
    else:
        st.title("Timekeeping")

    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")

        api_base = st.text_input("API Base", value=st.session_state.get("api_base", DEFAULT_API_BASE))
        st.session_state["api_base"] = api_base

        # CAM ID 자동 부여 + 수동 override 옵션(디버깅용)
        st.markdown("### Camera")
        auto_cam = st.session_state.get("cam_id")
        cam_id = st.selectbox("Assigned CAM ID", options=CAM_POOL, index=CAM_POOL.index(auto_cam))
        st.session_state["cam_id"] = cam_id

        threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
        st.session_state["threshold"] = float(threshold)

        st.markdown("### Event Mode")
        st.caption("Odd recognition = CHECK_IN, Even recognition = CHECK_OUT (session-based).")
        st.write(f"Next event_type: **{_next_event_type_by_count()}**")
        if st.button("Reset scan counter"):
            st.session_state["scan_count"] = 0

        st.markdown("---")
        st.caption("Tip: This simple CAM assignment can collide if 2 users hash to same slot. "
                   "If you want collision-free assignment, we can add a tiny API/DB lock using the cameras table.")

    api_base = st.session_state["api_base"]
    cam_id = st.session_state["cam_id"]
    threshold = st.session_state["threshold"]

    # (선택) 접속 시 카메라 레코드 보장 (있으면 생성, 실패해도 진행)
    try:
        api_service.create_camera(cam_id, is_active=True, api_base=api_base)  # :contentReference[oaicite:4]{index=4}
    except Exception:
        pass

    # Layout: left(cam) / right(info)
    col_cam, col_info = st.columns([2.0, 1.0], gap="large")

    with col_cam:
        st.markdown(f"### Camera View — `{cam_id}`")

        ctx = webrtc_streamer(
            key=f"webrtc-{cam_id}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=CamProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Greeting message area (요구사항 5)
        msg = st.session_state.get("last_event_message", "")
        if msg:
            st.success(msg)

        # Logs below camera (요구사항 4)
        st.markdown("### Recent Logs (this camera)")
        logs_rows = st.session_state.get("camera_logs", [])
        if logs_rows:
            # 표시용 컬럼만
            show = []
            for r in logs_rows[:10]:
                show.append(
                    {
                        "event_time": r.get("event_time"),
                        "event_type": r.get("event_type"),
                        "employee_id": r.get("employee_id"),
                        "recognized": r.get("recognized"),
                        "similarity": r.get("similarity"),
                        "log_id": r.get("log_id"),
                    }
                )
            st.dataframe(show, use_container_width=True, hide_index=True)
        else:
            st.info("No logs yet.")

    with col_info:
        st.markdown("### Last Recognition")
        last = st.session_state.get("last_scan_result")

        if last:
            recognized = bool(last.get("recognized"))
            st.write(f"recognized: **{recognized}**")
            st.write(f"employee: **{_safe_str(last.get('name'))}**  (id={_safe_str(last.get('employee_id'))})")
            st.write(f"similarity: **{_safe_str(last.get('similarity'))}**")
            st.write(f"event_type: **{_safe_str(last.get('event_type'))}**")
            st.write(f"log_id: **{_safe_str(last.get('log_id'))}**")
        else:
            st.info("Waiting for scan...")

        # Schedule on right (요구사항 3)
        st.markdown("### Schedules (recognized user)")
        sched_rows = st.session_state.get("user_schedules", [])
        if sched_rows:
            st.dataframe(_format_schedule_rows(sched_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No schedule loaded yet.")

    # -----------------------------
    # Background scan loop
    # -----------------------------
    if ctx and ctx.state.playing and ctx.video_processor:
        now = time.time()
        if now - float(st.session_state.get("last_scan_ts", 0.0)) > SCAN_INTERVAL_SEC:
            frame = getattr(ctx.video_processor, "latest_bgr", None)
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    event_type = _next_event_type_by_count()

                    try:
                        resp = api_service.recognize(
                            buf.tobytes(),
                            event_type=event_type,
                            camera_id=cam_id,
                            api_base=api_base,
                        )  # :contentReference[oaicite:5]{index=5}

                        st.session_state["last_scan_result"] = resp
                        st.session_state["last_scan_ts"] = now

                        # recognized일 때만 카운트 증가 + 메시지/스케줄/로그 갱신
                        if bool(resp.get("recognized")) and resp.get("employee_id") is not None:
                            st.session_state["scan_count"] = int(st.session_state.get("scan_count", 0)) + 1

                            name = _safe_str(resp.get("name") or "there")
                            # 요구사항 5: Hello/Bye
                            if event_type == "CHECK_IN":
                                st.session_state["last_event_message"] = f"Hello {name}!"
                                try:
                                    st.toast(f"Hello {name}!", icon="✅")
                                except Exception:
                                    pass
                            else:
                                st.session_state["last_event_message"] = f"Bye {name}!"
                                try:
                                    st.toast(f"Bye {name}!", icon="👋")
                                except Exception:
                                    pass

                            # 요구사항 3: 스케줄 로딩
                            try:
                                emp_id = int(resp["employee_id"])
                                st.session_state["user_schedules"] = _refresh_schedules(api_base, emp_id)
                            except Exception:
                                st.session_state["user_schedules"] = []

                            # 요구사항 4: 로그 로딩
                            st.session_state["camera_logs"] = _refresh_logs(api_base, cam_id)

                        # 화면 갱신
                        st.rerun()

                    except Exception as e:
                        st.session_state["last_scan_ts"] = now
                        st.session_state["last_event_message"] = f"Recognize failed: {repr(e)}"
                        st.error(st.session_state["last_event_message"])
                        st.stop()



if __name__ == "__main__":
    main()
