"""
Attendance logging and management utilities.
- Robust to missing/empty CSV (prevents Streamlit crash)
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd

from utils.db_utils import (
    ensure_camera,
    fetch_attendance_logs,
    get_or_create_employee_id,
    insert_attendance_log,
    is_db_enabled,
)

from pandas.errors import EmptyDataError


COLUMNS = ["Name", "Date", "Time", "Status", "Confidence"]


def _ensure_csv(csv_path: str) -> None:
    """Ensure attendance CSV exists and has header."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Create file with header if missing
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=COLUMNS).to_csv(csv_path, index=False)
        return

    # If exists but empty (0 bytes), write header
    try:
        if os.path.getsize(csv_path) == 0:
            pd.DataFrame(columns=COLUMNS).to_csv(csv_path, index=False)
    except OSError:
        # If file is in a weird state, try rewriting header safely
        pd.DataFrame(columns=COLUMNS).to_csv(csv_path, index=False)


def read_attendance(csv_path: str = "data/attendance.csv") -> pd.DataFrame:
    """Read attendance records.

    - If DATABASE_URL is set, reads recent rows from Supabase Postgres (attendance_logs join employees).
    - Otherwise, reads local CSV (backward compatible).
    """
    if is_db_enabled():
        try:
            rows = fetch_attendance_logs(limit=500)
            if not rows:
                return pd.DataFrame(columns=COLUMNS + ["Camera"])
            df = pd.DataFrame(rows)
            # Normalize columns
            for c in (COLUMNS + ["Camera"]):
                if c not in df.columns:
                    df[c] = ""
            return df[COLUMNS + ["Camera"]]
        except Exception:
            # Fail safe: return empty with optional Camera column
            return pd.DataFrame(columns=COLUMNS + ["Camera"])

    # CSV fallback
    _ensure_csv(csv_path)

    try:
        df = pd.read_csv(csv_path)
    except EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    # Normalize: if file has only headers or partial columns
    missing = [c for c in COLUMNS if c not in df.columns]
    for c in missing:
        df[c] = ""

    df = df[COLUMNS]
    return df



def log_attendance(
    name: str,
    status: str = "IN",
    confidence: float = 0.0,
    csv_path: str = "data/attendance.csv",
) -> None:
    """Log attendance.

    If DATABASE_URL is set, write to Supabase Postgres (attendance_logs).
    Otherwise, append to CSV.

    Args:
        name: recognized name label (string)
        status: "IN" or "OUT"
        confidence: similarity/confidence score (float)
    """
    status = (status or "IN").upper().strip()
    name = (name or "").strip()

    # DB mode
    if is_db_enabled():
        camera_id = os.getenv("CAMERA_ID", "CAM_UNKNOWN")
        camera_label = os.getenv("CAMERA_LABEL", camera_id)
        camera_location = os.getenv("CAMERA_LOCATION", "")

        try:
            ensure_camera(camera_id, label=camera_label, location=camera_location)
        except Exception:
            # If camera upsert fails, still try to insert log (FK may fail if strict)
            pass

        recognized = bool(name) and name.lower() != "unknown"
        emp_id = None
        if recognized:
            try:
                emp_id = get_or_create_employee_id(name)
            except Exception:
                emp_id = None

        event_type = "CHECK_IN" if status == "IN" else "CHECK_OUT"
        similarity = float(confidence) if confidence is not None else 0.0

        insert_attendance_log(
            event_type=event_type,
            camera_id=camera_id,
            recognized=recognized,
            similarity=similarity,
            employee_id=emp_id,
        )
        return

    # CSV fallback
    _ensure_csv(csv_path)

    # Current local time (naive)
    now = datetime.now()
    new_row = {
        "Name": name if name else "Unknown",
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Status": status if status in ("IN", "OUT") else "IN",
        "Confidence": float(confidence) if confidence is not None else 0.0,
    }

    df = read_attendance(csv_path)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)



def export_attendance(
    csv_path: str = "data/attendance.csv",
    output_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    df = read_attendance(csv_path)
    if df.empty:
        print("❌ No attendance data found")
        return

    if start_date:
        df = df[df["Date"] >= start_date]
    if end_date:
        df = df[df["Date"] <= end_date]

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/attendance_export_{timestamp}.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Exported {len(df)} records to {output_path}")


if __name__ == "__main__":
    log_attendance("Test User", "IN", 0.95)
    print("\nToday's Attendance:")
    print(get_today_attendance())
