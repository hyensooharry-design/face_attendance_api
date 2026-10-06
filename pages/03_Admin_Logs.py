from datetime import date

import pandas as pd
import streamlit as st

import api_client as api_service
from styles import theme
from ui import header, sidebar, tables

st.set_page_config(page_title="History", page_icon="🕒", layout="wide")
theme.apply()
sidebar.render_sidebar()
api_base = (st.session_state.get("api_base") or "http://127.0.0.1:8000").rstrip("/")

header.render_header("Access Logs", "Monitor system access history.")

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        limit = st.selectbox("Number of rows", [20, 50, 100, 200], index=1)
    with c2:
        status_filter = st.selectbox("Status", ["All", "Success", "Failed"])
    with c3:
        date_filter = st.date_input("Date", value=None)

try:
    logs = api_service.fetch_logs(limit=limit, api_base=api_base)

    if status_filter == "Success":
        logs = [row for row in logs if row.get("recognized")]
    elif status_filter == "Failed":
        logs = [row for row in logs if not row.get("recognized")]

    if isinstance(date_filter, date):
        filtered = []
        for row in logs:
            raw = row.get("event_time")
            if not raw:
                continue
            parsed = pd.to_datetime(raw, errors="coerce", utc=True)
            if pd.notna(parsed) and parsed.date() == date_filter:
                filtered.append(row)
        logs = filtered

except Exception as exc:
    st.error(f"Data loading error: {exc}")
    logs = []

tables.render_logs_table(logs)
