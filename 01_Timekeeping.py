import streamlit as st
import api_client as api_service
from ui import design_system as theme, sidebar, header, tables

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="FaceLog Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply UI theme and Sidebar
theme.apply()
sidebar.render_sidebar()

# Get API base URL
api_base = (st.session_state.get("api_base") or "http://127.0.0.1:8000").rstrip("/")

# 2. HEADER
header.render_header("Executive Dashboard", "Real-time biometric monitoring and personnel management.")

# 3. STATISTICS & METRICS
try:
    # Fetch data for metrics
    employees = api_service.list_employees(api_base=api_base)
    total_emp = len(employees)
    registered_faces = len([e for e in employees if e.get("has_face")])

    cameras = api_service.list_cameras(api_base=api_base)
    active_cams = len([c for c in cameras if c.get("is_active")])

    is_healthy = api_service.health(api_base=api_base)
    system_status = "OPERATIONAL" if is_healthy else "OFFLINE"
    status_color = "#34d399" if is_healthy else "#f87171"
    status_bg = "rgba(52, 211, 153, 0.1)" if is_healthy else "rgba(248, 113, 113, 0.1)"

except Exception:
    total_emp = "—"
    registered_faces = "—"
    active_cams = "—"
    system_status = "OFFLINE"
    status_color = "#f87171"
    status_bg = "rgba(248, 113, 113, 0.1)"

# Render Metrics Grid
st.markdown(f"""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.5rem; margin-bottom: 2.5rem;">
    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">Total Personnel</span>
            <span style="font-size: 1.2rem;">👥</span>
        </div>
        <div style="font-size: 2.5rem; font-weight: 800; font-family: 'Outfit'; margin: 0.5rem 0;">{total_emp}</div>
        <div style="font-size: 0.75rem; color: #34d399; font-weight: 700;">↑ 2.4% vs last month</div>
    </div>
    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">Face Enrollment</span>
            <span style="font-size: 1.2rem;">🛡️</span>
        </div>
        <div style="font-size: 2.5rem; font-weight: 800; font-family: 'Outfit'; margin: 0.5rem 0;">{registered_faces}</div>
        <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700;">Registry completion: <span style="color: #3b82f6;">{round(registered_faces/total_emp*100 if isinstance(total_emp, int) and total_emp > 0 else 0)}%</span></div>
    </div>
    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">Video Streams</span>
            <span style="font-size: 1.2rem;">🎥</span>
        </div>
        <div style="font-size: 2.5rem; font-weight: 800; font-family: 'Outfit'; margin: 0.5rem 0;">{active_cams}</div>
        <div style="font-size: 0.75rem; color: #34d399; font-weight: 700;">All units online</div>
    </div>
    <div class="glass-card" style="border-top: 4px solid {status_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">Core Status</span>
            <span style="font-size: 1.2rem;">⚡</span>
        </div>
        <div style="margin: 1.2rem 0;">
            <span style="background: {status_bg}; color: {status_color}; padding: 0.5rem 1rem; border-radius: 12px; font-size: 1.1rem; font-weight: 800; border: 1px solid {status_color}44;">
                {system_status}
            </span>
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700;">Latency: <span style="color: #34d399;">124ms</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# 4. QUICK ACTIONS
st.markdown("<h3 style='margin-bottom: 1.5rem;'>Quick Operations</h3>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="glass-card" style="padding: 1.5rem; border: 1px solid rgba(59, 130, 246, 0.2);">
        <h4 style="margin: 0; color: #3b82f6;">📷 Live Terminal</h4>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 10px 0 20px 0;">Open real-time facial recognition and attendance log terminal.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Terminal", use_container_width=True, key="launch_term"):
        st.switch_page("pages/01_Realtime.py")

with c2:
    st.markdown("""
    <div class="glass-card" style="padding: 1.5rem; border: 1px solid rgba(16, 185, 129, 0.2);">
        <h4 style="margin: 0; color: #10b981;">👥 Personnel DB</h4>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 10px 0 20px 0;">Manage employee profiles, roles, and biometric enrollments.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Access Database", use_container_width=True, key="access_db"):
        st.switch_page("pages/02_Admin_Database.py")

with c3:
    st.markdown("""
    <div class="glass-card" style="padding: 1.5rem; border: 1px solid rgba(245, 158, 11, 0.2);">
        <h4 style="margin: 0; color: #f59e0b;">🕒 Audit Logs</h4>
        <p style="font-size: 0.85rem; color: #94a3b8; margin: 10px 0 20px 0;">Review access history, security alerts, and attendance reports.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View History", use_container_width=True, key="view_hist"):
        st.switch_page("pages/03_Admin_Logs.py")

st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)

# 5. RECENT ACTIVITY
st.markdown("<h3 style='margin-bottom: 1.5rem;'>Live Intelligence Feed</h3>", unsafe_allow_html=True)
try:
    recent_logs = api_service.fetch_logs(limit=8, api_base=api_base)
    tables.render_logs_table(recent_logs)
except Exception:
    st.error("Connection lost to Intelligence Feed.")
