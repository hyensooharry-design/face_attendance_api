import streamlit as st

import api_client as api_service
from styles import theme
from ui import header, sidebar

st.set_page_config(page_title="Database", page_icon="🗃️", layout="wide")
theme.apply()
sidebar.render_sidebar()

api_base = (st.session_state.get("api_base") or "http://127.0.0.1:8000").rstrip("/")

header.render_header("Database", "Manage employees and face enrollment.")


def _pick_emp_id(emp: dict) -> int:
    value = emp.get("employee_id", emp.get("emp_id", emp.get("id")))
    return int(value) if value is not None else -1


def _has_face(emp: dict) -> bool:
    if "has_face" in emp:
        return bool(emp["has_face"])
    return emp.get("face_id") not in (None, "", 0)


st.session_state.setdefault("pending_delete_emp", None)

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("➕ Add New Employee", use_container_width=True):
        st.session_state["show_add_employee"] = True

with c2:
    query = st.text_input(
        "Search by name or ID...",
        value="",
        placeholder="Search by name or ID...",
    )

if st.session_state.get("show_add_employee"):
    with st.container(border=True):
        st.subheader("Create Employee")
        n1, n2 = st.columns(2)
        with n1:
            new_name = st.text_input("Name", key="new_emp_name")
        with n2:
            new_code = st.text_input("Employee Code (optional)", key="new_emp_code")

        a1, a2 = st.columns(2)
        with a1:
            if st.button("Create", type="primary", use_container_width=True):
                try:
                    if not new_name.strip():
                        st.error("Name is required.")
                    else:
                        api_service.create_employee(
                            new_name.strip(),
                            new_code.strip() or None,
                            api_base=api_base,
                        )
                        st.success("Employee created.")
                        st.session_state["show_add_employee"] = False
                        st.rerun()
                except Exception as exc:
                    st.error(f"Create failed: {exc}")
        with a2:
            if st.button("Cancel", use_container_width=True):
                st.session_state["show_add_employee"] = False
                st.rerun()

st.divider()

try:
    employees = api_service.list_employees(query=query, limit=200, api_base=api_base)
except Exception as exc:
    st.error(f"Data loading error: {exc}")
    employees = []

st.subheader("Employee List")

if not employees:
    st.info("No employees yet.")
else:
    for emp in employees:
        emp_id = _pick_emp_id(emp)
        name = emp.get("name", "")
        code = emp.get("employee_code", "") or "—"

        left, mid, right = st.columns([6, 3, 1])
        with left:
            st.write(f"**{name}**")
            st.caption(f"EMP: {code} | ID: {emp_id}")
        with mid:
            st.write("✅ Face enrolled" if _has_face(emp) else "❌ No face")
        with right:
            if st.button("🗑️", key=f"del_{emp_id}", help="Delete employee"):
                st.session_state["pending_delete_emp"] = {
                    "employee_id": emp_id,
                    "name": name,
                }
                st.rerun()

pending = st.session_state.get("pending_delete_emp")
if pending:
    with st.container(border=True):
        st.warning(f"Delete **{pending['name']}** (ID: {pending['employee_id']})?")
        d1, d2 = st.columns(2)
        with d1:
            if st.button("Confirm Delete", type="primary", use_container_width=True):
                try:
                    emp_id = int(pending["employee_id"])
                    try:
                        api_service.delete_face(emp_id, api_base=api_base)
                    except Exception:
                        pass
                    api_service.delete_employee(emp_id, api_base=api_base)
                    st.success("Deleted.")
                    st.session_state["pending_delete_emp"] = None
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
        with d2:
            if st.button("Cancel", use_container_width=True):
                st.session_state["pending_delete_emp"] = None
                st.rerun()

st.divider()
st.subheader("Face Enrollment")

if employees:
    labels = {
        f"{emp.get('name', 'Employee')} — {emp.get('employee_code') or _pick_emp_id(emp)}": _pick_emp_id(emp)
        for emp in employees
    }
    selected_label = st.selectbox("Employee", list(labels.keys()))
    image_file = st.file_uploader(
        "Face image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear frontal face image. Enrolling again replaces the previous embedding.",
    )

    if st.button("Enroll / Replace Face", type="primary", disabled=image_file is None):
        try:
            image_bytes = image_file.getvalue() if image_file else b""
            if not image_bytes:
                st.error("Please select an image.")
            else:
                api_service.enroll_face(
                    labels[selected_label],
                    image_bytes,
                    api_base=api_base,
                )
                st.success("Face enrollment completed.")
                st.rerun()
        except Exception as exc:
            st.error(f"Face enrollment failed: {exc}")
