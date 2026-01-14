import streamlit as st
import api_client as api_service

from styles import theme
from ui import header, sidebar, tables

st.set_page_config(page_title="Database", page_icon="🗃️", layout="wide")
theme.apply()
sidebar.render_sidebar()

api_base = (st.session_state.get("api_base") or "http://127.0.0.1:8000").rstrip("/")

header.render_header("Database", "Add new employees and register faces.")

# ----------------------------
# helpers
# ----------------------------
def _pick_emp_id(emp: dict) -> int:
    # UI/백엔드 필드가 섞여있을 수 있어서 안전하게
    v = emp.get("employee_id", emp.get("emp_id", emp.get("id")))
    return int(v) if v is not None else -1


def _has_face(emp: dict) -> bool:
    # 백엔드 EmployeeResponse에 has_face가 있을 수 있음
    if "has_face" in emp:
        return bool(emp["has_face"])
    # UI가 face_id 같은 걸 쓰면 그것도 감지
    if emp.get("face_id") not in (None, "", 0):
        return True
    return False


# ----------------------------
# state
# ----------------------------
st.session_state.setdefault("pending_delete_emp", None)

# ----------------------------
# top actions
# ----------------------------
c1, c2 = st.columns([1, 3])
with c1:
    if st.button("➕ Add New Employee", use_container_width=True):
        st.session_state["show_add_employee"] = True

with c2:
    query = st.text_input("Search by name or ID...", value="", placeholder="Search by name or ID...")


# ----------------------------
# add employee modal-ish
# ----------------------------
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
                        api_service.create_employee(new_name.strip(), new_code.strip() or None, api_base=api_base)
                        st.success("Employee created.")
                        st.session_state["show_add_employee"] = False
                        st.rerun()
                except Exception as e:
                    st.error(f"Create failed: {e}")
        with a2:
            if st.button("Cancel", use_container_width=True):
                st.session_state["show_add_employee"] = False
                st.rerun()

st.divider()

# ----------------------------
# load employees
# ----------------------------
try:
    employees = api_service.list_employees(query=query, limit=200, api_base=api_base)
except Exception as e:
    st.error(f"Data loading error: {e}")
    employees = []

# ----------------------------
# render list + delete actions
# ----------------------------
# tables.render_employee_table(...) 같은 게 있다면 그걸 쓰고,
# 없으면 여기서 간단히 그려줌.
st.subheader("Employee List")

if not employees:
    st.info("No employees yet.")
else:
    # 테이블 출력(네가 쓰는 ui/tables.py에 맞춰서 유지)
    # 기존 UI 테이블이 actions 컬럼을 내부에서 처리 못하면 아래 커스텀 리스트 방식이 더 안전함.
    # 여기서는 '간단 리스트 + 버튼' 방식으로 확실히 동작하게 함.

    for emp in employees:
        emp_id = _pick_emp_id(emp)
        name = emp.get("name", "")
        code = emp.get("employee_code", "")

        left, mid, right = st.columns([6, 3, 1])
        with left:
            st.write(f"**{name}**  \nEMP: `{code}`  | ID: `{emp_id}`")
        with mid:
            st.write("✅ Face" if _has_face(emp) else "❌ No Face")
        with right:
            if st.button("🗑️", key=f"del_{emp_id}", help="Delete employee"):
                st.session_state["pending_delete_emp"] = {
                    "employee_id": emp_id,
                    "name": name,
                }
                st.rerun()

# ----------------------------
# confirm delete
# ----------------------------
pending = st.session_state.get("pending_delete_emp")
if pending:
    with st.container(border=True):
        st.warning(f"Delete **{pending['name']}** (ID: {pending['employee_id']}) ?")
        d1, d2 = st.columns(2)

        with d1:
            if st.button("Confirm Delete", type="primary", use_container_width=True):
                try:
                    emp_id = int(pending["employee_id"])

                    # 1) face_embeddings 먼저 제거 (FK 안전)
                    try:
                        api_service.delete_face(emp_id, api_base=api_base)
                    except Exception:
                        # face 없으면 404 뜰 수 있으니 무시
                        pass

                    # 2) employee 제거
                    api_service.delete_employee(emp_id, api_base=api_base)

                    st.success("Deleted.")
                    st.session_state["pending_delete_emp"] = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

        with d2:
            if st.button("Cancel", use_container_width=True):
                st.session_state["pending_delete_emp"] = None
                st.rerun()
