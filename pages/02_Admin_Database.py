import streamlit as st
import api_client as api_service
import pandas as pd
from ui import design_system, header, sidebar, tables, overlays
import cv2

st.set_page_config(page_title="Employee Database", page_icon="👥", layout="wide")
design_system.apply()
sidebar.render_sidebar()

header.render_header("Personnel Intelligence", "Manage secure database, biometric records, and system access.")

# --- FILTERS & ACTIONS ---
with st.container():
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1], gap="medium")
    with c1:
        search_q = st.text_input("🔍 Search Personnel", placeholder="Name or ID...")
    with c2:
        status_filter = st.selectbox("Registry Status", ["All Personnel", "Active Registry", "Unregistered"])
    with c3:
        st.write("") # Padding
        if st.button("➕ Add New", use_container_width=True):
            st.session_state.show_add_modal = True
    with c4:
        st.write("") # Padding
        if st.button("📊 Export CSV", use_container_width=True):
            try:
                all_emp = api_service.list_employees()
                df = pd.DataFrame(all_emp)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Report",
                    data=csv,
                    file_name="personnel_registry.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error("Export Failed")

# --- DATA TABLE ---
try:
    with st.spinner("Syncing Global Registry..."):
        employees = api_service.list_employees(query=search_q)
        
        # Apply Registry Status Filter
        if status_filter == "Active Registry":
            employees = [e for e in employees if e.get("has_face")]
        elif status_filter == "Unregistered":
            employees = [e for e in employees if not e.get("has_face")]

    tables.render_employee_table(employees)
except Exception as e:
    st.error(f"Registry Synchronizer Error: {e}")

# --- MODALS: ADD PERSONNEL ---
if st.session_state.get("show_add_modal"):
    with st.container():
        st.markdown('<div class="glass-card" style="border: 1px solid rgba(59, 130, 246, 0.3);">', unsafe_allow_html=True)
        st.markdown("### ➕ Register New Personnel")
        
        c_fields1, c_fields2 = st.columns(2)
        with c_fields1:
            new_name = st.text_input("Full Display Name", placeholder="e.g. John Doe")
            new_code = st.text_input("Personnel Code", placeholder="e.g. EMP-101")
        with c_fields2:
            new_role = st.selectbox("Designation", ["Worker", "Team Leader", "Manager"], index=0)
            st.caption("Access level depends on role selection.")

        st.markdown("---")
        st.caption("Biometric Enrollment")
        img_file = st.camera_input("Scanner Link active", label_visibility="collapsed")
        
        c_act1, c_act2 = st.columns(2)
        if c_act1.button("Cancel Registry", use_container_width=True):
            st.session_state.show_add_modal = False
            st.rerun()
            
        if c_act2.button("Begin Enrollment", use_container_width=True, type="primary"):
            if not new_name:
                st.toast("Full Name is mandatory", icon="⚠️")
            else:
                try:
                    with st.spinner("Encrypting and Syncing..."):
                        # 1. Create Employee
                        emp = api_service.create_employee(name=new_name, employee_code=new_code, role=new_role)
                        eid = emp.get("employee_id")
                        
                        # 2. Enroll Face if image provided
                        if img_file and eid:
                            api_service.enroll_face(eid, img_file.getvalue())
                        
                        st.success(f"Successfully registered {new_name}")
                        st.balloons()
                        st.session_state.show_add_modal = False
                        st.rerun()
                except Exception as e:
                    st.error(f"Registry Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- MODALS: MODIFY PERSONNEL ---
if st.session_state.get("show_edit_modal") and st.session_state.get("employee_to_edit"):
    emp = st.session_state.employee_to_edit
    with st.container():
        st.markdown('<div class="glass-card" style="border: 1px solid rgba(168, 85, 247, 0.3);">', unsafe_allow_html=True)
        st.markdown(f"### 📝 Modify Record: {emp['name']}")
        
        c_e1, c_e2 = st.columns(2)
        with c_e1:
            edit_name = st.text_input("Full Name", value=emp["name"])
            edit_code = st.text_input("Personnel Code", value=emp.get("employee_code") or "")
        with c_e2:
            current_role = emp.get("role", "Worker")
            roles_list = ["Worker", "Team Leader", "Manager"]
            role_idx = roles_list.index(current_role) if current_role in roles_list else 0
            edit_role = st.selectbox("Update Designation", roles_list, index=role_idx)

        c_edit_act1, c_edit_act2 = st.columns(2)
        if c_edit_act1.button("Abort Changes", key="edit_cancel", use_container_width=True):
            st.session_state.show_edit_modal = False
            st.session_state.employee_to_edit = None
            st.rerun()
            
        if c_edit_act2.button("Commit Updates", key="edit_save", use_container_width=True, type="primary"):
            try:
                with st.spinner("Updating Central DB..."):
                    api_service.update_employee(emp["employee_id"], name=edit_name, employee_code=edit_code, role=edit_role)
                    st.toast("Profile Synchronized", icon="✅")
                    st.session_state.show_edit_modal = False
                    st.session_state.employee_to_edit = None
                    st.rerun()
            except Exception as e:
                st.error(f"Sync Failure: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- MODALS: DECOMMISSION PERSONNEL ---
if st.session_state.get("show_delete_confirm") and st.session_state.get("employee_to_delete"):
    emp = st.session_state.employee_to_delete
    with st.container():
        st.markdown('<div class="glass-card" style="border: 1px solid rgba(248, 113, 113, 0.3); background: rgba(248, 113, 113, 0.05);">', unsafe_allow_html=True)
        st.markdown(f"### ⚠️ Decommission Personnel: {emp['name']}")
        st.warning(f"Are you certain you want to remove this record? This action is irreversible and will purge all biometric hashes.")
        
        c_del_1, c_del_2 = st.columns(2)
        if c_del_1.button("Cancel Purge", key="del_cancel", use_container_width=True):
            st.session_state.show_delete_confirm = False
            st.session_state.employee_to_delete = None
            st.rerun()
            
        if c_del_2.button("Execute Decommission", key="del_confirm", use_container_width=True, type="primary"):
            try:
                with st.spinner("Purging data..."):
                    api_service.delete_employee(emp["employee_id"])
                    st.toast("Record Extinguished", icon="🗑️")
                    st.session_state.show_delete_confirm = False
                    st.session_state.employee_to_delete = None
                    st.rerun()
            except Exception as e:
                st.error(f"Purge Failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
