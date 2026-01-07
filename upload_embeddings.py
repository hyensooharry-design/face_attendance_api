import os
import numpy as np
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SERVICE_ROLE_KEY)

face_db = np.load("data/face_db.npy", allow_pickle=True).item()
  # dict: name -> (512,) float32

def vec_to_pgvector_str(v: np.ndarray) -> str:
    # pgvector는 문자열 "[...]" 형태도 잘 받음
    v = v.astype(np.float32).tolist()
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"

for name, emb in face_db.items():
    # 1) employees upsert (employee_code는 임시로 name 사용)
    emp_payload = {
        "employee_code": name,   # 나중에 실제 학번/사번으로 업데이트 가능
        "name": name,
        "is_active": True
    }
    emp_res = supabase.table("employees").upsert(emp_payload, on_conflict="employee_code").execute()

    # employee_id 가져오기 (upsert 반환이 애매할 수 있어서 select로 안전하게)
    emp_row = supabase.table("employees").select("employee_id").eq("employee_code", name).single().execute()
    employee_id = emp_row.data["employee_id"]

    # 2) face_embeddings upsert (PK=employee_id 구조)
    fe_payload = {
        "employee_id": employee_id,
        "embedding_dim": 512,
        "embedding_vec": vec_to_pgvector_str(emb),
    }
    supabase.table("face_embeddings").upsert(fe_payload, on_conflict="employee_id").execute()

print("✅ Upload complete.")
