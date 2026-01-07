"use client";

import { useEffect, useState } from "react";
import { apiGet, apiPostFile } from "@/lib/api";

type Employee = {
  employee_id: number;
  employee_code?: string | null;
  name: string;
  is_active: boolean;
  has_face?: boolean | null;
};

export default function EmployeeDetailPage({ params }: { params: { id: string } }) {
  const employeeId = Number(params.id);

  const [emp, setEmp] = useState<Employee | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  async function load() {
    const data = await apiGet<Employee>(`/employees/${employeeId}`);
    setEmp(data);
  }

  useEffect(() => {
    if (!Number.isFinite(employeeId)) {
      setErr("Invalid employee id");
      return;
    }
    load().catch((e) => setErr(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [employeeId]);

  async function onEnroll() {
    if (!file) return;
    setErr("");
    setMsg("");
    try {
      await apiPostFile(`/employees/${employeeId}/faces`, file);
      setMsg("✅ Face enrolled.");
      setFile(null);
      await load();
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    }
  }

  return (
    <main style={{ padding: 24, display: "grid", gap: 16 }}>
      <h1>Employee Detail</h1>

      {err && <div style={{ color: "red", whiteSpace: "pre-wrap" }}>{err}</div>}
      {msg && <div style={{ color: "green" }}>{msg}</div>}

      <section style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8 }}>
        <h3>Info</h3>
        {!emp ? (
          <div>Loading...</div>
        ) : (
          <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(emp, null, 2)}</pre>
        )}
      </section>

      <section style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8, display: "grid", gap: 8 }}>
        <h3>Enroll Face</h3>
        <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <button onClick={onEnroll} disabled={!file}>
          Upload & Enroll
        </button>
        <p style={{ color: "#666" }}>
          직원 얼굴 사진을 업로드하면 서버에서 임베딩을 생성해서 face_embeddings에 저장합니다.
        </p>
      </section>
    </main>
  );
}
