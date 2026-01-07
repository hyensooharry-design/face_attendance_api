"use client";

import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api";

type LogRow = {
  log_id?: number;
  event_time?: string;
  event_type?: string;
  camera_id?: string;
  recognized?: boolean;
  similarity?: number;
  employee_id?: number;
  name?: string | null;
};

export default function LogsPage() {
  const [logs, setLogs] = useState<LogRow[]>([]);
  const [err, setErr] = useState("");

  async function load() {
    const rows = await apiGet<LogRow[]>("/logs?limit=500");
    setLogs(rows);
  }

  useEffect(() => {
    load().catch((e) => setErr(String(e)));
  }, []);

  return (
    <main style={{ padding: 24, display: "grid", gap: 16 }}>
      <h1>Logs</h1>

      {err && <div style={{ color: "red", whiteSpace: "pre-wrap" }}>{err}</div>}

      <section style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8 }}>
        <h3>Recent Logs</h3>
        <button onClick={() => load().catch((e) => setErr(String(e)))}>Refresh</button>

        <div style={{ overflowX: "auto", marginTop: 12 }}>
          <table cellPadding={8} style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th>Time</th>
                <th>Name</th>
                <th>Employee</th>
                <th>Type</th>
                <th>Camera</th>
                <th>Recognized</th>
                <th>Similarity</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((r, idx) => (
                <tr key={r.log_id ?? idx}>
                  <td>{r.event_time ?? ""}</td>
                  <td>{r.name ?? ""}</td>
                  <td>{r.employee_id ?? ""}</td>
                  <td>{r.event_type ?? ""}</td>
                  <td>{r.camera_id ?? ""}</td>
                  <td>{String(r.recognized)}</td>
                  <td>{r.similarity ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
