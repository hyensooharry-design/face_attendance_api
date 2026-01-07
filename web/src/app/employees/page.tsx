"use client";

import { useEffect, useState } from "react";
import { apiGet, apiPostFile } from "@/lib/api";

type Camera = {
  camera_id: string;
  label?: string | null;
  location?: string | null;
};

type RecognizeResp = {
  recognized: boolean;
  employee_id: number | null;
  similarity: number | null;
  event_type?: string;
  camera_id?: string;
  event_time?: string;
  name?: string;
};

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

export default function AttendancePage() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [cameraId, setCameraId] = useState<string>("");
  const [mode, setMode] = useState<"CHECK_IN" | "CHECK_OUT">("CHECK_IN");
  const [file, setFile] = useState<File | null>(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RecognizeResp | null>(null);
  const [err, setErr] = useState<string>("");

  const [logs, setLogs] = useState<LogRow[]>([]);

  async function refreshAll() {
    const cams = await apiGet<Camera[]>("/cameras");
    setCameras(cams);
    if (!cameraId && cams.length) setCameraId(cams[0].camera_id);

    const rows = await apiGet<LogRow[]>("/logs?limit=200");
    setLogs(rows);
  }

  useEffect(() => {
    refreshAll().catch((e) => setErr(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onRecognize() {
    if (!file || !cameraId) return;

    setLoading(true);
    setErr("");
    setResult(null);

    try {
      const resp = await apiPostFile<RecognizeResp>("/recognize", file, {
        camera_id: cameraId,
        event_type: mode,
      });

      setResult(resp);

      const rows = await apiGet<LogRow[]>("/logs?limit=200");
      setLogs(rows);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ padding: 24, display: "grid", gap: 16 }}>
      <h1>Attendance</h1>

      {err && <div style={{ color: "red", whiteSpace: "pre-wrap" }}>{err}</div>}

      <section style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
        <label>
          Camera:&nbsp;
          <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
            <option value="" disabled>
              Select camera
            </option>
            {cameras.map((c) => (
              <option key={c.camera_id} value={c.camera_id}>
                {c.camera_id} {c.label ? `(${c.label})` : ""}
              </option>
            ))}
          </select>
        </label>

        <label>
          Mode:&nbsp;
          <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
            <option value="CHECK_IN">CHECK_IN</option>
            <option value="CHECK_OUT">CHECK_OUT</option>
          </select>
        </label>

        <input
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />

        <button onClick={onRecognize} disabled={!file || !cameraId || loading}>
          {loading ? "Recognizing..." : "Recognize & Log"}
        </button>

        <button onClick={() => refreshAll().catch((e) => setErr(String(e)))}>
          Refresh
        </button>
      </section>

      <section style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8 }}>
        <h3>Result</h3>
        {!result ? (
          <div style={{ color: "#666" }}>No result yet.</div>
        ) : (
          <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(result, null, 2)}</pre>
        )}
      </section>

      <section style={{ border: "1px solid #ddd", padding: 12, borderRadius: 8 }}>
        <h3>Recent Logs</h3>
        <div style={{ overflowX: "auto" }}>
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
