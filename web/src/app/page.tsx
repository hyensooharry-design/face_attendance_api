import Link from "next/link";

export default function HomePage() {
  return (
    <main style={{ padding: 24, display: "grid", gap: 12 }}>
      <h1>Face Attendance Web</h1>

      <div style={{ display: "grid", gap: 8 }}>
        <Link href="/attendance">➡️ Attendance (Recognize)</Link>
        <Link href="/employees">➡️ Employees</Link>
        <Link href="/logs">➡️ Logs</Link>
      </div>

      <p style={{ color: "#666" }}>
        MVP: 직원 등록 → 얼굴 등록 → 출석 인식(/recognize) → 로그 확인
      </p>
    </main>
  );
}
