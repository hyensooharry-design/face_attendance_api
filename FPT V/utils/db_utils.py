"""Database utilities for Supabase Postgres.

Design goals:
- No Streamlit dependency (usable from scripts as well).
- Safe defaults for Supabase (sslmode=require).
- Minimal surface area: get connection, upsert employee/camera, insert attendance log, fetch logs.

Environment variables:
- DATABASE_URL: Postgres connection string. Example:
  postgresql://user:pass@host:6543/postgres
- DB_SSLMODE: optional, default 'require'
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.errors import UniqueViolation
except Exception as e:  # pragma: no cover
    psycopg2 = None  # type: ignore


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_database_url() -> Optional[str]:
    return os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DATABASE_URL") or os.getenv("POSTGRES_URL")


def is_db_enabled() -> bool:
    return bool(get_database_url())


@contextmanager
def db_conn():
    """Context manager that yields a psycopg2 connection."""
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Add psycopg2-binary to requirements.txt")

    dsn = get_database_url()
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")

    sslmode = os.getenv("DB_SSLMODE", "require")
    conn = psycopg2.connect(dsn, sslmode=sslmode)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_or_create_employee_id(name: str) -> Optional[int]:
    """Return employee_id for a given name. If missing, create a new employee row.

    Note:
    - Schema expects (employee_id BIGSERIAL PK, employee_code UNIQUE, name, is_active, created_at, updated_at)
      per the provided table structure.
    - Because 'name' is not guaranteed unique, we generate a stable employee_code fallback.
    """
    name = (name or "").strip()
    if not name or name.lower() == "unknown":
        return None

    # Primary approach: find active employee by name
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT employee_id
                FROM employees
                WHERE name = %s AND (is_active IS TRUE OR is_active IS NULL)
                ORDER BY employee_id ASC
                LIMIT 1
                """,
                (name,),
            )
            row = cur.fetchone()
            if row:
                return int(row[0])

            # Insert with employee_code = name if possible, else fallback to hash-based code
            employee_code = name
            try:
                cur.execute(
                    """
                    INSERT INTO employees (employee_code, name, is_active, created_at)
                    VALUES (%s, %s, TRUE, NOW())
                    RETURNING employee_id
                    """,
                    (employee_code, name),
                )
                emp_id = cur.fetchone()[0]
                conn.commit()
                return int(emp_id)
            except Exception:
                conn.rollback()

            # Fallback: hash-based unique code
            digest = __import__("hashlib").sha1(name.encode("utf-8")).hexdigest()[:10]
            employee_code = f"NAME_{digest}"
            cur.execute(
                """
                INSERT INTO employees (employee_code, name, is_active, created_at)
                VALUES (%s, %s, TRUE, NOW())
                ON CONFLICT (employee_code) DO UPDATE SET name = EXCLUDED.name
                RETURNING employee_id
                """,
                (employee_code, name),
            )
            emp_id = cur.fetchone()[0]
            conn.commit()
            return int(emp_id)


def ensure_camera(camera_id: str, label: Optional[str] = None, location: Optional[str] = None) -> None:
    """Ensure a camera exists in cameras table."""
    camera_id = (camera_id or "").strip()
    if not camera_id:
        return
    label = label or camera_id
    location = location or ""

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cameras (camera_id, label, location, direction, is_active, created_at)
                VALUES (%s, %s, %s, 'FRONT', TRUE, NOW())
                ON CONFLICT (camera_id) DO UPDATE SET
                    label = EXCLUDED.label,
                    location = EXCLUDED.location,
                    is_active = TRUE
                """,
                (camera_id, label, location),
            )
            conn.commit()


def insert_attendance_log(
    *,
    event_type: str,
    camera_id: str,
    recognized: bool,
    similarity: float,
    employee_id: Optional[int],
    event_time: Optional[datetime] = None,
) -> None:
    """Insert a row into attendance_logs."""
    event_time = event_time or _now_utc()

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attendance_logs (event_time, event_type, camera_id, recognized, similarity, employee_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """,
                (event_time, event_type, camera_id, recognized, float(similarity), employee_id),
            )
            conn.commit()


def fetch_attendance_logs(limit: int = 200) -> List[Dict[str, Any]]:
    """Fetch recent attendance logs joined with employee name."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    l.event_time,
                    COALESCE(e.name, 'Unknown') AS name,
                    CASE WHEN l.event_type = 'CHECK_IN' THEN 'IN' ELSE 'OUT' END AS status,
                    l.similarity,
                    l.camera_id
                FROM attendance_logs l
                LEFT JOIN employees e ON e.employee_id = l.employee_id
                ORDER BY l.event_time DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            rows = cur.fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        dt, name, status, sim, cam = r
        # Convert to naive local display strings later; keep as ISO UTC here.
        out.append(
            {
                "Date": dt.date().isoformat() if hasattr(dt, "date") else "",
                "Time": dt.time().strftime("%H:%M:%S") if hasattr(dt, "time") else "",
                "Name": name,
                "Status": status,
                "Confidence": float(sim) if sim is not None else 0.0,
                "Camera": cam,
            }
        )
    return out
