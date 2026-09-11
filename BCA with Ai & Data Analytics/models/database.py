"""AgriSmart shared SQLite database layer."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_NAME = str(BASE_DIR / "agrismart.db")
MODEL_DIR = BASE_DIR / "models"
TOKEN_DAYS = 30


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return f"pbkdf2_sha256$310000${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    if not stored:
        return False
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, rounds, salt_hex, digest_hex = stored.split("$", 3)
            digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds))
            return hmac.compare_digest(digest.hex(), digest_hex)
        except (ValueError, TypeError):
            return False
    # Backward compatibility for the old plaintext database.
    return hmac.compare_digest(stored, password)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def init_db() -> None:
    conn = get_db_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                state TEXT,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'User',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                inputs TEXT NOT NULL,
                result TEXT NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                timestamp TEXT,
                module TEXT,
                inputs TEXT,
                result TEXT
            );

            CREATE TABLE IF NOT EXISTS auth_tokens (
                token_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                state TEXT,
                expires_at TEXT NOT NULL
            );
            """
        )
        cols = _columns(conn, "users")
        if "email" in cols:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(lower(email))")
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            conn.execute(
                "INSERT INTO users(username,email,state,password,role) VALUES(?,?,?,?,?)",
                ("admin", "admin@agrismart.local", "Delhi", hash_password("admin123"), "Admin"),
            )
            conn.execute(
                "INSERT INTO users(username,email,state,password,role) VALUES(?,?,?,?,?)",
                ("user", "user@agrismart.local", "Delhi", hash_password("user123"), "User"),
            )
        conn.commit()
    finally:
        conn.close()


def verify_user(username: str, password: str) -> Optional[str]:
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT password, role FROM users WHERE lower(username)=lower(?) OR lower(email)=lower(?) LIMIT 1",
            (username, username),
        ).fetchone()
        if row and verify_password(password, row["password"]):
            return row["role"]
        return None
    finally:
        conn.close()


def register_user(username: str, password: str, role: str = "User", email: Optional[str] = None, state: Optional[str] = None) -> bool:
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO users(username,email,state,password,role) VALUES(?,?,?,?,?)",
            (username.strip(), email.strip().lower() if email else None, state, hash_password(password), role),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def ensure_user_columns() -> None:
    # Kept for compatibility with older project versions.
    init_db()


def create_login_token(username: str, role: str, state: str = "Delhi") -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.now() + timedelta(days=TOKEN_DAYS)
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO auth_tokens(token_hash,username,role,state,expires_at) VALUES(?,?,?,?,?)",
            (hashlib.sha256(token.encode()).hexdigest(), username, role, state, expires.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def get_login_from_token(token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT * FROM auth_tokens WHERE token_hash=?", (_token_hash(token),)).fetchone()
        if not row:
            return None
        if datetime.fromisoformat(row["expires_at"]) <= datetime.now():
            conn.execute("DELETE FROM auth_tokens WHERE token_hash=?", (_token_hash(token),))
            conn.commit()
            return None
        return dict(row)
    finally:
        conn.close()


def revoke_login_token(token: str) -> None:
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM auth_tokens WHERE token_hash=?", (_token_hash(token),))
        conn.commit()
    finally:
        conn.close()


def log_history(username: str, action: str, inputs: Dict[str, Any], result: str) -> int:
    payload = json.dumps(inputs, ensure_ascii=False, default=str)
    conn = get_db_connection()
    try:
        cur = conn.execute("INSERT INTO history(username,action,inputs,result) VALUES(?,?,?,?)", (username, action, payload, result))
        conn.execute("INSERT INTO predictions(username,timestamp,module,inputs,result) VALUES(?,?,?,?,?)", (username, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, payload, result))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_all_history(username: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        if username:
            rows = conn.execute("SELECT * FROM history WHERE username=? ORDER BY id DESC", (username,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try: item["inputs"] = json.loads(item["inputs"])
            except Exception: pass
            result.append(item)
        return result
    finally:
        conn.close()


def get_history_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT * FROM history WHERE id=?", (record_id,)).fetchone()
        if not row: return None
        item = dict(row)
        try: item["inputs"] = json.loads(item["inputs"])
        except Exception: pass
        return item
    finally:
        conn.close()


def update_history_record(record_id: int, action: Optional[str] = None, inputs: Optional[Dict[str, Any]] = None, result: Optional[str] = None) -> bool:
    updates, params = [], []
    if action is not None: updates.append("action=?"); params.append(action)
    if inputs is not None: updates.append("inputs=?"); params.append(json.dumps(inputs, ensure_ascii=False, default=str))
    if result is not None: updates.append("result=?"); params.append(result)
    if not updates: return False
    params.append(record_id)
    conn = get_db_connection()
    try:
        cur = conn.execute(f"UPDATE history SET {', '.join(updates)} WHERE id=?", params)
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def remove_history(record_id: int) -> bool:
    conn = get_db_connection()
    try:
        cur = conn.execute("DELETE FROM history WHERE id=?", (record_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def execute_admin_sql(query: str):
    """Execute SQL for trusted local administration. Prefer read-only SELECTs."""
    conn = get_db_connection()
    try:
        cur = conn.execute(query)
        if query.lstrip().lower().startswith("select"):
            return [dict(row) for row in cur.fetchall()]
        conn.commit()
        return {"rows_affected": cur.rowcount}
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        conn.close()


def get_ml_metadata() -> Optional[Dict[str, Any]]:
    path = MODEL_DIR / "ml_metadata.json"
    if not path.exists(): return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_ml_model_status() -> Dict[str, Any]:
    names = {
        "recommendation_model": "rf_recommendation.joblib",
        "yield_model": "rf_yield.joblib",
        "yield_features": "yield_features.joblib",
        "selling_model": "rf_selling.joblib",
        "selling_features": "selling_features.joblib",
        "metadata": "ml_metadata.json",
    }
    status = {key: (MODEL_DIR / filename).exists() for key, filename in names.items()}
    status["all_models_ready"] = all(status.values())
    return status


def get_ml_information() -> Dict[str, Any]:
    return {"status": "success", "models_ready": get_ml_model_status()["all_models_ready"], "model_status": get_ml_model_status(), "data": get_ml_metadata()}


init_db()
ensure_user_columns()
