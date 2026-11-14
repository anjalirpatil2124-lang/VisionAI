import sqlite3
import hashlib
import secrets
import hmac
from datetime import datetime

DATABASE_NAME = "visionai.db"


def get_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE COLLATE NOCASE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            total_objects INTEGER,
            average_confidence REAL,
            created_at TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detected_objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            object_name TEXT,
            confidence REAL,
            FOREIGN KEY (run_id) REFERENCES detection_runs(id)
        )
    """)
    cursor.execute("PRAGMA table_info(detection_runs)")
    columns = [column[1] for column in cursor.fetchall()]
    if "user_id" not in columns:
        cursor.execute("""
            ALTER TABLE detection_runs ADD COLUMN user_id INTEGER
            REFERENCES users(id)
        """)
    conn.commit()
    conn.close()


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 200000
    )
    return password_hash.hex(), salt


def create_user(username, password):
    initialize_database()
    username = username.strip()
    if len(username) < 3:
        return False, "Username must contain at least 3 characters."
    if len(username) > 30:
        return False, "Username cannot exceed 30 characters."
    if not username.replace("_", "").isalnum():
        return False, "Use only letters, numbers and underscores."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."
    password_hash, salt = hash_password(password)
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (username, password_hash, salt, created_at)
            VALUES (?, ?, ?, ?)
        """, (username, password_hash, salt,
              datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "This username is already registered."
    finally:
        conn.close()


def authenticate_user(username, password):
    initialize_database()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, username, password_hash, salt FROM users WHERE username = ?
    """, (username.strip(),))
    user = cursor.fetchone()
    conn.close()
    if user is None:
        return None
    user_id, stored_username, stored_hash, salt = user
    entered_hash, _ = hash_password(password, salt)
    if hmac.compare_digest(entered_hash, stored_hash):
        return {"id": user_id, "username": stored_username}
    return None


def get_user_by_id(user_id):
    initialize_database()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, created_at FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user


def save_detection(source, objects, user_id=None):
    initialize_database()
    conn = get_connection()
    cursor = conn.cursor()
    total_objects = len(objects)
    confidence_values = [obj["confidence"] for obj in objects]
    average_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO detection_runs
        (source, total_objects, average_confidence, created_at, user_id)
        VALUES (?, ?, ?, ?, ?)
    """, (source, total_objects, average_confidence, created_at, user_id))
    run_id = cursor.lastrowid
    for obj in objects:
        cursor.execute("""
            INSERT INTO detected_objects (run_id, object_name, confidence)
            VALUES (?, ?, ?)
        """, (run_id, obj["name"], obj["confidence"]))
    conn.commit()
    conn.close()


def get_detection_history(user_id=None):
    initialize_database()
    conn = get_connection()
    cursor = conn.cursor()
    if user_id is None:
        cursor.execute("""
            SELECT id, source, total_objects, average_confidence, created_at
            FROM detection_runs ORDER BY id DESC
        """)
    else:
        cursor.execute("""
            SELECT id, source, total_objects, average_confidence, created_at
            FROM detection_runs WHERE user_id = ? ORDER BY id DESC
        """, (user_id,))
    records = cursor.fetchall()
    conn.close()
    return records


def get_detection_details(run_id, user_id=None):
    initialize_database()
    conn = get_connection()
    cursor = conn.cursor()
    if user_id is None:
        cursor.execute("SELECT object_name, confidence FROM detected_objects WHERE run_id = ?", (run_id,))
    else:
        cursor.execute("""
            SELECT detected_objects.object_name, detected_objects.confidence
            FROM detected_objects
            JOIN detection_runs ON detected_objects.run_id = detection_runs.id
            WHERE detected_objects.run_id = ? AND detection_runs.user_id = ?
        """, (run_id, user_id))
    records = cursor.fetchall()
    conn.close()
    return records


def get_object_statistics(user_id=None):
    initialize_database()
    conn = get_connection()
    cursor = conn.cursor()
    if user_id is None:
        cursor.execute("""
            SELECT object_name, COUNT(*) FROM detected_objects
            GROUP BY object_name ORDER BY COUNT(*) DESC
        """)
    else:
        cursor.execute("""
            SELECT detected_objects.object_name, COUNT(*)
            FROM detected_objects JOIN detection_runs
            ON detected_objects.run_id = detection_runs.id
            WHERE detection_runs.user_id = ?
            GROUP BY detected_objects.object_name ORDER BY COUNT(*) DESC
        """, (user_id,))
    records = cursor.fetchall()
    conn.close()
    return records


def get_database_analytics(user_id=None):
    initialize_database()
    conn = get_connection()
    cursor = conn.cursor()
    if user_id is None:
        cursor.execute("SELECT COUNT(*), COALESCE(SUM(total_objects), 0) FROM detection_runs")
        total_sessions, total_objects = cursor.fetchone()
        cursor.execute("SELECT source, COUNT(*), SUM(total_objects) FROM detection_runs GROUP BY source")
        source_data = cursor.fetchall()
        cursor.execute("""
            SELECT object_name, COUNT(*) FROM detected_objects
            GROUP BY object_name ORDER BY COUNT(*) DESC
        """)
        object_data = cursor.fetchall()
        cursor.execute("SELECT AVG(confidence) FROM detected_objects")
        average_confidence = cursor.fetchone()[0] or 0
    else:
        cursor.execute("""
            SELECT COUNT(*), COALESCE(SUM(total_objects), 0)
            FROM detection_runs WHERE user_id = ?
        """, (user_id,))
        total_sessions, total_objects = cursor.fetchone()
        cursor.execute("""
            SELECT source, COUNT(*), SUM(total_objects) FROM detection_runs
            WHERE user_id = ? GROUP BY source
        """, (user_id,))
        source_data = cursor.fetchall()
        cursor.execute("""
            SELECT detected_objects.object_name, COUNT(*)
            FROM detected_objects JOIN detection_runs
            ON detected_objects.run_id = detection_runs.id
            WHERE detection_runs.user_id = ?
            GROUP BY detected_objects.object_name
            ORDER BY COUNT(*) DESC
        """, (user_id,))
        object_data = cursor.fetchall()
        cursor.execute("""
            SELECT AVG(detected_objects.confidence)
            FROM detected_objects JOIN detection_runs
            ON detected_objects.run_id = detection_runs.id
            WHERE detection_runs.user_id = ?
        """, (user_id,))
        average_confidence = cursor.fetchone()[0] or 0
    conn.close()
    return {
        "total_sessions": total_sessions,
        "total_objects": total_objects,
        "source_data": source_data,
        "object_data": object_data,
        "average_confidence": round(average_confidence, 2)
    }


initialize_database()
