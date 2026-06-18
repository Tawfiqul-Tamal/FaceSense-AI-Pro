import os
import sqlite3
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_path="database/facesense.db"):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._initialize_schema()

    def _initialize_schema(self):
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                emotion TEXT,
                age TEXT,
                gender TEXT,
                attention TEXT,
                stress TEXT,
                head_pose TEXT,
                face_count INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                gender TEXT,
                registered_on TEXT
            )
            """
        )
        self.connection.commit()
        self._migrate_schema()

    def _migrate_schema(self):
        cursor = self.connection.cursor()
        cursor.execute("PRAGMA table_info(sessions)")
        columns = [row[1] for row in cursor.fetchall()]
        if "head_pose" not in columns:
            cursor.execute("ALTER TABLE sessions ADD COLUMN head_pose TEXT")
            self.connection.commit()

    def insert_session(self, summary):
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO sessions (timestamp, emotion, age, gender, attention, stress, head_pose, face_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(),
                summary.get("emotion"),
                summary.get("age"),
                summary.get("gender"),
                summary.get("attention"),
                summary.get("stress"),
                summary.get("head_pose"),
                summary.get("face_count", 0),
            ),
        )
        self.connection.commit()

    def fetch_history(self, limit=50):
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT timestamp, emotion, age, gender, attention, stress, head_pose, face_count FROM sessions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cursor.fetchall()

    def fetch_statistics(self):
        cursor = self.connection.cursor()

        cursor.execute("SELECT COUNT(*), AVG(face_count), AVG(CAST(age AS INTEGER)) FROM sessions")
        count_row = cursor.fetchone() or (0, 0, None)
        total_sessions, avg_faces, avg_age = count_row

        cursor.execute("SELECT emotion, COUNT(*) FROM sessions GROUP BY emotion")
        emotion_counts = {row[0] or "Unknown": row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT gender, COUNT(*) FROM sessions GROUP BY gender")
        gender_counts = {row[0] or "Unknown": row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT head_pose, COUNT(*) FROM sessions GROUP BY head_pose")
        head_pose_counts = {row[0] or "Unknown": row[1] for row in cursor.fetchall()}

        return {
            "total_sessions": total_sessions or 0,
            "average_faces": round(avg_faces or 0, 2),
            "average_age": int(avg_age) if avg_age is not None else 0,
            "emotion_counts": emotion_counts,
            "gender_counts": gender_counts,
            "head_pose_counts": head_pose_counts,
        }

    def close(self):
        self.connection.close()
