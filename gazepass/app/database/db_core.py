import sqlite3
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any, Optional
import uuid
from datetime import datetime

class DatabaseManager:
    """
    Core SQLite database manager for Gazepass 2026.
    Replaces JSON file storage with ACID-compliant relational schema.
    """
    def __init__(self, db_path: str = "gazepass.db"):
        self.db_path = db_path
        self._initialize_schema()
        
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
        
    def _initialize_schema(self):
        """Initializes the required tables outlined in the specification."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 1. Students Table (Section 54)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                admission_number TEXT UNIQUE,
                name TEXT NOT NULL,
                class_id TEXT,
                status TEXT DEFAULT 'ACTIVE',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 2. Biometric Templates Table (Section 55)
        # embedding is stored as BLOB (numpy float32 array bytes)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS biometric_templates (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                embedding_dimension INTEGER DEFAULT 128,
                alignment_version TEXT,
                quality_score REAL,
                yaw REAL,
                pitch REAL,
                roll REAL,
                source TEXT,
                cluster_id TEXT,
                template_status TEXT DEFAULT 'ACTIVE',
                capture_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        ''')
        
        # 3. Recognition Events (Section 57)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_events (
                id TEXT PRIMARY KEY,
                camera_id TEXT,
                track_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                candidate_student_id TEXT,
                top1_similarity REAL,
                top2_similarity REAL,
                margin REAL,
                quality_score REAL,
                liveness_score REAL,
                temporal_consistency REAL,
                decision TEXT,
                model_version TEXT,
                threshold_version TEXT,
                latency_ms REAL
            )
        ''')
        
        # 4. Attendance Records
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_records (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                recognition_event_id TEXT,
                attendance_date DATE NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id),
                FOREIGN KEY (recognition_event_id) REFERENCES recognition_events(id),
                UNIQUE(student_id, attendance_date) -- Enforces Idempotency (Section 52)
            )
        ''')

        # 5. Liveness Events Table (V2)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS liveness_events (
                id TEXT PRIMARY KEY,
                track_id TEXT,
                liveness_session_id TEXT,
                recognition_session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                camera_id TEXT,
                face_quality_score REAL,
                passive_pad_v1_score REAL,
                passive_pad_v2_score REAL,
                temporal_pad_score REAL,
                gaze_score REAL,
                blink_score REAL,
                head_pose_score REAL,
                challenge_score REAL,
                challenge_sequence TEXT,
                challenge_step_count INTEGER,
                challenge_completed_steps INTEGER,
                liveness_mode TEXT,
                passive_pad_decision TEXT,
                active_liveness_decision TEXT,
                final_liveness_decision TEXT,
                model_version TEXT,
                threshold_version TEXT
            )
        ''')

        # 6. Template Update Candidates Table (V2)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS template_update_candidates (
                id TEXT PRIMARY KEY,
                student_id TEXT NOT NULL,
                recognition_event_id TEXT,
                embedding BLOB NOT NULL,
                quality_score REAL,
                yaw REAL,
                pitch REAL,
                roll REAL,
                promotion_status TEXT DEFAULT 'ACTIVE_STAGED',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id),
                FOREIGN KEY (recognition_event_id) REFERENCES recognition_events(id)
            )
        ''')
        
        # 7. Camera Gaze Calibrations Table (V2)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS camera_gaze_calibrations (
                camera_id TEXT PRIMARY KEY,
                calibration_version TEXT,
                calibration_status TEXT DEFAULT 'UNCALIBRATED',
                a1 REAL,
                a2 REAL,
                a3 REAL,
                b1 REAL,
                b2 REAL,
                b3 REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Migration helper: add missing columns if upgrading schema
        def _add_column_if_not_exists(table_name, column_name, column_type):
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            if column_name not in columns:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

        _add_column_if_not_exists("recognition_events", "recognition_session_id", "TEXT")
        _add_column_if_not_exists("recognition_events", "identity_decision", "TEXT")
        _add_column_if_not_exists("recognition_events", "template_consistency", "REAL")

        conn.commit()
        conn.close()
        
    def add_student(self, name: str, admission_number: str = None) -> str:
        """Registers a new student and returns their UUID."""
        student_id = str(uuid.uuid4())
        if admission_number is None:
            admission_number = f"ADM-{np.random.randint(10000, 99999)}"
            
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (id, admission_number, name) VALUES (?, ?, ?)",
            (student_id, admission_number, name)
        )
        conn.commit()
        conn.close()
        return student_id
        
    def save_template(self, student_id: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Saves a biometric template generated during enrollment."""
        template_id = str(uuid.uuid4())
        # Convert numpy array to bytes for SQLite BLOB storage
        embedding_blob = embedding.astype(np.float32).tobytes()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO biometric_templates 
            (id, student_id, embedding, model_id, model_version, quality_score, yaw, pitch, roll, source, template_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            template_id, student_id, embedding_blob, 
            metadata.get('model_id', 'sface'),
            metadata.get('model_version', 'v1'),
            metadata.get('quality_score', 1.0),
            metadata.get('yaw', 0.0),
            metadata.get('pitch', 0.0),
            metadata.get('roll', 0.0),
            metadata.get('source', 'enrollment'),
            'ACTIVE'
        ))
        conn.commit()
        conn.close()

    def save_template_candidate(self, student_id: str, recognition_event_id: str, embedding: np.ndarray, quality_score: float, yaw: float, pitch: float, roll: float):
        """Stages a candidate template in the quarantine zone."""
        candidate_id = str(uuid.uuid4())
        embedding_blob = embedding.astype(np.float32).tobytes()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO template_update_candidates
            (id, student_id, recognition_event_id, embedding, quality_score, yaw, pitch, roll, promotion_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE_STAGED')
        ''', (candidate_id, student_id, recognition_event_id, embedding_blob, quality_score, yaw, pitch, roll))
        conn.commit()
        conn.close()

    def get_staged_candidates(self) -> List[Dict[str, Any]]:
        """Retrieves all staged candidate templates."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, student_id, recognition_event_id, embedding, quality_score, yaw, pitch, roll, promotion_status, created_at FROM template_update_candidates")
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "student_id": r["student_id"],
                "recognition_event_id": r["recognition_event_id"],
                "embedding": np.frombuffer(r["embedding"], dtype=np.float32),
                "quality_score": r["quality_score"],
                "yaw": r["yaw"],
                "pitch": r["pitch"],
                "roll": r["roll"],
                "promotion_status": r["promotion_status"],
                "created_at": r["created_at"]
            })
        return results

    def update_candidate_status(self, candidate_id: str, status: str):
        """Updates promotion_status for a template update candidate."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE template_update_candidates SET promotion_status = ? WHERE id = ?",
            (status, candidate_id)
        )
        conn.commit()
        conn.close()

    def log_liveness_event(self, event_data: Dict[str, Any]):
        """Logs a liveness assessment audit record."""
        event_id = str(uuid.uuid4())
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO liveness_events (
                id, track_id, liveness_session_id, recognition_session_id,
                camera_id, face_quality_score, passive_pad_v1_score, passive_pad_v2_score,
                temporal_pad_score, gaze_score, blink_score, head_pose_score, challenge_score,
                challenge_sequence, challenge_step_count, challenge_completed_steps,
                liveness_mode, passive_pad_decision, active_liveness_decision,
                final_liveness_decision, model_version, threshold_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_id,
            event_data.get("track_id"),
            event_data.get("liveness_session_id"),
            event_data.get("recognition_session_id"),
            event_data.get("camera_id"),
            event_data.get("face_quality_score"),
            event_data.get("passive_pad_v1_score"),
            event_data.get("passive_pad_v2_score"),
            event_data.get("temporal_pad_score"),
            event_data.get("gaze_score"),
            event_data.get("blink_score"),
            event_data.get("head_pose_score"),
            event_data.get("challenge_score"),
            json.dumps(event_data.get("challenge_sequence", [])),
            event_data.get("challenge_step_count"),
            event_data.get("challenge_completed_steps"),
            event_data.get("liveness_mode"),
            event_data.get("passive_pad_decision"),
            event_data.get("active_liveness_decision"),
            event_data.get("final_liveness_decision"),
            event_data.get("model_version"),
            event_data.get("threshold_version")
        ))
        conn.commit()
        conn.close()

    def get_all_active_templates(self) -> Tuple[List[str], np.ndarray]:
        """
        Retrieves all active templates across all students.
        Returns a tuple of (List of student_ids, numpy array of embeddings Nx128).
        This enables extremely fast in-memory exact matching.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, embedding FROM biometric_templates WHERE template_status = 'ACTIVE'")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return [], np.empty((0, 128), dtype=np.float32)
            
        student_ids = []
        embeddings = []
        
        for row in rows:
            student_ids.append(row['student_id'])
            # Reconstruct numpy array from BLOB
            emb = np.frombuffer(row['embedding'], dtype=np.float32).reshape(1, -1)
            embeddings.append(emb)
            
        return student_ids, np.vstack(embeddings)

    def get_gaze_calibration(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the gaze calibration coefficients for the specified camera.
        Returns a dictionary or None if not calibrated.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT camera_id, calibration_version, calibration_status, a1, a2, a3, b1, b2, b3
            FROM camera_gaze_calibrations WHERE camera_id = ?
        ''', (camera_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None

    def save_gaze_calibration(self, camera_id: str, coeffs: Dict[str, Any]):
        """
        Saves the gaze calibration coefficients for the specified camera.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO camera_gaze_calibrations
            (camera_id, calibration_version, calibration_status, a1, a2, a3, b1, b2, b3)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(camera_id) DO UPDATE SET
                calibration_version=excluded.calibration_version,
                calibration_status=excluded.calibration_status,
                a1=excluded.a1, a2=excluded.a2, a3=excluded.a3,
                b1=excluded.b1, b2=excluded.b2, b3=excluded.b3,
                created_at=CURRENT_TIMESTAMP
        ''', (
            camera_id,
            coeffs.get("calibration_version", "1.0"),
            coeffs.get("calibration_status", "CALIBRATED"),
            coeffs.get("a1", 1.0),
            coeffs.get("a2", 0.0),
            coeffs.get("a3", 0.0),
            coeffs.get("b1", 1.0),
            coeffs.get("b2", 0.0),
            coeffs.get("b3", 0.0)
        ))
        conn.commit()
        conn.close()

class VectorSearchEngine:
    """
    Implements fast Top-K Cosine Similarity search over SQLite templates in memory.
    Replaces pgvector per Phase 8 instructions.
    """
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.student_ids = []
        self.gallery = np.empty((0, 128), dtype=np.float32)
        self.reload_gallery()
        
    def reload_gallery(self):
        """Loads all active embeddings into contiguous memory for fast matrix multiplication."""
        self.student_ids, self.gallery = self.db.get_all_active_templates()
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Finds the closest matches using exact cosine similarity.
        query_embedding must be L2 normalized (done by SFace wrapper).
        """
        if len(self.gallery) == 0:
            return []
            
        # Matrix multiplication against normalized gallery
        # Score ranges from -1 to 1 (1 is perfect match)
        scores = np.dot(self.gallery, query_embedding.T).flatten()
        
        # Get Top-K indices sorted descending
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "student_id": self.student_ids[idx],
                "similarity": float(scores[idx])
            })
            
        return results
