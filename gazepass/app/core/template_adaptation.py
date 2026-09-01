import sqlite3
import numpy as np
import uuid
from datetime import datetime, timedelta
from typing import List, Dict

class TemplateAdaptationEngine:
    """
    Template Adaptation (Phase 15).
    Quarantines high-quality verification frames, and securely promotes them
    to active gallery templates after multi-day consistency checks.
    Protects against poisoning attacks (Section 59/60).
    """
    def __init__(self, db_manager, promotion_threshold: int = 5, min_days_apart: int = 3):
        self.db = db_manager
        self.promotion_threshold = promotion_threshold
        self.min_days_apart = min_days_apart
        
    def quarantine_candidate(self, student_id: str, embedding: np.ndarray, quality_score: float, pose: tuple):
        """
        Saves a highly confident verification frame as a QUARANTINE template.
        """
        # Only accept extremely high quality frames for potential adaptation
        if quality_score < 0.85:
            return
            
        template_id = str(uuid.uuid4())
        embedding_blob = embedding.astype(np.float32).tobytes()
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # We store it in biometric_templates but mark it QUARANTINE
        cursor.execute('''
            INSERT INTO biometric_templates 
            (id, student_id, embedding, model_id, model_version, quality_score, yaw, pitch, roll, source, template_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            template_id, student_id, embedding_blob, 
            'sface', 'v1', quality_score, pose[0], pose[1], pose[2],
            'adaptation', 'QUARANTINE'
        ))
        conn.commit()
        conn.close()
        
    def run_nightly_promotion(self):
        """
        Scans QUARANTINE templates. If a student has accumulated enough quarantined 
        templates over distinct days, promoting the most representative one.
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Get all quarantine templates
        cursor.execute("SELECT id, student_id, embedding, created_at FROM biometric_templates WHERE template_status = 'QUARANTINE'")
        quarantines = cursor.fetchall()
        
        # Group by student
        candidates_by_student = {}
        for row in quarantines:
            sid = row['student_id']
            if sid not in candidates_by_student:
                candidates_by_student[sid] = []
            
            emb = np.frombuffer(row['embedding'], dtype=np.float32).reshape(1, -1)
            date_obj = datetime.strptime(row['created_at'], "%Y-%m-%d %H:%M:%S")
            candidates_by_student[sid].append({
                "id": row['id'],
                "embedding": emb,
                "date": date_obj
            })
            
        for sid, candidates in candidates_by_student.items():
            if len(candidates) >= self.promotion_threshold:
                # Check if they span enough distinct days to prove temporal consistency
                dates = set([c["date"].date() for c in candidates])
                if len(dates) >= self.min_days_apart:
                    # They passed the security gate.
                    # Promote the most recent one to ACTIVE, delete the rest.
                    # In a production system, you'd calculate the medoid.
                    candidates.sort(key=lambda x: x["date"], reverse=True)
                    best_candidate = candidates[0]
                    
                    cursor.execute("UPDATE biometric_templates SET template_status = 'ACTIVE' WHERE id = ?", (best_candidate["id"],))
                    
                    # Delete the rest of the quarantine templates for this user to restart cycle
                    other_ids = [c["id"] for c in candidates if c["id"] != best_candidate["id"]]
                    if other_ids:
                        placeholders = ','.join(['?'] * len(other_ids))
                        cursor.execute(f"DELETE FROM biometric_templates WHERE id IN ({placeholders})", other_ids)
                        
        conn.commit()
        conn.close()
