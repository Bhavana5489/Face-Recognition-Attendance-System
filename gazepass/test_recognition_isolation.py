import sys
import os
import unittest
import numpy as np

# Insert gazepass path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.attendance_engine import AttendanceEngine, TrackSession, AttendanceDecision
from app.liveness.active_challenge.gaze_estimator import GazeObservation

class DummyDB:
    def get_connection(self):
        class DummyConn:
            def execute(self, *args, **kwargs):
                pass
            def rollback(self):
                pass
            def commit(self):
                pass
            def close(self):
                pass
            def cursor(self):
                class DummyCursor:
                    def execute(self, *args, **kwargs):
                        pass
                    def fetchone(self):
                        return {"name": "Test Student"}
                return DummyCursor()
        return DummyConn()

class TestRecognitionIsolation(unittest.TestCase):
    def setUp(self):
        self.db = DummyDB()
        self.engine = AttendanceEngine(self.db)
        # Ensure dev mode defaults or matches config
        self.engine.T_rec = 0.70
        self.engine.T_margin = 0.05
        self.engine.T_quality = 0.40
        self.engine.T_pad = 0.60
        self.engine.dev_recognition_debug = False

    def test_consensus_identity_high_variance_success(self):
        session = TrackSession(track_id=999)
        # Add identities with high variance but same student_id and high median
        # Previously, variance > 0.05 would fail identity gate
        identities = [
            [{"student_id": "STUDENT_A", "similarity": 0.95}],
            [{"student_id": "STUDENT_A", "similarity": 0.35}], # massive drop (expression change)
            [{"student_id": "STUDENT_A", "similarity": 0.90}],
            [{"student_id": "STUDENT_A", "similarity": 0.85}],
            [{"student_id": "STUDENT_A", "similarity": 0.80}]
        ]
        
        for idx, id_res in enumerate(identities):
            # Compute a second-best match for margin calculation
            # If no second best match is in list, we add a dummy with margin check
            id_res.append({"student_id": "STUDENT_B", "similarity": 0.20})
            session.add_evidence(id_res, quality=0.80, liveness=0.80)
            
        decision, student_id = session.get_consensus_identity(
            min_votes=5,
            cosine_threshold=0.70,
            margin_threshold=0.05
        )
        
        # Verify that it succeeds despite the variance
        self.assertEqual(decision, "MATCH")
        self.assertEqual(student_id, "STUDENT_A")

    def test_dev_recognition_debug_bypass(self):
        self.engine.dev_recognition_debug = True
        
        # We start a process event
        track_id = 123
        gaze_obs = GazeObservation(0.0, 0.0, 0.0, 1.0, "FULL", 0.5, 0.5, 0.0)
        
        # Feed 5 frames of high quality and matching identity
        for i in range(5):
            matches = [
                {"student_id": "STUDENT_A", "similarity": 0.90},
                {"student_id": "STUDENT_B", "similarity": 0.20}
            ]
            decision = self.engine.process_frame_event(
                track_id=track_id,
                identity_results=matches,
                quality_score=0.85,
                liveness_score=0.10, # Passive PAD is very low (should fail if not bypassed)
                gaze_obs=gaze_obs,
                embedding=np.zeros(128)
            )
            
        # On the 5th frame, since min_votes=5 and identity matches, it should succeed directly (bypass liveness/PAD/DB)
        self.assertEqual(decision.attendance_state, "MARKED")
        self.assertEqual(decision.identity_state, "MATCH")
        self.assertEqual(decision.liveness_state, "PASS")

if __name__ == "__main__":
    unittest.main()
