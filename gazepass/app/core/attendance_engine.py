import time
import numpy as np
from datetime import datetime, date
from typing import Dict, Any, List, Tuple, Optional
import uuid
import yaml
import os

from app.liveness.active_challenge.gaze_estimator import GazeObservation
from app.liveness.active_challenge.challenge_engine import ActiveChallengeEngine, ChallengeState
from app.core.template_adaptation import TemplateAdaptationEngine

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except:
        # Fallbacks (calibrated loosely based on Phase 2 benchmark)
        return {
            'recognition': {'cosine_threshold': 0.70, 'margin_threshold': 0.05, 'temporal_window': 5},
            'quality': {'min_quality': 0.40},
            'pad': {'passive_threshold': 0.5, 'temporal_threshold': 0.6, 'pass_fraction': 0.6}
        }

class AttendanceDecision:
    def __init__(self, 
                 attendance_state: str, 
                 challenge_command: Optional[str] = None, 
                 identity_state: str = "UNKNOWN", 
                 liveness_state: str = "PENDING", 
                 student_name: Optional[str] = None,
                 failure_reason: Optional[str] = None):
        self.attendance_state = attendance_state       # "TRACKED", "QUALITY_OK", "RECOGNITION_PENDING", "LIVENESS_PENDING", "VERIFIED", "MARKED", "RETRY", "FAILED", "ALREADY_MARKED"
        self.challenge_command = challenge_command     # "LEFT", "RIGHT", "UP", "DOWN", "BLINK", "CENTER", None
        self.identity_state = identity_state           # "UNKNOWN", "IDENTIFIED", "MATCH", "UNCERTAIN"
        self.liveness_state = liveness_state           # "PENDING", "PASS", "FAIL", "INCONCLUSIVE", "SPOOF"
        self.student_name = student_name
        self.failure_reason = failure_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.attendance_state,
            "gaze": self.challenge_command or "N/A",
            "recognized": self.student_name or "Unknown",
            "identity_state": self.identity_state,
            "liveness_state": self.liveness_state,
            "failure_reason": self.failure_reason
        }

class TrackSession:
    """Maintains state for a single tracked individual over time (Phase 5)."""
    def __init__(self, track_id: int, num_challenges: int = 3, step_timeout: float = 3.5):
        self.track_id = track_id
        self.identities = []          # List of (student_id, similarity, margin)
        self.quality_scores = []
        self.liveness_scores = []
        self.start_time = time.time()
        
        self.challenge_engine = ActiveChallengeEngine(num_challenges=num_challenges, step_timeout=step_timeout)
        self.challenge_engine.generate_session(str(track_id))
        self.attendance_marked = False
        self.student_name = None
        
    def add_evidence(self, identity_results: list, quality: float, liveness: float):
        self.quality_scores.append(quality)
        self.liveness_scores.append(liveness)
        
        if identity_results:
            top1 = identity_results[0]
            
            # Find the best match for a DIFFERENT identity to compute margin
            second_best_sim = 0.0
            for res in identity_results[1:]:
                if res["student_id"] != top1["student_id"]:
                    second_best_sim = res["similarity"]
                    break
                    
            margin = top1["similarity"] - second_best_sim
            self.identities.append((top1["student_id"], top1["similarity"], margin))
            
    def get_consensus_identity(self, min_votes: int = 5, cosine_threshold: float = 0.70, margin_threshold: float = 0.05) -> Tuple[str, Optional[str]]:
        """Returns the robust statistical aggregation of identity (Phase 5)."""
        if len(self.identities) == 0:
            return "UNKNOWN", None
            
        if len(self.identities) < min_votes:
            return "RETRY", None
            
        votes = {}
        # Only look at the most recent temporal window (e.g., last 15 frames) to allow fast recovery
        recent_window = max(min_votes, 15)
        recent_identities = self.identities[-recent_window:]
        
        for uid, sim, margin in recent_identities:
            if uid not in votes:
                votes[uid] = {"sims": [], "margins": []}
            votes[uid]["sims"].append(sim)
            votes[uid]["margins"].append(margin)
            
        if not votes:
            return "UNKNOWN", None
            
        best_id = max(votes, key=lambda k: len(votes[k]["sims"]))
        
        if len(votes[best_id]["sims"]) < (min_votes * 0.6): # 60% consensus
            return "RETRY", best_id
            
        median_sim = float(np.median(votes[best_id]["sims"]))
        median_margin = float(np.median(votes[best_id]["margins"]))
        
        if median_sim < cosine_threshold:
            return "UNKNOWN", best_id
            
        if median_margin < margin_threshold:
            return "UNCERTAIN", best_id
            
        return "MATCH", best_id
        
    def get_temporal_pad(self, min_frames: int = 5, threshold: float = 0.6) -> str:
        """Returns PASS, FAIL, or INCONCLUSIVE for Temporal PAD (Phase 6)."""
        if len(self.liveness_scores) < min_frames:
            return "INCONCLUSIVE"
            
        recent_pad = self.liveness_scores[-min_frames:]
        recent_quality = self.quality_scores[-min_frames:]
        
        if sum(recent_quality) == 0:
            return "INCONCLUSIVE"
            
        # Quality-weighted average
        weighted_pad = np.sum(np.array(recent_pad) * np.array(recent_quality)) / np.sum(recent_quality)
        
        # If overall quality over the window is very poor, it's inconclusive
        if np.mean(recent_quality) < 0.4:
            return "INCONCLUSIVE"
            
        if weighted_pad >= threshold:
            return "PASS"
        else:
            return "FAIL"

class AttendanceEngine:
    def __init__(self, db_manager):
        self.db = db_manager
        self.config = load_config()
        self.sessions: Dict[int, TrackSession] = {}
        self.adaptation_engine = TemplateAdaptationEngine(self.db)
        
        # Versioned calibration thresholds
        self.threshold_version = self.config.get('threshold_version', 'v0_dev')
        self.T_quality = self.config.get('quality', {}).get('min_quality')
        if self.T_quality is None: self.T_quality = 0.40
        self.T_rec = self.config.get('recognition', {}).get('cosine_threshold')
        if self.T_rec is None: self.T_rec = 0.70
        self.T_margin = self.config.get('recognition', {}).get('margin_threshold')
        if self.T_margin is None: self.T_margin = 0.05
        self.T_pad = self.config.get('pad', {}).get('temporal_threshold')
        if self.T_pad is None: self.T_pad = 0.60
        self.T_temporal = self.config.get('recognition', {}).get('temporal_threshold')
        if self.T_temporal is None: self.T_temporal = 0.60
        self.dev_recognition_debug = self.config.get('dev_mode', {}).get('recognition_debug', False)
        
    def process_frame_event(self, track_id: int, identity_results: list, quality_score: float, liveness_score: float, 
                            gaze_obs: GazeObservation, embedding: np.ndarray = None) -> AttendanceDecision:
        decision = self._process_frame_event_inner(track_id, identity_results, quality_score, liveness_score, gaze_obs, embedding)
        if track_id in self.sessions:
            self.sessions[track_id].latest_decision = decision
        return decision

    def _get_challenge_command(self, active_sess) -> str:
        curr_idx = active_sess["current_index"]
        cmd = "CENTER"
        if active_sess["state"] not in [ChallengeState.WAITING, ChallengeState.BASELINE_ACQUISITION]:
            if curr_idx < len(active_sess["sequence"]):
                cmd = active_sess["sequence"][curr_idx].command
        return cmd

    def _process_frame_event_inner(self, track_id: int, identity_results: list, quality_score: float, liveness_score: float, 
                             gaze_obs: GazeObservation, embedding: np.ndarray = None) -> AttendanceDecision:
        if track_id not in self.sessions:
            num_challenges = self.config.get('liveness', {}).get('num_challenges', 3)
            step_timeout = self.config.get('liveness', {}).get('challenge_timeout_seconds', 3.5)
            self.sessions[track_id] = TrackSession(track_id, num_challenges=num_challenges, step_timeout=step_timeout)
            
        session = self.sessions[track_id]
        
        if session.attendance_marked:
            name = session.student_name or "Student"
            return AttendanceDecision(
                attendance_state="ALREADY_MARKED",
                identity_state="MATCH",
                liveness_state="PASS",
                student_name=name
            )

        # 0. UPDATE ACTIVE LIVENESS CHALLENGE UNCONDITIONALLY
        challenge_state = session.challenge_engine.verify_observation(gaze_obs, quality_score)
        active_sess = session.challenge_engine.active_session
        cmd = self._get_challenge_command(active_sess)
            
        # 1. QUALITY CHECK
        quality_pass = quality_score >= self.T_quality
        if not quality_pass:
            return AttendanceDecision(
                attendance_state="RETRY",
                challenge_command=cmd,
                identity_state="UNKNOWN",
                liveness_state="PENDING",
                failure_reason="Face quality below threshold"
            )
            
        # Add evidence
        session.add_evidence(identity_results, quality_score, liveness_score)
        
        # 2. IDENTITY CONSISTENCY CHECK
        window = self.config.get('recognition', {}).get('temporal_window', 5)
        identity_decision, student_id = session.get_consensus_identity(
            min_votes=window,
            cosine_threshold=self.T_rec,
            margin_threshold=self.T_margin
        )
        
        student_name = None
        if student_id:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM students WHERE id = ?", (student_id,))
            row = cursor.fetchone()
            if row:
                student_name = row['name']
            conn.close()
            session.student_name = student_name
            
        if identity_decision == "RETRY":
            return AttendanceDecision(
                attendance_state="RECOGNITION_PENDING",
                challenge_command=cmd,
                identity_state="UNKNOWN",
                liveness_state="PENDING",
                student_name=student_name
            )
        elif identity_decision == "UNKNOWN":
            TERMINAL_STATES = {"COMPLETED", "FAILED", "TIMEOUT", "INCONCLUSIVE"}
            if challenge_state.upper() not in TERMINAL_STATES:
                return AttendanceDecision(
                    attendance_state="RECOGNITION_PENDING",
                    challenge_command=cmd,
                    identity_state="UNKNOWN",
                    liveness_state="PENDING",
                    student_name=student_name,
                    failure_reason="Recognition in progress"
                )
            return AttendanceDecision(
                attendance_state="FAILED",
                challenge_command=cmd,
                identity_state="UNKNOWN",
                liveness_state="PENDING",
                student_name=student_name,
                failure_reason="Unknown face identity"
            )
        elif identity_decision == "UNCERTAIN":
            TERMINAL_STATES = {"COMPLETED", "FAILED", "TIMEOUT", "INCONCLUSIVE"}
            if challenge_state.upper() not in TERMINAL_STATES:
                return AttendanceDecision(
                    attendance_state="RECOGNITION_PENDING",
                    challenge_command=cmd,
                    identity_state="UNCERTAIN",
                    liveness_state="PENDING",
                    student_name=student_name,
                    failure_reason="Recognition in progress"
                )
            return AttendanceDecision(
                attendance_state="RETRY",
                challenge_command=cmd,
                identity_state="UNCERTAIN",
                liveness_state="PENDING",
                student_name=student_name,
                failure_reason="Identity margin too low"
            )

        # Bypass PAD + liveness + DB write if in dev debug mode
        if self.dev_recognition_debug:
            if identity_decision == "MATCH":
                session.attendance_marked = True
                return AttendanceDecision(
                    attendance_state="MARKED",
                    challenge_command=cmd,
                    identity_state="MATCH",
                    liveness_state="PASS",
                    student_name=student_name
                )
            
        # 3. PASSIVE PAD CHECK
        pad_decision = session.get_temporal_pad(
            min_frames=window,
            threshold=self.T_pad
        )
        
        if pad_decision == "INCONCLUSIVE":
            return AttendanceDecision(
                attendance_state="LIVENESS_PENDING",
                challenge_command=cmd,
                identity_state="IDENTIFIED",
                liveness_state="PENDING",
                student_name=student_name
            )
        elif pad_decision == "FAIL":
            return AttendanceDecision(
                attendance_state="FAILED",
                challenge_command=cmd,
                identity_state="IDENTIFIED",
                liveness_state="SPOOF",
                student_name=student_name,
                failure_reason="Passive PAD failed"
            )
            
        # 4. ACTIVE LIVENESS STATUS EVALUATION (Once Quality, Identity, and Passive PAD pass)
        TERMINAL_STATES = {"COMPLETED", "FAILED", "TIMEOUT", "INCONCLUSIVE"}
        is_terminal = challenge_state.upper() in TERMINAL_STATES
        
        if is_terminal:
            challenge_res = session.challenge_engine.get_challenge_result()
            
            if challenge_res.decision == "PASS":
                # Fall through to atomic transaction below
                pass
            elif challenge_res.decision == "INCONCLUSIVE":
                return AttendanceDecision(
                    attendance_state="INCONCLUSIVE",
                    challenge_command=cmd,
                    identity_state="MATCH",
                    liveness_state="INCONCLUSIVE",
                    student_name=student_name,
                    failure_reason=challenge_res.failure_reason
                )
            else:  # FAIL
                return AttendanceDecision(
                    attendance_state="FAILED",
                    challenge_command=cmd,
                    identity_state="MATCH",
                    liveness_state="FAIL",
                    student_name=student_name,
                    failure_reason=challenge_res.failure_reason or "Challenge failed"
                )
        else:
            # Challenge still in progress
            return AttendanceDecision(
                attendance_state="LIVENESS_PENDING",
                challenge_command=cmd,
                identity_state="IDENTIFIED",
                liveness_state="PENDING",
                student_name=student_name
            )
            
        # 5. ATOMIC ATTENDANCE TRANSACTION (VERIFIED / ELIGIBLE state reached)
        conn = self.db.get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            today = date.today().isoformat()
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM attendance_records WHERE student_id = ? AND attendance_date = ?", (student_id, today))
            already_exists = cursor.fetchone() is not None
            
            if already_exists:
                conn.rollback()
                session.attendance_marked = True
                return AttendanceDecision(
                    attendance_state="ALREADY_MARKED",
                    identity_state="MATCH",
                    liveness_state="PASS",
                    student_name=student_name
                )
                
            event_id = str(uuid.uuid4())
            best_sim = float(np.median([s for uid, s, m in session.identities if uid == student_id]))
            best_margin = float(np.median([m for uid, s, m in session.identities if uid == student_id]))
            median_q = float(np.median(session.quality_scores))
            median_l = float(np.median(session.liveness_scores))
            
            cursor.execute('''
                INSERT INTO recognition_events 
                (id, track_id, candidate_student_id, top1_similarity, margin, quality_score, liveness_score, decision, threshold_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (event_id, str(track_id), student_id, best_sim, best_margin, median_q, median_l, "MATCH", self.threshold_version))
            
            cursor.execute('''
                INSERT INTO attendance_records (id, student_id, recognition_event_id, attendance_date)
                VALUES (?, ?, ?, ?)
            ''', (str(uuid.uuid4()), student_id, event_id, today))
            
            cursor.execute('''
                INSERT INTO liveness_events 
                (id, recognition_event_id, challenge_type, total_steps, completed_steps, gaze_score, challenge_score, decision, threshold_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), event_id, "gaze_trajectory", challenge_res.total_steps, challenge_res.completed_steps, challenge_res.gaze_score, challenge_res.challenge_score, "PASS", self.threshold_version))
            
            conn.commit()
            session.attendance_marked = True
            
            # Template Adaptation
            if embedding is not None and quality_score > 0.85:
                self.adaptation_engine.quarantine_candidate(
                    student_id, embedding, quality_score, (gaze_obs.yaw, gaze_obs.pitch, 0.0)
                )
                
            return AttendanceDecision(
                attendance_state="MARKED",
                identity_state="MATCH",
                liveness_state="PASS",
                student_name=student_name
            )
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
        
    def cleanup_lost_tracks(self, active_track_ids: List[int]):
        lost_ids = [tid for tid in self.sessions if tid not in active_track_ids]
        for tid in lost_ids:
            del self.sessions[tid]
