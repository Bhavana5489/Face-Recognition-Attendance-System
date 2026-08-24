import os
import sys
import time
import cv2
import numpy as np
import threading
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS

# Add Gazepass root to path to import the new backend
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAZEPASS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'gazepass'))
sys.path.insert(0, GAZEPASS_DIR)

from app.database.db_core import DatabaseManager, VectorSearchEngine
from app.vision.detector.yunet import YuNetDetector
from app.vision.tracker.bytetrack import ByteTracker
from app.vision.landmarks.facex import FaceXLandmarker
from app.vision.pose.head_pose import HeadPoseEstimator
from app.vision.quality.engine import FaceQualityEngine
from app.vision.recognition.sface import SFaceRecognizer
from app.liveness.passive_pad.minifasnet import MiniFASNetEnsemble
from app.liveness.active_challenge.challenge_engine import ActiveChallengeEngine, ChallengeState
from app.liveness.active_challenge.gaze_estimator import FallbackGazeEstimator, FaceXGazeEstimator
from app.core.attendance_engine import AttendanceEngine
from app.enrollment.video_parser import VideoEnrollmentParser

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app)

# ── Gazepass Enterprise Backend Initialization ──────────────────────────
print("[*] Initializing Gazepass 2026 Enterprise Backend...")

MODELS_DIR = os.path.join(GAZEPASS_DIR, 'models')
DB_PATH = os.path.join(GAZEPASS_DIR, 'gazepass.db')

db_manager = DatabaseManager(DB_PATH)
vector_engine = VectorSearchEngine(db_manager)

detector = YuNetDetector(os.path.join(MODELS_DIR, 'detector', 'face_detection_yunet_2023mar.onnx'))
sface = SFaceRecognizer(os.path.join(MODELS_DIR, 'recognition', 'face_recognition_sface_2021dec.onnx'))

# Fallback paths if FaceX/MiniFASNet aren't strictly downloaded yet
facex_path = os.path.join(MODELS_DIR, 'landmarks', 'facex_landmark.onnx')
minifas_v1_path = os.path.join(MODELS_DIR, 'liveness', 'minifasnet_v1se.onnx')
minifas_v2_path = os.path.join(MODELS_DIR, 'liveness', 'minifasnet_v2.onnx')

landmarker = FaceXLandmarker(facex_path)
pad_ensemble = MiniFASNetEnsemble(minifas_v1_path, minifas_v2_path)

quality_engine = FaceQualityEngine()
tracker = ByteTracker()
attendance_engine = AttendanceEngine(db_manager)
challenge_engine = ActiveChallengeEngine()
fallback_gaze_estimator = FallbackGazeEstimator(db_manager)
facex_gaze_estimator = FaceXGazeEstimator(db_manager)

# Global State
current_mode = "idle" # idle, register, mark
registration_parser = None
registration_student_id = None
shared_status = {"message": "Idle", "student_name": "Unknown"}
latest_metrics = {}

# ── Camera Threading ────────────────────────────────────────────────
class CameraStream:
    def __init__(self):
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.clients = 0
        
    def acquire(self):
        with self.lock:
            self.clients += 1
            if not self.running:
                self.start()
                
    def release(self):
        with self.lock:
            self.clients -= 1
            if self.clients <= 0:
                self.clients = 0
                self.stop()
                
    def start(self):
        if self.running: return
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()
        
    def _update(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)
                
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

cam_stream = CameraStream()

import threading
import queue

class InferenceWorker:
    """Runs heavy AI models on a background thread to prevent UI blocking (Phase 10)."""
    def __init__(self):
        self.q = queue.Queue(maxsize=2)
        self.results = {}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def _run(self):
        while True:
            item = self.q.get()
            if item is None: continue
            frame, bbox, best_det, track_id, conf = item
            
            try:
                # 1. FaceX (Dense Landmarks)
                dense_landmarks = landmarker.get_landmarks(frame, bbox)
                
                # 2. Quality
                yaw, pitch, roll = 0.0, 0.0, 0.0
                pose_estimator = HeadPoseEstimator((frame.shape[1], frame.shape[0]))
                if dense_landmarks is not None and len(dense_landmarks) == 98:
                    yaw, pitch, roll = pose_estimator.estimate_facex(dense_landmarks)
                else:
                    yaw, pitch, roll = pose_estimator.estimate(best_det.landmarks)
                    
                quality_eval = quality_engine.evaluate(frame, bbox, (yaw, pitch, roll), conf)
                
                # 3. PAD
                score_v1, score_v2, fusion_score = pad_ensemble.evaluate_detailed(frame, bbox)
                
                # 4. SFace
                aligned_face = sface.align(frame, best_det)
                emb = sface.get_embedding(aligned_face)
                matches = vector_engine.search(emb, top_k=5)
                
                self.results[track_id] = {
                    "dense_landmarks": dense_landmarks,
                    "yaw": yaw, "pitch": pitch, "roll": roll,
                    "quality_eval": quality_eval,
                    "liveness_score": fusion_score,
                    "passive_pad_v1_score": score_v1,
                    "passive_pad_v2_score": score_v2,
                    "matches": matches,
                    "emb": emb,
                    "timestamp": time.time()
                }
            except Exception as e:
                print(f"[Worker Error] {e}")
            finally:
                self.q.task_done()
                
    def push(self, frame, bbox, best_det, track_id, conf):
        if self.q.qsize() < 2:
            self.q.put((frame, bbox, best_det, track_id, conf))
            
    def get_result(self, track_id):
        return self.results.get(track_id)

inference_worker = InferenceWorker()

# ── Legacy-Proven GazeSession (ported from app_legacy.py) ───────────────
DOT_RADIUS = 18
GAZE_MARGIN = 0.05
RED_COLOR = (30, 30, 220)
GREEN_COLOR = (30, 210, 30)

def _get_pupil_x_ratio(eye_roi_bgr):
    """Find horizontal pupil ratio (0=left, 1=right) in an eye crop."""
    if eye_roi_bgr is None or eye_roi_bgr.size == 0:
        return None
    gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    margin_x = int(w * 0.25)
    margin_y = int(h * 0.30)
    crop = gray[margin_y:h-margin_y, margin_x:w-margin_x]
    if crop.size == 0:
        return None
    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    _, _, min_loc, _ = cv2.minMaxLoc(crop)
    cx = min_loc[0] + margin_x
    return float(cx / w)

def _get_pupil_x_ratio_average(frame, face_bbox, yunet_landmarks=None):
    """Average pupil X-ratio across both eyes using YuNet landmarks."""
    if yunet_landmarks is not None and len(yunet_landmarks) >= 2:
        # YuNet 5-point: [right_eye, left_eye, nose, right_mouth, left_mouth]
        right_eye = np.array(yunet_landmarks[0])
        left_eye  = np.array(yunet_landmarks[1])
        eye_dist = float(np.linalg.norm(left_eye - right_eye))
        if eye_dist > 0:
            crop_w = int(eye_dist * 0.35)
            crop_h = int(eye_dist * 0.25)
            h_img, w_img = frame.shape[:2]
            ratios = []
            for pt in [right_eye, left_eye]:
                ex, ey = int(pt[0]), int(pt[1])
                ex_min = max(0, ex - crop_w)
                ex_max = min(w_img - 1, ex + crop_w)
                ey_min = max(0, ey - crop_h)
                ey_max = min(h_img - 1, ey + crop_h)
                eye_roi = frame[ey_min:ey_max, ex_min:ex_max]
                if eye_roi.size > 0:
                    r = _get_pupil_x_ratio(eye_roi)
                    if r is not None:
                        ratios.append(r)
            if ratios:
                return float(np.mean(ratios))
    # Fallback: use face bbox + Haar eye cascade
    fx, fy, fw, fh = face_bbox
    face_roi = frame[fy:fy+fh, fx:fx+fw]
    if face_roi.size == 0:
        return None
    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))
    if len(eyes) == 0:
        return None
    ratios = []
    for (ex, ey, ew, eh) in eyes[:2]:
        eye_roi = face_roi[ey:ey+int(eh*0.65), ex:ex+ew]
        r = _get_pupil_x_ratio(eye_roi)
        if r is not None:
            ratios.append(r)
    return float(np.mean(ratios)) if ratios else None

def _generate_waypoints(frame_w, frame_h):
    """Generate left and right waypoints at random Y positions on the screen extremes."""
    margin_x = 60
    margin_y = 80
    zones = [
        (margin_x, frame_w // 3, margin_y, frame_h - margin_y, "left"),
        (frame_w * 2 // 3, frame_w - margin_x, margin_y, frame_h - margin_y, "right"),
    ]
    import random
    random.shuffle(zones)
    waypoints = []
    for (xmin, xmax, ymin, ymax, label) in zones:
        x = random.randint(int(xmin), int(xmax))
        y = random.randint(int(ymin), int(ymax))
        waypoints.append((x, y, label))
    return waypoints

class GazeSession:
    """Manages the red-dot gaze liveness challenge (ported from app_legacy.py)."""
    def __init__(self, frame_w, frame_h):
        import uuid
        self.recognition_session_id = str(uuid.uuid4())
        self.liveness_session_id = str(uuid.uuid4())
        self.waypoints = _generate_waypoints(frame_w, frame_h)
        self.wp_idx = 0
        self.wp_start = time.time()
        self.wp_duration = 1.5  # seconds per dot
        self.gaze_samples = []
        self.results = []
        self.dot_pos = (self.waypoints[0][0], self.waypoints[0][1])
        self.dot_alpha = 1.0
        self.state = "running"  # "running", "success", "failed"

    def update(self, frame, face_bbox, yunet_landmarks=None):
        """Call each frame. Returns True when challenge is finished."""
        now = time.time()
        elapsed = now - self.wp_start
        wp = self.waypoints[self.wp_idx]
        target_x, target_y, zone_label = wp
        self.dot_pos = (target_x, target_y)
        self.dot_alpha = 0.7 + 0.3 * abs(np.sin(now * 6))

        # Accumulate pupil samples after 0.5s reaction window
        if elapsed >= 0.50 and face_bbox is not None:
            r = _get_pupil_x_ratio_average(frame, face_bbox, yunet_landmarks)
            if r is not None:
                self.gaze_samples.append(r)

        if elapsed >= self.wp_duration:
            stop_avg = float(np.mean(self.gaze_samples)) if self.gaze_samples else None
            self.results.append((zone_label, stop_avg))
            self.wp_idx += 1
            self.wp_start = now
            self.gaze_samples = []

            if self.wp_idx < len(self.waypoints):
                nxt = self.waypoints[self.wp_idx]
                self.dot_pos = (nxt[0], nxt[1])
            else:
                # Evaluate results
                left_ratio  = next((v for z, v in self.results if z == "left"),  None)
                right_ratio = next((v for z, v in self.results if z == "right"), None)
                passed = False
                if left_ratio is not None and right_ratio is not None:
                    diff = right_ratio - left_ratio
                    print(f"[GazeSession] Relative shift: {diff:.4f} (L={left_ratio:.4f}, R={right_ratio:.4f}) -> {'PASS' if diff >= 0.01 else 'FAIL'}")
                    if diff >= 0.01:
                        passed = True
                else:
                    # Fallback: absolute thresholds
                    print(f"[GazeSession] Fallback check: L={left_ratio}, R={right_ratio}")
                    if left_ratio is not None and left_ratio < 0.49:
                        passed = True
                    elif right_ratio is not None and right_ratio > 0.51:
                        passed = True
                self.state = "success" if passed else "failed"
                return True  # Done
        return False  # Still running

# ── Core Video Pipeline (Phase 2 & 14) ─────────────────────────────────
def process_frame(frame):
    global shared_status, current_mode, registration_parser, registration_student_id, latest_metrics
    
    # Pipeline Timings
    t_start = time.time()
    timings = {}
    
    # 1. Detection
    t0 = time.time()
    detections = detector.detect(frame)
    timings['Detection'] = (time.time() - t0) * 1000
    
    # Format for ByteTrack
    track_dets = [(d.bbox, d.confidence) for d in detections]
    
    # 2. Tracking
    t0 = time.time()
    active_tracks = tracker.update(track_dets)
    timings['Tracking'] = (time.time() - t0) * 1000
    active_track_ids = [t.track_id for t in active_tracks]
    
    # Cleanup lost tracks in attendance engine
    attendance_engine.cleanup_lost_tracks(active_track_ids)
    
    for track in active_tracks:
        x, y, w, h = track.bbox
        
        # We need the original YuNet 5-point landmarks for SFace alignment
        # Map tracker bbox back to original detection to grab landmarks
        best_iou = 0
        best_det = None
        for d in detections:
            dx, dy, dw, dh = d.bbox
            iou = tracker._iou(track.bbox, (dx, dy, dw, dh))
            if iou > best_iou:
                best_iou = iou
                best_det = d
                
        if not best_det: continue
        
        if current_mode == "mark":
            tid = track.track_id
            h_frame, w_frame = frame.shape[:2]

            # ── Phase 1: Identity recognition via gazepass backend ──────────
            inference_worker.push(frame.copy(), track.bbox, best_det, tid, track.confidence)
            res = inference_worker.get_result(tid)

            if not res:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 128, 128), 2)
                cv2.putText(frame, "ANALYZING...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                continue

            yaw, pitch, roll = res["yaw"], res["pitch"], res["roll"]
            quality_eval  = res["quality_eval"]
            liveness_score = res["liveness_score"]
            matches        = res["matches"]
            emb            = res["emb"]

            # Update latest metrics for API exposure
            latest_metrics = {
                "quality_score": float(quality_eval.get("quality_score", 0.0)),
                "passive_pad_v1": float(res.get("passive_pad_v1_score", 0.0)),
                "passive_pad_v2": float(res.get("passive_pad_v2_score", 0.0)),
                "passive_pad_fusion": float(liveness_score),
                "top1_similarity": float(matches[0].get("similarity", 0.0)) if matches else 0.0,
                "margin": float(matches[0].get("similarity", 0.0) - matches[1].get("similarity", 0.0)) if len(matches) > 1 else (float(matches[0].get("similarity", 0.0)) if matches else 0.0)
            }

            # Estimate gaze and blinking using correct estimator
            if res.get("dense_landmarks") is not None:
                obs = facex_gaze_estimator.estimate(frame, best_det, res.get("dense_landmarks"), (yaw, pitch, roll))
            else:
                obs = fallback_gaze_estimator.estimate(frame, best_det, None, (yaw, pitch, roll))

            # Run verification step through AttendanceEngine (delegated policy)
            decision = attendance_engine.process_frame_event(
                tid, matches, float(quality_eval.get("quality_score", 0.0)), float(liveness_score), obs, emb
            )

            # Bounding box color and display text
            is_recognized = decision.identity_state in ["MATCH", "IDENTIFIED", "UNCERTAIN"]
            if is_recognized:
                box_color = GREEN_COLOR
                display_text = decision.student_name or "Student"
                shared_status["student_name"] = display_text
            else:
                box_color = (0, 0, 255)
                display_text = "Unknown"
                shared_status["student_name"] = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Draw recognition debug overlay if enabled
            debug_y_offset = 20
            if attendance_engine.dev_recognition_debug:
                # Calculate variance
                track_session = attendance_engine.sessions.get(tid)
                variance_str = "N/A"
                if track_session and track_session.identities:
                    votes = {}
                    recent_window = max(5, 15)
                    recent_identities = track_session.identities[-recent_window:]
                    for uid, sim, margin in recent_identities:
                        if uid not in votes:
                            votes[uid] = []
                        votes[uid].append(sim)
                    if votes:
                        best_id = max(votes, key=lambda k: len(votes[k]))
                        if len(votes[best_id]) > 1:
                            variance_str = f"{float(np.var(votes[best_id])):.4f}"
                        else:
                            variance_str = "0.0000"
                
                top1_sim = latest_metrics.get("top1_similarity", 0.0)
                margin = latest_metrics.get("margin", 0.0)
                q_score = latest_metrics.get("quality_score", 0.0)
                
                cv2.putText(frame, f"[DBG] TOP1: {display_text}", (x, y + h + debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                debug_y_offset += 15
                cv2.putText(frame, f"SIM: {top1_sim:.3f} | MARGIN: {margin:.3f}", (x, y + h + debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                debug_y_offset += 15
                cv2.putText(frame, f"VAR: {variance_str} | QUAL: {q_score:.3f}", (x, y + h + debug_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                debug_y_offset += 20 # Add space for decision state text below debug overlay

            # Check decision states
            if decision.attendance_state in ["MARKED", "ALREADY_MARKED"]:
                shared_status["message"] = f"MARKED: {decision.student_name}"
                cv2.putText(frame, "\u2713 ATTENDANCE MARKED", (x, y+h+debug_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN_COLOR, 2)
                continue
            elif decision.attendance_state == "FAILED":
                reason = decision.failure_reason or "Verification failed"
                shared_status["message"] = f"Failed ({reason})"
                cv2.putText(frame, f"FAILED: {reason}", (x, y+h+debug_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                continue
            elif decision.attendance_state == "RETRY" and not is_recognized:
                reason = decision.failure_reason or "Please align face"
                shared_status["message"] = reason
                cv2.putText(frame, reason, (x, y+h+debug_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                continue

            # Active liveness / calibration display
            # Always show the challenge dot once a TrackSession exists,
            # regardless of whether identity/PAD gate has cleared yet.
            track_session = attendance_engine.sessions.get(tid)
            if track_session is None:
                continue

            active_sess = track_session.challenge_engine.active_session
            if active_sess is None:
                continue

            current_index = active_sess["current_index"]
            sequence = active_sess["sequence"]
            target_cmd = decision.challenge_command or "CENTER"

            if active_sess["state"] in [ChallengeState.WAITING, ChallengeState.BASELINE_ACQUISITION]:
                pct_x, pct_y = 50, 50
            elif current_index < len(sequence):
                pct_x = sequence[current_index].screen_x
                pct_y = sequence[current_index].screen_y
            else:
                pct_x, pct_y = 50, 50

            xd = int(pct_x * w_frame / 100.0)
            yd = int(pct_y * h_frame / 100.0)

            # Eye outline indicators
            yunet_lm = best_det.landmarks if best_det and best_det.landmarks is not None else None
            if yunet_lm is not None and len(yunet_lm) >= 2:
                right_pt = yunet_lm[0]
                left_pt  = yunet_lm[1]
                eye_dist = float(np.linalg.norm(np.array(left_pt) - np.array(right_pt)))
                crop_w = int(eye_dist * 0.35)
                crop_h = int(eye_dist * 0.25)
                for pt in [right_pt, left_pt]:
                    ex, ey = int(pt[0]), int(pt[1])
                    cv2.rectangle(frame,
                                  (ex - crop_w, ey - crop_h),
                                  (ex + crop_w, ey + crop_h),
                                  (0, 220, 220), 1)

            # Stimulus dot: always render once face is in frame
            dot_alpha = 0.7 + 0.3 * abs(np.sin(time.time() * 6))
            glow_r  = int(DOT_RADIUS * 1.8)
            overlay = frame.copy()
            dot_color = RED_COLOR
            if active_sess["state"] in [ChallengeState.BASELINE_LOCKED, ChallengeState.TARGET_ACQUIRED]:
                dot_color = GREEN_COLOR

            cv2.circle(overlay, (xd, yd), glow_r, dot_color, -1)
            cv2.addWeighted(overlay, 0.3 * dot_alpha, frame, 1.0 - 0.3 * dot_alpha, 0, frame)
            cv2.circle(frame, (xd, yd), DOT_RADIUS, dot_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (xd, yd), DOT_RADIUS // 3,
                       (120, 120, 255) if dot_color == RED_COLOR else (120, 255, 120), -1, cv2.LINE_AA)

            # Debug logs
            dx = obs.gaze_x - active_sess["baseline_x"]
            dy = obs.gaze_y - active_sess["baseline_y"]
            target_distance_str = "N/A"
            if target_cmd in ["LEFT", "RIGHT", "UP", "DOWN"]:
                if target_cmd == "LEFT":
                    target_distance_str = f"{abs(dx - (-active_sess['T_x'])):.2f}"
                elif target_cmd == "RIGHT":
                    target_distance_str = f"{abs(dx - active_sess['T_x']):.2f}"
                elif target_cmd == "UP":
                    target_distance_str = f"{abs(dy - (-active_sess['T_y'])):.2f}"
                elif target_cmd == "DOWN":
                    target_distance_str = f"{abs(dy - active_sess['T_y']):.2f}"

            debug_lines = [
                f"Decision: {decision.attendance_state}",
                f"Mode: {active_sess['liveness_mode']}",
                f"Gaze X: {obs.gaze_x:+.2f}",
                f"Gaze Y: {obs.gaze_y:+.2f}",
                f"Delta X: {dx:+.2f}",
                f"Delta Y: {dy:+.2f}",
                f"Yaw: {yaw:+.1f} deg",
                f"Pitch: {pitch:+.1f} deg",
                f"Target: {target_cmd}",
                f"T_x/T_y: {active_sess['T_x']:.2f}/{active_sess['T_y']:.2f}",
                f"Target dist: {target_distance_str}",
                f"Confidence: {obs.confidence:.2f}",
                f"Step: {current_index + 1} / {len(sequence)}",
                f"Status: {active_sess['state'].value}"
            ]
            dy_offset = 20
            for i, line in enumerate(debug_lines):
                cv2.putText(frame, line, (15, 30 + i * dy_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # On-screen instruction
            if active_sess["state"] in [ChallengeState.WAITING, ChallengeState.BASELINE_ACQUISITION]:
                if decision.attendance_state == "RECOGNITION_PENDING":
                    instruction_text = "Identifying... look at center dot"
                    shared_status["message"] = "Identifying face..."
                else:
                    instruction_text = "Look at the center target dot"
                    shared_status["message"] = "Acquiring baseline..."
            elif decision.attendance_state == "LIVENESS_PENDING":
                instruction_text = f"Follow the dot  ({current_index + 1}/{len(sequence)})"
                shared_status["message"] = f"Gaze check {current_index + 1}/{len(sequence)}"
            else:
                instruction_text = "Processing..."
                shared_status["message"] = "Processing..."

            cv2.putText(frame, instruction_text, (10, h_frame - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        elif current_mode == "register":
            dense_landmarks = landmarker.get_landmarks(frame, track.bbox)
            pose_estimator = HeadPoseEstimator((frame.shape[1], frame.shape[0]))
            if dense_landmarks is not None and len(dense_landmarks) == 98:
                yaw, pitch, roll = pose_estimator.estimate_facex(dense_landmarks)
            else:
                yaw, pitch, roll = pose_estimator.estimate(best_det.landmarks)
                
            if registration_parser:
                quality_eval = quality_engine.evaluate(frame, track.bbox, (yaw, pitch, roll), track.confidence)
                
                registration_parser.add_frame_candidate(frame, best_det, (yaw, pitch, roll))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                cv2.putText(frame, "ENROLLING...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                if not quality_eval["accepted"]:
                    reason_str = ", ".join(quality_eval["reasons"])
                    cv2.putText(frame, f"WAIT: {reason_str}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                poses_gathered = set([c["pose_category"] for c in registration_parser.candidate_pool])
                shared_status["message"] = f"Frames: {len(registration_parser.candidate_pool)}/30 | Poses: {len(poses_gathered)}/3"
                
                has_all_poses = "FRONT" in poses_gathered and "UP" in poses_gathered and "DOWN" in poses_gathered
                if len(registration_parser.candidate_pool) >= 30 and has_all_poses:
                    templates = registration_parser.get_diverse_templates(sface, max_per_pose=3)
                    for cand in templates:
                        aligned = sface.align(cand["full_frame"], cand["best_det"])
                        emb = sface.get_embedding(aligned)
                        db_manager.save_template(registration_student_id, emb, {
                                "quality_score": cand["quality_score"],
                                "yaw": cand["pose"][0]
                            })
                    
                    vector_engine.reload_gallery()
                    current_mode = "idle"
                    shared_status["message"] = "Registration Complete"
            
            # Render timings for dev mode
            if current_mode == "mark":
                y_offset = 30
                for k, v in timings.items():
                    cv2.putText(frame, f"{k}: {v:.1f}ms", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 20
            
    return frame

# ── Endpoints ────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('admin.html')

@app.route('/register_page')
def register_page():
    return render_template('register.html')

@app.route('/student')
@app.route('/mark_page')
def mark_page():
    return render_template('student.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        cam_stream.acquire()
        try:
            while True:
                with cam_stream.lock:
                    frame = cam_stream.latest_frame
                    
                if frame is None:
                    time.sleep(0.05)
                    continue
                    
                try:
                    frame = process_frame(frame.copy())
                except Exception as e:
                    import traceback
                    with open("crash_log.txt", "a") as f:
                        f.write(traceback.format_exc() + "\n")
                    # Return an error frame so the stream doesn't die immediately
                    cv2.putText(frame, "SYSTEM CRASHED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.03)
        finally:
            cam_stream.release()
            
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_register', methods=['POST'])
def start_register():
    global current_mode, registration_parser, registration_student_id
    data = request.json
    name = data.get("name", "").strip()
    admission_number = data.get("admission_number", "").strip() or None
    
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    # Do not look up by name. Use admission_number to identify re-registrations.
    if admission_number:
        cursor.execute("SELECT id FROM students WHERE admission_number = ?", (admission_number,))
        row = cursor.fetchone()
    else:
        row = None
        
    if row:
        registration_student_id = row['id']
        # Do not delete old templates. Retire them for auditability.
        cursor.execute("UPDATE biometric_templates SET template_status = 'RETIRED' WHERE student_id = ?", (registration_student_id,))
        conn.commit()
    else:
        # Create DB entry
        registration_student_id = db_manager.add_student(name, admission_number)
    conn.close()
    
    registration_parser = VideoEnrollmentParser(quality_engine)
    current_mode = "register"
    
    return jsonify({"success": True, "message": f"Started enrollment for {name}"})

@app.route('/api/start_attendance', methods=['POST'])
def start_attendance():
    global current_mode
    current_mode = "mark"
    return jsonify({"success": True, "message": "Attendance scanning started"})

@app.route('/api/stop_attendance', methods=['POST'])
def stop_attendance():
    global current_mode
    current_mode = "idle"
    # Clear all sessions so next student starts fresh
    attendance_engine.sessions.clear()
    return jsonify({"success": True, "message": "Stopped scanning"})

@app.route('/api/delete_student', methods=['POST'])
def delete_student():
    data = request.json or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "message": "Student name is required"}), 400
        
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    # Get student id
    cursor.execute("SELECT id FROM students WHERE name = ?", (name,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"success": False, "message": f"Student '{name}' not found"}), 404
        
    student_id = row['id']
    
    # Delete templates, records, and student
    cursor.execute("DELETE FROM biometric_templates WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM attendance_records WHERE student_id = ?", (student_id,))
    cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()
    
    # Reload vector engine gallery so the embeddings are removed from memory
    vector_engine.reload_gallery()
    
    return jsonify({"success": True, "message": f"Deleted student '{name}' successfully"})

@app.route('/api/mark_absent', methods=['POST'])
def mark_absent():
    """Remove a student's attendance record for today, effectively marking them absent."""
    from datetime import date
    data = request.json or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "message": "Student name is required"}), 400

    today = date.today().isoformat()
    conn = db_manager.get_connection()
    cursor = conn.cursor()

    # Find student ID by name
    cursor.execute("SELECT id FROM students WHERE name = ?", (name,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"success": False, "message": f"Student '{name}' not found"}), 404

    student_id = row['id']

    # Delete today's attendance record only
    cursor.execute(
        "DELETE FROM attendance_records WHERE student_id = ? AND attendance_date = ?",
        (student_id, today)
    )
    conn.commit()
    conn.close()

    return jsonify({"success": True, "message": f"'{name}' marked absent for today"})

@app.route('/api/students', methods=['GET'])
def get_students():
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM students")
    students = [row['name'] for row in cursor.fetchall()]
    conn.close()
    return jsonify(students)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    # Format to match legacy UI structure: {"YYYY-MM-DD": ["Name1", "Name2"]}
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.attendance_date, s.name 
        FROM attendance_records a 
        JOIN students s ON a.student_id = s.id
    """)
    rows = cursor.fetchall()
    conn.close()
    
    logs = {}
    for row in rows:
        d = row['attendance_date']
        n = row['name']
        if d not in logs:
            logs[d] = []
        if n not in logs[d]:
            logs[d].append(n)
            
    return jsonify(logs)

@app.route('/api/admin_stats', methods=['GET'])
def get_admin_stats():
    from datetime import datetime
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    # Get all students
    cursor.execute("SELECT name FROM students")
    students = [row['name'] for row in cursor.fetchall()]
    total_students = len(students)
    
    # Get today's attendance
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT s.name 
        FROM attendance_records a 
        JOIN students s ON a.student_id = s.id 
        WHERE a.attendance_date = ?
    """, (today,))
    today_present = [row['name'] for row in cursor.fetchall()]
    conn.close()
    
    today_absent = list(set(students) - set(today_present))
    present_count = len(today_present)
    absent_count = len(today_absent)
    rate = (present_count / total_students * 100) if total_students > 0 else 0.0
    
    return jsonify({
        "total_students": total_students,
        "present_count": present_count,
        "absent_count": absent_count,
        "attendance_rate": rate,
        "students": students,
        "today_present": today_present,
        "today_absent": today_absent,
        "history": [] # Leaving history stubbed for brevity
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    global latest_metrics
    step = "front"
    if current_mode == "register" and registration_parser:
        poses = set([c["pose_category"] for c in registration_parser.candidate_pool])
        count = len(registration_parser.candidate_pool)
        
        if "FRONT" in poses and count > 10: step = "up"
        if "UP" in poses and count > 20: step = "down"
        if "DOWN" in poses and count >= 30: step = "done"
    elif current_mode == "idle":
        step = "done"
        
    # Default values for active challenge
    ui_state = current_mode
    waypoint = 0
    total_waypoints = 3
    gaze_dir = "N/A"

    if current_mode == "mark":
        tids = list(attendance_engine.sessions.keys())
        if tids:
            active_tid = tids[0]
            session = attendance_engine.sessions[active_tid]
            decision = getattr(session, 'latest_decision', None)
            
            if decision:
                state_map = {
                    "MARKED": "success_feedback",
                    "ALREADY_MARKED": "success_feedback",
                    "FAILED": "fail_feedback",
                }
                ui_state = state_map.get(decision.attendance_state, "mark")
                
                # Active challenge progress setup
                active_sess = session.challenge_engine.active_session
                if active_sess:
                    total_waypoints = len(active_sess["sequence"])
                    waypoint = active_sess["completed_steps"]
                    
                    if decision.attendance_state in ["MARKED", "ALREADY_MARKED"]:
                        waypoint = total_waypoints
                        gaze_dir = "COMPLETED"
                    elif decision.attendance_state == "FAILED":
                        waypoint = 0
                        gaze_dir = "FAILED"
                    else:
                        gaze_dir = decision.challenge_command or "N/A"
            else:
                ui_state = "mark"

    recognized_name = shared_status["student_name"]
    if current_mode == "mark" and tids:
        active_tid = tids[0]
        session = attendance_engine.sessions.get(active_tid)
        if session:
            active_sess = session.challenge_engine.active_session
            if active_sess and active_sess["state"] not in [ChallengeState.WAITING, ChallengeState.BASELINE_ACQUISITION]:
                if recognized_name == "Unknown":
                    recognized_name = "Authenticating Student"

    return jsonify({
        "state": ui_state,
        "recognized": recognized_name,
        "message": shared_status["message"],
        "register_step": step,
        "latest_metrics": latest_metrics,
        "waypoint": waypoint,
        "total_waypoints": total_waypoints,
        "gaze": gaze_dir
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
