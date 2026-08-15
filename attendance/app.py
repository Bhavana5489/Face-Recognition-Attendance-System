# -*- coding: utf-8 -*-
"""
=============================================================
  Gaze-Tracking Attendance Web Dashboard (Flask Backend)
  File: d:/projects/face_detection/attendance/app.py
=============================================================
"""

import os
import cv2
import numpy as np
import time
import json
import random
import sys
import threading
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# ── Directories & Paths ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Anti-Spoofing Integration ───────────────────────────
ANTI_SPOOF_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'anti_spoofing'))
sys.path.insert(0, ANTI_SPOOF_DIR)
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
app = Flask(__name__)
CORS(app)

DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
DB_PATH = os.path.join(BASE_DIR, 'database.json')

os.makedirs(DATASET_DIR, exist_ok=True)

# ── Device & FaceNet Setup ──────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Attendance System loading FaceNet on: {device}")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)

# ── Anti-Spoofing Models ────────────────────────────────
print("[*] Loading Deep Learning Anti-Spoofing models...")
model_test = AntiSpoofPredict(0)  # Use GPU 0
image_cropper = CropImage()
SPOOF_MODEL_DIR = os.path.join(ANTI_SPOOF_DIR, 'resources', 'anti_spoof_models')
SPOOF_THRESHOLD = 0.75
# ── Cascades ───────────────────────────────────────────
HC = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(HC + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(HC + "haarcascade_eye.xml")
eye_glasses_cascade = cv2.CascadeClassifier(HC + "haarcascade_eye_tree_eyeglasses.xml")

# ── Configuration Constants ─────────────────────────────
FACE_REC_THRESHOLD = 0.60      # Max Euclidean distance to count as recognized
GAZE_MARGIN = 0.05             # Gaze shift sensitivity
DOT_RADIUS = 18
RED = (30, 30, 220)
GREEN = (30, 210, 30)
YELLOW = (0, 210, 220)
CYAN = (220, 200, 0)
WHITE = (255, 255, 255)
DARK = (20, 20, 20)
GRAY = (160, 160, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT2 = cv2.FONT_HERSHEY_DUPLEX

# ── Anti-Spoofing Helper ────────────────────────────────
def check_liveness_deep_model(frame, bbox):
    """
    frame: original BGR frame
    bbox: [x, y, w, h] of the face
    Returns: True if REAL, False if FAKE/SPOOF
    """
    try:
        prediction = np.zeros((1, 3))
        for model_name in os.listdir(SPOOF_MODEL_DIR):
            if not model_name.endswith(".pth"): continue
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": frame, "bbox": bbox,
                "scale": scale, "out_w": w_input,
                "out_h": h_input, "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, os.path.join(SPOOF_MODEL_DIR, model_name))
        
        label = int(np.argmax(prediction))
        score = float(prediction[0][label] / 2)
        
        # label 1 is REAL, score must be >= threshold
        if label == 1 and score >= SPOOF_THRESHOLD:
            return True, score
        else:
            return False, score
    except Exception as e:
        print(f"[!] Anti-spoof prediction error: {e}")
        return False, 0.0

# ── Database Utilities ──────────────────────────────────
def load_db():
    if not os.path.exists(DB_PATH):
        default_db = {"students": [], "attendance": {}}
        with open(DB_PATH, 'w') as f:
            json.dump(default_db, f, indent=4)
        return default_db
    try:
        with open(DB_PATH, 'r') as f:
            return json.load(f)
    except:
        return {"students": [], "attendance": {}}

def save_db(data):
    with open(DB_PATH, 'w') as f:
        json.dump(data, f, indent=4)

# ── Face Recognition Embedding Cache ────────────────────
known_embeddings = {}  # { "Name": [emb1, emb2, ...] }

def load_face_embeddings():
    global known_embeddings
    known_embeddings = {}
    db = load_db()
    for name in db["students"]:
        student_dir = os.path.join(DATASET_DIR, name)
        if not os.path.exists(student_dir):
            continue
        embeddings = []
        for img_name in ['front.jpg', 'up.jpg', 'down.jpg']:
            img_path = os.path.join(student_dir, img_name)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            # The saved images are already tightly cropped by MTCNN
            face_crop = cv2.resize(img, (160, 160))
            # Standardize and compute embedding
            face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float().to(device)
            face_tensor = (face_tensor - 127.5) / 128.0
            face_tensor = face_tensor.unsqueeze(0)
            with torch.no_grad():
                emb = resnet(face_tensor).cpu().numpy().flatten()
                embeddings.append(emb)
        if embeddings:
            known_embeddings[name] = embeddings
    print(f"[*] Loaded face profiles for: {list(known_embeddings.keys())}")

# Load on start
load_face_embeddings()

# ── Gaze Tracking Logic (from attendance_tracker.py) ─────
def get_pupil_x_ratio(eye_roi_bgr):
    if eye_roi_bgr is None or eye_roi_bgr.size == 0:
        return None
    gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    margin_x = int(w * 0.25)  # Exclude more side margins (eyelashes/glasses frames)
    margin_y = int(h * 0.30)  # Exclude more top/bottom margins (eyelashes/eyelids)
    crop = gray[margin_y:h-margin_y, margin_x:w-margin_x]
    if crop.size == 0:
        return None
    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(crop)
    cx = min_loc[0] + margin_x
    return float(cx / w)

def get_pupil_x_ratio_average(frame, face_bbox, face_landmarks=None):
    if face_landmarks is not None and len(face_landmarks) > 0:
        # Use MTCNN landmarks
        pts = face_landmarks[0]  # shape: (5, 2)
        lx, ly = pts[0]
        rx, ry = pts[1]
        
        # Calculate eye crop size relative to distance between eyes
        eye_dist = np.sqrt((rx - lx)**2 + (ry - ly)**2)
        if eye_dist > 0:
            crop_w = int(eye_dist * 0.35)
            crop_h = int(eye_dist * 0.25)
            
            ratios = []
            for (ex, ey) in [(lx, ly), (rx, ry)]:
                ex_min = max(0, int(ex - crop_w))
                ex_max = max(0, int(ex + crop_w))
                ey_min = max(0, int(ey - crop_h))
                ey_max = max(0, int(ey + crop_h))
                
                h_img, w_img = frame.shape[:2]
                ex_min = min(w_img - 1, ex_min)
                ex_max = min(w_img - 1, ex_max)
                ey_min = min(h_img - 1, ey_min)
                ey_max = min(h_img - 1, ey_max)
                
                eye_roi = frame[ey_min:ey_max, ex_min:ex_max]
                if eye_roi.size > 0:
                    r = get_pupil_x_ratio(eye_roi)
                    if r is not None:
                        ratios.append(r)
            if ratios:
                return float(np.mean(ratios))

    # Fallback to Haar Cascade if landmarks not present or failed
    fx, fy, fw, fh = face_bbox
    face_roi = frame[fy: fy + fh, fx: fx + fw]
    if face_roi.size == 0:
        return None
    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # 1. Try standard eye cascade
    eyes = eye_cascade.detectMultiScale(
        face_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)
    )
    # 2. Fallback to eyeglasses eye cascade if standard fails
    if len(eyes) == 0:
        eyes = eye_glasses_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(18, 18)
        )
        
    if len(eyes) == 0:
        return None
        
    ratios = []
    for (ex, ey, ew, eh) in eyes[:2]:
        eye_roi = face_roi[ey: ey + int(eh * 0.65), ex: ex + ew]
        r = get_pupil_x_ratio(eye_roi)
        if r is not None:
            ratios.append(r)
            
    if not ratios:
        return None
    return float(np.mean(ratios))

def sample_pupil_direction(frame, face_bbox, face_landmarks=None):
    avg = get_pupil_x_ratio_average(frame, face_bbox, face_landmarks)
    if avg is None:
        return None
    if avg < (0.5 - GAZE_MARGIN):
        return "left"
    elif avg > (0.5 + GAZE_MARGIN):
        return "right"
    else:
        return "center"

def generate_waypoints(frame_w, frame_h, n=2):
    margin_x = 120
    margin_y = 120
    zones = [
        (margin_x, frame_w // 3, margin_y, frame_h - margin_y, "left"),
        (frame_w * 2 // 3, frame_w - margin_x, margin_y, frame_h - margin_y, "right")
    ]
    random.shuffle(zones)
    waypoints = []
    for (xmin, xmax, ymin, ymax, label) in zones[:n]:
        x = random.randint(int(xmin), int(xmax))
        y = random.randint(int(ymin), int(ymax))
        waypoints.append((x, y, label))
    return waypoints

class GazeSession:
    def __init__(self, frame_w, frame_h):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.state = "running"
        self.waypoints = generate_waypoints(frame_w, frame_h, 2)  # Generate 2 points (Left and Right)
        self.wp_idx = 0
        self.wp_start = time.time()
        self.wp_duration = 1.5  # 1.5s per static dot stop
        self.results = []
        self.dot_pos = (self.waypoints[0][0], self.waypoints[0][1])
        self.dot_alpha = 1.0
        self.gaze_samples = []  # Accumulate raw ratio samples for robustness

    def update(self, frame, face_bbox, face_landmarks=None):
        now = time.time()
        elapsed = now - self.wp_start
        wp = self.waypoints[self.wp_idx]
        target_x, target_y, zone_label = wp

        # Static dot position (no sliding!)
        self.dot_pos = (target_x, target_y)
        self.dot_alpha = 0.7 + 0.3 * abs(np.sin(now * 6))

        # Accumulate raw pupil X-ratio samples after 0.5s reaction window
        if elapsed >= 0.50 and face_bbox is not None:
            r = get_pupil_x_ratio_average(frame, face_bbox, face_landmarks)
            if r is not None:
                self.gaze_samples.append(r)

        if elapsed >= self.wp_duration:
            # Save the average ratio for this stop
            stop_avg = np.mean(self.gaze_samples) if self.gaze_samples else None
            self.results.append((zone_label, stop_avg))
            
            self.wp_idx += 1
            self.wp_start = now
            self.gaze_samples = []  # Reset samples for next waypoint
            
            if self.wp_idx < len(self.waypoints):
                next_wp = self.waypoints[self.wp_idx]
                self.dot_pos = (next_wp[0], next_wp[1])
            else:
                # Finished all waypoints: Evaluate
                left_ratio = None
                right_ratio = None
                for zone, avg_val in self.results:
                    if zone == "left":
                        left_ratio = avg_val
                    elif zone == "right":
                        right_ratio = avg_val
                
                passed = False
                if left_ratio is not None and right_ratio is not None:
                    # Relative shift check (Right pupil ratio must be greater than Left)
                    diff = right_ratio - left_ratio
                    print(f"[*] Relative shift: {diff:.4f} (Left Avg: {left_ratio:.4f}, Right Avg: {right_ratio:.4f})")
                    # Gaze shift is positive and significant
                    if diff >= 0.015:
                        passed = True
                else:
                    # Fallback to absolute thresholds if one side was completely missed
                    print(f"[!] Gaze fallback: Left: {left_ratio}, Right: {right_ratio}")
                    if left_ratio is not None and left_ratio < 0.49:
                        passed = True
                    elif right_ratio is not None and right_ratio > 0.51:
                        passed = True
                        
                self.state = "success" if passed else "failed"
                return True
        return False

# ── Camera & State Manager ──────────────────────────────
class CameraManager:
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        
        # State: idle, register, mark, success_feedback, fail_feedback
        self.state = "idle"
        self.student_name = ""
        self.register_start_time = 0.0
        self.register_step = "front"
        
        # Best clear frames for registration
        self.best_front_frame = None
        self.best_front_var = 0.0
        self.best_up_frame = None
        self.best_up_var = 0.0
        self.best_down_frame = None
        self.best_down_var = 0.0
        self.best_overall_frame = None
        self.best_overall_var = 0.0
        
        # Verification / Gaze check session
        self.gaze_session = None
        self.feedback_start_time = 0.0
        self.current_gaze = "N/A"
        
        self.latest_frame = None
        self.running = False
        
        # --- ASYNC AI STATE ---
        self.ai_running = False
        self.shared_bbox = None
        self.shared_landmarks = None
        self.shared_identity = "Unknown"
        self.shared_dist = 1.0
        self.shared_spoof = (True, 1.0)
        self.shared_pitch = 0.5
        self.shared_var = 0.0
        self.shared_background_faces = []
        self.shared_warning = ""
        self.last_spoof_check_time = 0.0
        self.last_rec_time = 0.0
        
    def start(self):
        with self.lock:
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.running = True
                self.ai_running = True
                threading.Thread(target=self._update_frame, daemon=True).start()
                threading.Thread(target=self._ai_worker, daemon=True).start()
                
    def _update_frame(self):
        while self.running:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    self.latest_frame = frame
            import time
            time.sleep(0.01)
            
    def _ai_worker(self):
        import time
        import cv2
        import torch
        import numpy as np
        
        while self.ai_running:
            frame = self.latest_frame
            if frame is None:
                time.sleep(0.02)
                continue
                
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # 1. MTCNN Detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
            
            face_bbox = None
            face_landmarks = None
            bg_faces = []
            if boxes is not None and len(boxes) > 0:
                center_x, center_y = w / 2.0, h / 2.0
                best_dist = float('inf')
                best_idx = -1
                
                for i, box in enumerate(boxes):
                    if probs[i] < 0.85: continue
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w - 1, int(x2)), min(h - 1, int(y2))
                    fw, fh = x2 - x1, y2 - y1
                    if fw >= 80 and fh >= 80:
                        bcx, bcy = x1 + fw/2.0, y1 + fh/2.0
                        dist = (bcx - center_x)**2 + (bcy - center_y)**2
                        if dist < best_dist:
                            if best_idx != -1:
                                # Add previous best to background
                                px1, py1, px2, py2 = boxes[best_idx]
                                bg_faces.append([int(px1), int(py1), int(px2-px1), int(py2-py1)])
                            best_dist = dist
                            best_idx = i
                        else:
                            bg_faces.append([x1, y1, fw, fh])
                            
                if best_idx != -1:
                    x1, y1, x2, y2 = boxes[best_idx]
                    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w - 1, int(x2)), min(h - 1, int(y2))
                    face_bbox = [x1, y1, x2 - x1, y2 - y1]
                    face_landmarks = [landmarks[best_idx]]
                    
            self.shared_background_faces = bg_faces
            
            self.shared_bbox = face_bbox
            self.shared_landmarks = face_landmarks
            
            if face_bbox:
                fx, fy, fw, fh = face_bbox
                face_roi = frame[fy:fy+fh, fx:fx+fw]
                
                if face_roi.size > 0:
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    
                    # Brightness Warning
                    brightness = np.mean(face_gray)
                    if brightness < 40:
                        self.shared_warning = "Low Light Detected! Increase lighting."
                    elif brightness > 220:
                        self.shared_warning = "Too Bright / Backlit! Avoid windows."
                    else:
                        self.shared_warning = ""
                    
                    # Sharpness Calculation
                    self.shared_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
                    
                    # Pitch Calculation
                    if face_landmarks is not None and len(face_landmarks) > 0:
                        pts = face_landmarks[0]
                        ly, ry = pts[0][1], pts[1][1]
                        ny = pts[2][1]
                        lmy, rmy = pts[3][1], pts[4][1]
                        eye_y = (ly + ry) / 2.0
                        mouth_y = (lmy + rmy) / 2.0
                        face_h = mouth_y - eye_y
                        if face_h > 0:
                            self.shared_pitch = (ny - eye_y) / face_h
                            
                now = time.time()
                
                # 2. Anti-Spoofing Check (Throttled)
                if self.state in ["register", "mark"] and now - self.last_spoof_check_time > 0.5:
                    self.shared_spoof = check_liveness_deep_model(frame, face_bbox)
                    self.last_spoof_check_time = now
                    
                # 3. FaceNet Recognition (Throttled)
                if self.state in ["idle", "mark"] and now - self.last_rec_time > 0.5:
                    if face_roi.size > 0:
                        face_crop = cv2.resize(face_roi, (160, 160))
                        face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float().to(device)
                        face_tensor = (face_tensor - 127.5) / 128.0
                        face_tensor = face_tensor.unsqueeze(0)
                        
                        with torch.no_grad():
                            live_emb = resnet(face_tensor).cpu().numpy().flatten()
                        
                        recognized = "Unknown"
                        min_dist = 1.0
                        for name, embs in known_embeddings.items():
                            for emb in embs:
                                dist = np.linalg.norm(live_emb - emb)
                                if dist < min_dist:
                                    min_dist = dist
                                    recognized = name
                        
                        self.shared_identity = recognized
                        self.shared_dist = float(min_dist)
                        self.last_rec_time = now
            else:
                self.shared_identity = "Unknown"
                self.shared_dist = 1.0
                self.shared_var = 0.0
                self.shared_pitch = 0.5
                
            # Throttle AI loop to prevent maxing out CPU
            time.sleep(0.05)
                
    def stop(self):
        self.running = False
        self.ai_running = False
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.latest_frame = None

camera_manager = CameraManager()

# ── Video Streaming & Processing ────────────────────────
def draw_face_box(frame, bbox, color=CYAN):
    x, y, w, h = bbox
    t = 4
    L = 20
    pts = [
        ((x, y), (x + L, y), (x, y + L)),
        ((x + w, y), (x + w - L, y), (x + w, y + L)),
        ((x, y + h), (x + L, y + h), (x, y + h - L)),
        ((x + w, y + h), (x + w - L, y + h), (x + w, y + h - L)),
    ]
    for corner in pts:
        c, h1, v1 = corner
        cv2.line(frame, c, h1, color, t, cv2.LINE_AA)
        cv2.line(frame, c, v1, color, t, cv2.LINE_AA)

def process_frame(frame):
    state = camera_manager.state
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    face_bbox = camera_manager.shared_bbox
    face_landmarks = camera_manager.shared_landmarks

    # Draw background faces
    for bg_box in camera_manager.shared_background_faces:
        bx, by, bw, bh = bg_box
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (100, 100, 100), 1, cv2.LINE_4)
        
    # Draw warnings
    if camera_manager.shared_warning:
        cv2.putText(frame, camera_manager.shared_warning, (10, h - 20), FONT, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    # ── SUCCESS/FAIL FEEDBACK STATES ────────────────────
    if state in ("success_feedback", "fail_feedback"):
        if time.time() - camera_manager.feedback_start_time > 3.0:
            camera_manager.state = "idle"
            camera_manager.gaze_session = None
            camera_manager.shared_identity = "Unknown"
        
        msg = f"ACCESS GRANTED: {camera_manager.shared_identity}" if state == "success_feedback" else "VERIFICATION FAILED"
        color = GREEN if state == "success_feedback" else RED
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 40), (w, h // 2 + 40), DARK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, msg, (w // 2 - (len(msg) * 8), h // 2 + 10), FONT2, 0.85, color, 2, cv2.LINE_AA)
        return frame

    # ── REGISTRATION STATE ──────────────────────────────
    if state == "register":
        cv2.rectangle(frame, (0, 0), (w, 40), DARK, -1)
        
        step = getattr(camera_manager, "register_step", "front")
        step_text = ""
        if step == "front":
            step_text = "Step 1: Look straight at camera (Front Face)"
            progress_w = int(w * 0.15)
        elif step == "up":
            step_text = "Step 2: Tilt Head Upward"
            progress_w = int(w * 0.50)
        else:
            step_text = "Step 3: Tilt Head Downward"
            progress_w = int(w * 0.85)
            
        cv2.rectangle(frame, (0, 36), (progress_w, 40), YELLOW, -1)
        cv2.putText(frame, step_text, (10, 26), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        
        if face_bbox:
            fx, fy, fw, fh = face_bbox
            is_real, spoof_score = camera_manager.shared_spoof
            if not is_real:
                draw_face_box(frame, face_bbox, RED)
                cv2.putText(frame, f"SPOOF DETECTED [{spoof_score:.2f}]", (fx, fy - 10), FONT, 0.6, RED, 2, cv2.LINE_AA)
                cv2.putText(frame, "Registration Blocked", (fx, fy + fh + 20), FONT, 0.55, RED, 1, cv2.LINE_AA)
                return frame
                
            draw_face_box(frame, face_bbox, YELLOW)
            
            var = camera_manager.shared_var
            pitch_ratio = camera_manager.shared_pitch
            
            if var >= 40.0:
                face_roi = frame[fy:fy+fh, fx:fx+fw]
                if var > camera_manager.best_overall_var:
                    camera_manager.best_overall_var = var
                    camera_manager.best_overall_frame = face_roi.copy()
                    
                if step == "front" and (0.43 <= pitch_ratio <= 0.58):
                    if var > camera_manager.best_front_var:
                        camera_manager.best_front_var = var
                        camera_manager.best_front_frame = face_roi.copy()
                    camera_manager.register_step = "up"
                elif step == "up" and (pitch_ratio < 0.43):
                    if var > camera_manager.best_up_var:
                        camera_manager.best_up_var = var
                        camera_manager.best_up_frame = face_roi.copy()
                    camera_manager.register_step = "down"
                elif step == "down" and (pitch_ratio > 0.58):
                    if var > camera_manager.best_down_var:
                        camera_manager.best_down_var = var
                        camera_manager.best_down_frame = face_roi.copy()
                    camera_manager.register_step = "done"
                    import threading
                    threading.Thread(target=save_registration).start()
                        
            cv2.putText(frame, f"Sharpness: {int(var)} (Target: >40)", (w - 230, 45), FONT, 0.50, GREEN if var >= 40 else RED, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Pitch: {pitch_ratio:.2f}", (w - 230, 65), FONT, 0.50, CYAN, 1, cv2.LINE_AA)
        return frame

    # ── ATTENDANCE MARKING STATE ────────────────────────
    if state == "mark":
        cv2.rectangle(frame, (0, 0), (w, 40), DARK, -1)
        cv2.putText(frame, "MARKING ATTENDANCE - Gaze Check Enabled", (10, 26), FONT, 0.52, CYAN, 1, cv2.LINE_AA)
        
        if face_bbox:
            fx, fy, fw, fh = face_bbox
            
            is_real, spoof_score = camera_manager.shared_spoof
            if not is_real:
                draw_face_box(frame, face_bbox, RED)
                cv2.putText(frame, f"SPOOF DETECTED [{spoof_score:.2f}]", (fx, fy - 10), FONT, 0.6, RED, 2, cv2.LINE_AA)
                cv2.putText(frame, "Access Blocked", (fx, fy + fh + 20), FONT, 0.55, RED, 1, cv2.LINE_AA)
                return frame
                
            recognized = camera_manager.shared_identity
            min_dist = camera_manager.shared_dist
            
            if min_dist <= FACE_REC_THRESHOLD:
                draw_face_box(frame, face_bbox, GREEN)
                cv2.putText(frame, f"{recognized} ({int((1-min_dist)*100)}% match)", (fx, fy - 10), FONT, 0.55, GREEN, 1, cv2.LINE_AA)
                
                if camera_manager.gaze_session is None:
                    camera_manager.gaze_session = GazeSession(w, h)
                
                done = camera_manager.gaze_session.update(frame, face_bbox, face_landmarks)
                
                if face_landmarks is not None and len(face_landmarks) > 0:
                    pts = face_landmarks[0]
                    lx, ly = pts[0]
                    rx, ry = pts[1]
                    eye_dist = np.sqrt((rx - lx)**2 + (ry - ly)**2)
                    crop_w = int(eye_dist * 0.20)
                    crop_h = int(eye_dist * 0.15)
                    for (ex, ey) in [(lx, ly), (rx, ry)]:
                        cv2.rectangle(frame, (int(ex-crop_w), int(ey-crop_h)), (int(ex+crop_w), int(ey+crop_h)), YELLOW, 1)
                
                session = camera_manager.gaze_session
                x_dot, y_dot = session.dot_pos
                glow_r = int(DOT_RADIUS * 1.8)
                overlay = frame.copy()
                cv2.circle(overlay, (x_dot, y_dot), glow_r, (50, 50, 230), -1)
                cv2.addWeighted(overlay, 0.3 * session.dot_alpha, frame, 1 - 0.3 * session.dot_alpha, 0, frame)
                cv2.circle(frame, (x_dot, y_dot), DOT_RADIUS, RED, -1, cv2.LINE_AA)
                cv2.circle(frame, (x_dot, y_dot), DOT_RADIUS // 3, (120, 120, 255), -1, cv2.LINE_AA)
                
                g = sample_pupil_direction(frame, face_bbox, face_landmarks)
                camera_manager.current_gaze = g.upper() if g else "N/A"
                
                if done:
                    camera_manager.feedback_start_time = time.time()
                    if session.state == "success":
                        camera_manager.state = "success_feedback"
                        import threading
                        threading.Thread(target=log_attendance, args=(recognized,)).start()
                    else:
                        camera_manager.state = "fail_feedback"
            else:
                camera_manager.gaze_session = None
                draw_face_box(frame, face_bbox, RED)
                cv2.putText(frame, "Unknown Student", (fx, fy - 10), FONT, 0.55, RED, 1, cv2.LINE_AA)
        else:
            camera_manager.gaze_session = None
            
        return frame

    # ── IDLE STATE ──────────────────────────────────────
    if face_bbox:
        draw_face_box(frame, face_bbox, CYAN)
        fx, fy, fw, fh = face_bbox
        
        recognized = camera_manager.shared_identity
        min_dist = camera_manager.shared_dist
        
        if min_dist <= FACE_REC_THRESHOLD:
            cv2.putText(frame, f"{recognized} ({int((1-min_dist)*100)}% match)", (fx, fy - 10), FONT, 0.55, GREEN, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Unknown Face", (fx, fy - 10), FONT, 0.55, RED, 1, cv2.LINE_AA)
            
    return frame

def save_registration():
    name = camera_manager.student_name
    student_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(student_dir, exist_ok=True)
    
    # Check fallback source
    fallback = camera_manager.best_overall_frame
    
    front = camera_manager.best_front_frame if camera_manager.best_front_frame is not None else fallback
    up = camera_manager.best_up_frame if camera_manager.best_up_frame is not None else front
    down = camera_manager.best_down_frame if camera_manager.best_down_frame is not None else front
    
    saved = False
    if front is not None:
        cv2.imwrite(os.path.join(student_dir, 'front.jpg'), front)
        saved = True
    if up is not None:
        cv2.imwrite(os.path.join(student_dir, 'up.jpg'), up)
    if down is not None:
        cv2.imwrite(os.path.join(student_dir, 'down.jpg'), down)
        
    if saved:
        # Update JSON DB
        db = load_db()
        if name not in db["students"]:
            db["students"].append(name)
            save_db(db)
        # Reload embeddings
        load_face_embeddings()
        print(f"[*] Registered student {name} successfully!")
    else:
        print(f"[!] Registration failed: No clear front face captured.")
        
    # Reset manager status
    camera_manager.state = "idle"
    camera_manager.student_name = ""

def log_attendance(name, subject=None):
    db = load_db()
    today = time.strftime("%Y-%m-%d")
    
    # Revert to flat list if it was mistakenly converted to dict
    if today in db["attendance"] and isinstance(db["attendance"][today], dict):
        flat_list = []
        for sub, students in db["attendance"][today].items():
            flat_list.extend(students)
        db["attendance"][today] = list(set(flat_list))
        
    if today not in db["attendance"]:
        db["attendance"][today] = []
        
    if name not in db["attendance"][today]:
        db["attendance"][today].append(name)
        save_db(db)
        print(f"[*] Logged attendance for {name} today!")

# ── Flask API Endpoints ─────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register_page')
def register_page():
    return render_template('register.html')

@app.route('/mark_page')
def mark_page():
    return render_template('mark.html')

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        camera_manager.start()
        while True:
            frame = camera_manager.latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
                
            # Process a copy to prevent thread tearing
            frame_processed = process_frame(frame.copy())
            ret, jpeg = cv2.imencode('.jpg', frame_processed)
            if not ret:
                continue
            data = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
            time.sleep(0.06)  # ~15 FPS target for web stream
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_register', methods=['POST'])
def start_register():
    data = request.json
    name = data.get("name", "").strip().replace(" ", "_")
    if not name:
        return jsonify({"success": False, "message": "Name cannot be empty"}), 400
    
    camera_manager.state = "register"
    camera_manager.student_name = name
    camera_manager.register_start_time = time.time()
    camera_manager.register_step = "front"
    camera_manager.best_front_frame = None
    camera_manager.best_front_var = 0.0
    camera_manager.best_up_frame = None
    camera_manager.best_up_var = 0.0
    camera_manager.best_down_frame = None
    camera_manager.best_down_var = 0.0
    camera_manager.best_overall_frame = None
    camera_manager.best_overall_var = 0.0
    
    return jsonify({"success": True, "message": f"Started registration recording for {name}"})

@app.route('/api/start_attendance', methods=['POST'])
def start_attendance():
    camera_manager.state = "mark"
    camera_manager.gaze_session = None
    camera_manager.shared_identity = "Unknown"
    return jsonify({"success": True, "message": "Gaze Attendance marking started"})

@app.route('/api/stop_attendance', methods=['POST'])
def stop_attendance():
    camera_manager.state = "idle"
    camera_manager.gaze_session = None
    camera_manager.shared_identity = "Unknown"
    return jsonify({"success": True, "message": "Camera state reset to idle"})

@app.route('/api/export_logs', methods=['GET'])
def export_logs():
    db = load_db()
    import csv
    from io import StringIO
    from flask import make_response
    
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(["Date", "Subject", "Student Name"])
    
    for date, subjects in db.get("attendance", {}).items():
        if isinstance(subjects, list):
            # Backwards compatibility
            for student in subjects:
                cw.writerow([date, "General", student])
        else:
            for subject, students in subjects.items():
                for student in students:
                    cw.writerow([date, subject, student])
                    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=attendance_export.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/api/students', methods=['GET'])
def get_students():
    db = load_db()
    return jsonify(db["students"])

@app.route('/api/logs', methods=['GET'])
def get_logs():
    db = load_db()
    return jsonify(db["attendance"])

@app.route('/api/status', methods=['GET'])
def get_status():
    gaze_state = "idle"
    wp_idx = 0
    if camera_manager.gaze_session is not None:
        gaze_state = camera_manager.gaze_session.state
        wp_idx = camera_manager.gaze_session.wp_idx
        
    return jsonify({
        "state": camera_manager.state,
        "recognized": camera_manager.shared_identity,
        "gaze": camera_manager.current_gaze,
        "gaze_state": gaze_state,
        "waypoint": wp_idx,
        "register_step": getattr(camera_manager, "register_step", "front")
    })

if __name__ == '__main__':
    # Make sure app cleans up camera on exit
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        camera_manager.stop()
