import cv2
import numpy as np
import time
import os
import yaml
from typing import Tuple, Dict, Any, Optional

def load_config():
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        config_path = os.path.join(base_dir, 'config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def fuse_gaze_observations(left_gaze_x: float, left_gaze_y: float, left_conf: float,
                           right_gaze_x: float, right_gaze_y: float, right_conf: float) -> Tuple[float, float, float]:
    """
    Fuses left and right eye gaze coordinates using soft confidence fusion with zero-denominator guards.
    """
    # Disagreement penalty
    disagreement = np.sqrt((left_gaze_x - right_gaze_x)**2 + (left_gaze_y - right_gaze_y)**2)
    if disagreement > 0.5:
        left_conf *= 0.5
        right_conf *= 0.5
        
    left_valid = left_conf >= 0.10
    right_valid = right_conf >= 0.10
    
    if left_valid and right_valid:
        denom = left_conf + right_conf
        if denom > 0.0:
            raw_x = (left_gaze_x * left_conf + right_gaze_x * right_conf) / denom
            raw_y = (left_gaze_y * left_conf + right_gaze_y * right_conf) / denom
            combined_conf = denom / 2.0
        else:
            raw_x = 0.0
            raw_y = 0.0
            combined_conf = 0.0
    elif left_valid:
        raw_x = left_gaze_x
        raw_y = left_gaze_y
        combined_conf = left_conf * 0.7  # Single-eye penalty
    elif right_valid:
        raw_x = right_gaze_x
        raw_y = right_gaze_y
        combined_conf = right_conf * 0.7  # Single-eye penalty
    else:
        raw_x = 0.0
        raw_y = 0.0
        combined_conf = 0.0
        
    return raw_x, raw_y, combined_conf

class GazeObservation:
    """
    Immutable data structure representing a single gaze observation.
    """
    def __init__(self, timestamp: float, gaze_x: float, gaze_y: float, confidence: float, source: str, yaw: float, pitch: float, roll: float):
        self.timestamp = timestamp
        self.gaze_x = gaze_x        # Normalized: -1.0 (LEFT) to +1.0 (RIGHT)
        self.gaze_y = gaze_y        # Normalized: -1.0 (UP) to +1.0 (DOWN)
        self.confidence = confidence
        self.source = source        # "DEGRADED" or "FULL"
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.is_blinking = False
        self.blink_score = None

class GazeEstimator:
    """
    Base class interface for gaze direction estimators.
    """
    def estimate(self, frame: np.ndarray, face_detection: Any, landmarks: np.ndarray = None, pose: Tuple[float, float, float] = (0.0, 0.0, 0.0), camera_id: str = "0") -> GazeObservation:
        raise NotImplementedError("Must implement estimate method")

class FallbackGazeEstimator(GazeEstimator):
    """
    Degraded mode estimator using YuNet 5-point eye centers to crop and trace 2D pupil shifts.
    """
    def __init__(self, db_manager: Any = None):
        self.db_manager = db_manager

    def _get_pupil_position_2d(self, eye_crop: np.ndarray) -> Tuple[float, float, float]:
        """
        Extracts the normalized horizontal and vertical coordinates of the pupil within the eye crop.
        Returns (norm_x, norm_y, confidence).
        """
        if eye_crop is None or eye_crop.size == 0 or eye_crop.shape[0] == 0 or eye_crop.shape[1] == 0:
            return 0.5, 0.5, 0.0
            
        gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)
        
        # Calculate contrast and intensity confidence
        contrast = float(max_val - min_val) / 255.0
        intensity_conf = max(0.0, 1.0 - (float(min_val) / 120.0))
        eye_confidence = contrast * intensity_conf
        
        # Segment dark regions
        _, thresh = cv2.threshold(blur, min_val + 15, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crop_h, crop_w = eye_crop.shape[:2]
        center_x, center_y = crop_w / 2.0, crop_h / 2.0
        
        if not contours:
            # Fallback to darkest pixel center if no contours
            cX, cY = min_loc
            norm_x = cX / crop_w
            norm_y = cY / crop_h
            return norm_x, norm_y, eye_confidence * 0.4
            
        best_contour = None
        best_score = -float('inf')
        
        # Load configurable center distance penalty from config
        config = load_config()
        center_distance_penalty = config.get('center_distance_penalty', 0.1)
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < 2 or area > (crop_w * crop_h * 0.45):
                continue
                
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            # Penalize distance from eye crop center
            dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            norm_dist = dist_from_center / np.sqrt(center_x**2 + center_y**2)
            
            # Reward circular shape
            peri = cv2.arcLength(c, True)
            circularity = (4 * np.pi * area) / (peri**2) if peri > 0 else 0.0
            
            score = circularity * 0.8 - norm_dist * center_distance_penalty
            if score > best_score:
                best_score = score
                best_contour = c
                
        if best_contour is None:
            # Fallback to largest contour
            best_contour = max(contours, key=cv2.contourArea)
            
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            cX, cY = min_loc
        else:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            
        norm_x = cX / crop_w
        norm_y = cY / crop_h
        
        # Calculate compactness multiplier
        area = cv2.contourArea(best_contour)
        peri = cv2.arcLength(best_contour, True)
        compactness = (4 * np.pi * area) / (peri**2) if (peri > 0 and area > 0) else 0.5
        
        final_confidence = eye_confidence * max(0.2, min(1.0, compactness))
        
        # Penalize confidence slightly if far from center to reduce shadow/eyelash impact
        dist_from_center = np.sqrt((cX - center_x)**2 + (cY - center_y)**2)
        norm_dist = dist_from_center / np.sqrt(center_x**2 + center_y**2)
        final_confidence *= (1.0 - 0.3 * norm_dist)
        
        # Classify pupil observation confidence
        if final_confidence >= 0.25 and compactness >= 0.70:
            # PUPIL_CONFIDENT
            pass
        elif final_confidence >= 0.10 and compactness >= 0.40:
            # PUPIL_WEAK
            pass
        else:
            # PUPIL_INVALID
            final_confidence = 0.0
            
        return norm_x, norm_y, final_confidence

    def estimate(self, frame: np.ndarray, face_detection: Any, landmarks: np.ndarray = None, pose: Tuple[float, float, float] = (0.0, 0.0, 0.0), camera_id: str = "0") -> GazeObservation:
        yaw, pitch, roll = pose
        if face_detection is None or face_detection.landmarks is None or len(face_detection.landmarks) < 2:
            return GazeObservation(time.time(), 0.0, 0.0, 0.0, "DEGRADED", yaw, pitch, roll)
            
        right_eye_pt = np.array(face_detection.landmarks[0])
        left_eye_pt = np.array(face_detection.landmarks[1])
        
        eye_dist = np.linalg.norm(left_eye_pt - right_eye_pt)
        if eye_dist == 0:
            return GazeObservation(time.time(), 0.0, 0.0, 0.0, "DEGRADED", yaw, pitch, roll)
            
        # Define eye region crops
        eye_w = eye_dist * 0.35
        eye_h = eye_dist * 0.25
        
        # Left eye crop
        lx, ly = left_eye_pt[0], left_eye_pt[1]
        l_x1, l_x2 = int(lx - eye_w/2), int(lx + eye_w/2)
        l_y1, l_y2 = int(ly - eye_h/2), int(ly + eye_h/2)
        l_x1, l_y1 = max(0, l_x1), max(0, l_y1)
        l_x2, l_y2 = min(frame.shape[1], l_x2), min(frame.shape[0], l_y2)
        left_eye_crop = frame[l_y1:l_y2, l_x1:l_x2]
        
        # Right eye crop
        rx, ry = right_eye_pt[0], right_eye_pt[1]
        r_x1, r_x2 = int(rx - eye_w/2), int(rx + eye_w/2)
        r_y1, r_y2 = int(ry - eye_h/2), int(ry + eye_h/2)
        r_x1, r_y1 = max(0, r_x1), max(0, r_y1)
        r_x2, r_y2 = min(frame.shape[1], r_x2), min(frame.shape[0], r_y2)
        right_eye_crop = frame[r_y1:r_y2, r_x1:r_x2]
        
        left_norm_x, left_norm_y, left_conf = self._get_pupil_position_2d(left_eye_crop)
        right_norm_x, right_norm_y, right_conf = self._get_pupil_position_2d(right_eye_crop)
        
        # Convert to [-1.0, 1.0] relative range
        left_gaze_x = 2.0 * left_norm_x - 1.0
        left_gaze_y = 2.0 * left_norm_y - 1.0
        right_gaze_x = 2.0 * right_norm_x - 1.0
        right_gaze_y = 2.0 * right_norm_y - 1.0
        
        # Fuse observations using helper
        raw_x, raw_y, combined_conf = fuse_gaze_observations(
            left_gaze_x, left_gaze_y, left_conf,
            right_gaze_x, right_gaze_y, right_conf
        )
            
        # Calibration Coefficients
        a1, a2, a3 = 1.0, 0.0, 0.0
        b1, b2, b3 = 1.0, 0.0, 0.0
        if self.db_manager:
            try:
                cal = self.db_manager.get_gaze_calibration(camera_id)
                if cal and cal.get("calibration_status") == "CALIBRATED":
                    a1 = cal.get("a1", 1.0)
                    a2 = cal.get("a2", 0.0)
                    a3 = cal.get("a3", 0.0)
                    b1 = cal.get("b1", 1.0)
                    b2 = cal.get("b2", 0.0)
                    b3 = cal.get("b3", 0.0)
            except Exception:
                pass
                
        # Compensate for head pose
        gaze_x = a1 * raw_x + a2 * yaw + a3
        gaze_y = b1 * raw_y + b2 * pitch + b3
        
        gaze_x = max(-1.0, min(1.0, gaze_x))
        gaze_y = max(-1.0, min(1.0, gaze_y))
        
        return GazeObservation(
            timestamp=time.time(),
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            confidence=combined_conf,
            source="DEGRADED",
            yaw=yaw,
            pitch=pitch,
            roll=roll
        )

class FaceXGazeEstimator(GazeEstimator):
    """
    Full mode estimator using 98-point dense mesh for iris tracking and blink evaluation.
    """
    def __init__(self, db_manager: Any = None):
        self.db_manager = db_manager

    def estimate(self, frame: np.ndarray, face_detection: Any, landmarks: np.ndarray = None, pose: Tuple[float, float, float] = (0.0, 0.0, 0.0), camera_id: str = "0") -> GazeObservation:
        if landmarks is None or len(landmarks) != 98:
            # Fallback to degraded estimator if landmarks missing
            fallback = FallbackGazeEstimator(self.db_manager)
            return fallback.estimate(frame, face_detection, None, pose, camera_id)
            
        yaw, pitch, roll = pose
        
        # Calculate Eye Aspect Ratio (EAR) for blink detection
        left_eye = landmarks[60:68]
        right_eye = landmarks[68:76]
        
        def eye_aspect_ratio(eye_points):
            A = np.linalg.norm(eye_points[1] - eye_points[7])
            B = np.linalg.norm(eye_points[2] - eye_points[6])
            C = np.linalg.norm(eye_points[3] - eye_points[5])
            D = np.linalg.norm(eye_points[0] - eye_points[4])
            if D == 0: return 0.0
            return (A + B + C) / (3.0 * D)
            
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        is_blinking = ear < 0.20
        
        # Extract eye regions using precise landmarks
        lx = int(min(left_eye[:, 0]))
        rx = int(max(left_eye[:, 0]))
        ty = int(min(left_eye[:, 1]))
        by = int(max(left_eye[:, 1]))
        
        pad_x = int((rx - lx) * 0.1)
        pad_y = int((by - ty) * 0.2)
        
        fallback = FallbackGazeEstimator(self.db_manager)
        
        l_x1, l_y1 = max(0, lx - pad_x), max(0, ty - pad_y)
        l_x2, l_y2 = min(frame.shape[1], rx + pad_x), min(frame.shape[0], by + pad_y)
        left_eye_crop = frame[l_y1:l_y2, l_x1:l_x2]
        
        rx_min = int(min(right_eye[:, 0]))
        rx_max = int(max(right_eye[:, 0]))
        ry_min = int(min(right_eye[:, 1]))
        ry_max = int(max(right_eye[:, 1]))
        
        r_x1, r_y1 = max(0, rx_min - pad_x), max(0, ry_min - pad_y)
        r_x2, r_y2 = min(frame.shape[1], rx_max + pad_x), min(frame.shape[0], ry_max + pad_y)
        right_eye_crop = frame[r_y1:r_y2, r_x1:r_x2]
        
        left_norm_x, left_norm_y, left_conf = fallback._get_pupil_position_2d(left_eye_crop)
        right_norm_x, right_norm_y, right_conf = fallback._get_pupil_position_2d(right_eye_crop)
        
        left_gaze_x = 2.0 * left_norm_x - 1.0
        left_gaze_y = 2.0 * left_norm_y - 1.0
        right_gaze_x = 2.0 * right_norm_x - 1.0
        right_gaze_y = 2.0 * right_norm_y - 1.0
        
        raw_x, raw_y, combined_conf = fuse_gaze_observations(
            left_gaze_x, left_gaze_y, left_conf,
            right_gaze_x, right_gaze_y, right_conf
        )
            
        a1, a2, a3 = 1.0, 0.0, 0.0
        b1, b2, b3 = 1.0, 0.0, 0.0
        if self.db_manager:
            try:
                cal = self.db_manager.get_gaze_calibration(camera_id)
                if cal and cal.get("calibration_status") == "CALIBRATED":
                    a1 = cal.get("a1", 1.0)
                    a2 = cal.get("a2", 0.0)
                    a3 = cal.get("a3", 0.0)
                    b1 = cal.get("b1", 1.0)
                    b2 = cal.get("b2", 0.0)
                    b3 = cal.get("b3", 0.0)
            except Exception:
                pass
                
        gaze_x = a1 * raw_x + a2 * yaw + a3
        gaze_y = b1 * raw_y + b2 * pitch + b3
        
        gaze_x = max(-1.0, min(1.0, gaze_x))
        gaze_y = max(-1.0, min(1.0, gaze_y))
        
        obs = GazeObservation(
            timestamp=time.time(),
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            confidence=combined_conf,
            source="FULL",
            yaw=yaw,
            pitch=pitch,
            roll=roll
        )
        obs.is_blinking = is_blinking
        obs.blink_score = 0.0 if is_blinking else 1.0
        return obs

class GazeBlinkEstimator:
    """
    Backward-compatible wrapper class for legacy code.
    """
    def __init__(self, db_manager: Any = None):
        self.fallback_estimator = FallbackGazeEstimator(db_manager)
        self.facex_estimator = FaceXGazeEstimator(db_manager)
        
    def analyze(self, frame: np.ndarray, face_detection, dense_landmarks: np.ndarray = None, pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[bool, str, str]:
        if dense_landmarks is not None and len(dense_landmarks) == 98:
            obs = self.facex_estimator.estimate(frame, face_detection, dense_landmarks, pose)
            is_blinking = obs.is_blinking
        else:
            obs = self.fallback_estimator.estimate(frame, face_detection, None, pose)
            is_blinking = False
            
        horz = "CENTER"
        if obs.gaze_x < -0.3: horz = "LEFT"
        elif obs.gaze_x > 0.3: horz = "RIGHT"
        
        vert = "CENTER"
        if obs.gaze_y < -0.3: vert = "UP"
        elif obs.gaze_y > 0.3: vert = "DOWN"
        
        return is_blinking, horz, vert
