import cv2
import numpy as np
from typing import Dict, Any, List

class FaceQualityEngine:
    """
    Evaluates the biometric quality of a detected face.
    Scores sharpness, brightness, face size, pose, and occlusion.
    """
    def __init__(self, min_face_size: int = 80, optimal_sharpness: float = 40.0):
        self.min_face_size = min_face_size
        self.optimal_sharpness = optimal_sharpness
        
    def _calculate_sharpness(self, face_crop_gray: np.ndarray) -> float:
        return cv2.Laplacian(face_crop_gray, cv2.CV_64F).var()
        
    def _calculate_brightness(self, face_crop_gray: np.ndarray) -> float:
        return np.mean(face_crop_gray)
        
    def evaluate(self, frame: np.ndarray, bbox: tuple, pose: tuple, detector_conf: float) -> Dict[str, Any]:
        """
        Evaluate frame quality.
        pose: (yaw, pitch, roll)
        """
        x, y, w, h = bbox
        
        # 1. Size score (0 to 1)
        size_score = min(1.0, float(w) / self.min_face_size)
        
        # 2. Extract face crop for texture metrics
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return {"accepted": False, "quality_score": 0.0, "reasons": ["empty_crop"]}
            
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # 3. Sharpness
        sharpness = self._calculate_sharpness(gray)
        sharpness_score = min(1.0, sharpness / self.optimal_sharpness)
        
        # 4. Brightness (penalize too dark or too washed out)
        brightness = self._calculate_brightness(gray)
        if 40 < brightness < 210:
            brightness_score = 1.0
        else:
            brightness_score = max(0.0, 1.0 - abs(125 - brightness) / 125.0)
            
        # 5. Pose score
        yaw, pitch, roll = pose
        # Penalize extreme angles (> 45 degrees)
        yaw_penalty = max(0.0, (abs(yaw) - 20) / 45.0)
        pitch_penalty = max(0.0, (abs(pitch) - 20) / 45.0)
        pose_score = max(0.0, 1.0 - (yaw_penalty + pitch_penalty))
        
        # Aggregate Quality Score
        total_score = (size_score * 0.2) + (sharpness_score * 0.4) + (brightness_score * 0.2) + (pose_score * 0.2)
        
        reasons = []
        if size_score < 0.5: reasons.append("face_too_small")
        if sharpness_score < 0.2: reasons.append(f"motion_blur({sharpness:.1f})")
        if brightness_score < 0.3: reasons.append("poor_exposure")
        if pose_score < 0.05: reasons.append("extreme_pose")
        if detector_conf < 0.7: reasons.append("low_detector_confidence")
        
        accepted = len(reasons) == 0
        
        return {
            "accepted": accepted,
            "quality_score": float(total_score),
            "reasons": reasons,
            "metrics": {
                "sharpness": float(sharpness),
                "brightness": float(brightness),
                "yaw": float(yaw),
                "pitch": float(pitch)
            }
        }
