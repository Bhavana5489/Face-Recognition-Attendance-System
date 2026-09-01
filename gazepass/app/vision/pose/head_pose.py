import cv2
import numpy as np
from typing import Tuple

class HeadPoseEstimator:
    """
    Geometric Head Pose Estimator using solvePnP.
    """
    def __init__(self, frame_size: Tuple[int, int]):
        self.frame_width, self.frame_height = frame_size
        
        # Approximate 3D facial model points
        # Nose tip, Chin, Left Eye L, Right Eye R, Left Mouth, Right Mouth
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)
        
        # Fake camera internals
        focal_length = self.frame_width
        center = (self.frame_width / 2, self.frame_height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
    def estimate(self, image_points: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimate head pose (yaw, pitch, roll) in degrees from 2D landmarks.
        image_points should contain the 6 key points corresponding to self.model_points.
        (If using YuNet 5-points, we interpolate or use a 5-point 3D model).
        """
        if len(image_points) != len(self.model_points):
            # Fallback if using 5-point landmarks (YuNet)
            if len(image_points) == 5:
                # Approximate chin from nose and eyes (very rough)
                chin_x = image_points[2][0]
                chin_y = image_points[2][1] + (image_points[2][1] - (image_points[0][1]+image_points[1][1])/2)
                image_points = np.array([
                    image_points[2],         # Nose
                    [chin_x, chin_y],        # Fake Chin
                    image_points[1],         # Left Eye (2D Image Left, Person's Right) -> matches negative X in 3D
                    image_points[0],         # Right Eye
                    image_points[4],         # Left Mouth
                    image_points[3]          # Right Mouth
                ], dtype=np.float32)
            else:
                return 0.0, 0.0, 0.0
                
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
            
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]
        
        return float(yaw), float(pitch), float(roll)
        
    def estimate_facex(self, dense_landmarks: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimate head pose using FaceX 98-point dense mesh.
        Maps to the 6 3D model points for solvePnP.
        """
        if dense_landmarks is None or len(dense_landmarks) != 98:
            return 0.0, 0.0, 0.0
            
        # Standard WFLW 98-point mapping
        # 54: Nose tip, 16: Chin, 60: Left eye outer, 72: Right eye outer
        # 76: Left mouth corner, 82: Right mouth corner
        image_points = np.array([
            dense_landmarks[54],  # Nose tip
            dense_landmarks[16],  # Chin
            dense_landmarks[60],  # Left eye corner
            dense_landmarks[72],  # Right eye corner
            dense_landmarks[76],  # Left mouth corner
            dense_landmarks[82]   # Right mouth corner
        ], dtype=np.float32)
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
            
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]
        
        return float(yaw), float(pitch), float(roll)
