import cv2
import numpy as np
from typing import Tuple, List

class FaceXLandmarker:
    """
    Dense 98-point (or 106-point) facial landmark extractor.
    Used for gaze tracking, head pose estimation, and face alignment.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        # In a real deployment, this would load the specific FaceX ONNX model
        # using cv2.dnn.readNetFromONNX or onnxruntime.
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
        except Exception as e:
            print(f"[!] Warning: Could not load FaceX model from {model_path}: {e}")
            self.net = None
            
    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        # Standardize to 112x112 for typical landmark models
        resized = cv2.resize(face_crop, (112, 112))
        blob = cv2.dnn.blobFromImage(resized, 1.0 / 255.0, (112, 112), (0, 0, 0), swapRB=True, crop=False)
        return blob

    def get_landmarks(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract dense landmarks from the face bounding box.
        Returns array of (x, y) coordinates.
        """
        if self.net is None:
            # Fallback/stub if model isn't downloaded yet
            return None
            
        x, y, w, h = bbox
        
        # Add padding
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return np.zeros((98, 2), dtype=np.float32)
            
        blob = self._preprocess(face_crop)
        self.net.setInput(blob)
        out = self.net.forward()
        
        # Assuming output is shape (1, 196) for 98 points (x, y)
        out = out.reshape(-1, 2)
        
        # Scale back to original frame coordinates
        scale_x = (x2 - x1) / 112.0
        scale_y = (y2 - y1) / 112.0
        
        landmarks = []
        for pt in out:
            lx = int(pt[0] * 112 * scale_x) + x1
            ly = int(pt[1] * 112 * scale_y) + y1
            landmarks.append([lx, ly])
            
        return np.array(landmarks, dtype=np.float32)
