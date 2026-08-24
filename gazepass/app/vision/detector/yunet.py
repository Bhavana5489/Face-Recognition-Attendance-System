import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    landmarks: List[Tuple[int, int]]  # 5 landmarks: right eye, left eye, nose tip, right mouth, left mouth

class YuNetDetector:
    """
    OpenCV Zoo YuNet Face Detector implementation.
    Lightweight, fast CPU/GPU face detection with 5-point landmarks.
    """
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (320, 320), conf_threshold: float = 0.8, nms_threshold: float = 0.3):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Initialize the detector
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=self.input_size,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000
        )
        
    def set_input_size(self, size: Tuple[int, int]):
        """Update input size if frame dimensions change."""
        if self.input_size != size:
            self.input_size = size
            self.detector.setInputSize(size)
            
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a BGR frame.
        Returns a list of FaceDetection objects.
        """
        h, w = frame.shape[:2]
        self.set_input_size((w, h))
        
        # YuNet expects BGR format directly
        _, faces = self.detector.detect(frame)
        
        detections = []
        if faces is not None:
            for face in faces:
                # face is an array of 15 elements:
                # 0-3: bbox x, y, w, h
                # 4-13: 5 landmarks (x, y)
                # 14: confidence score
                
                box_x, box_y, box_w, box_h = map(int, face[0:4])
                
                # Constrain bbox to frame boundaries
                box_x = max(0, box_x)
                box_y = max(0, box_y)
                box_w = min(w - box_x, box_w)
                box_h = min(h - box_y, box_h)
                
                confidence = float(face[-1])
                
                landmarks = []
                for i in range(5):
                    lx = int(face[4 + (i * 2)])
                    ly = int(face[4 + (i * 2) + 1])
                    landmarks.append((lx, ly))
                    
                detections.append(FaceDetection(
                    bbox=(box_x, box_y, box_w, box_h),
                    confidence=confidence,
                    landmarks=landmarks
                ))
                
        return detections
