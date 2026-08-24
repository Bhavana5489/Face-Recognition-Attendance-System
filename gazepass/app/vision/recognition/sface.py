import cv2
import numpy as np
from typing import Tuple

class SFaceRecognizer:
    """
    OpenCV Zoo SFace ONNX model integration.
    Generates 128-d normalized embeddings.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        try:
            self.recognizer = cv2.FaceRecognizerSF.create(self.model_path, "")
        except Exception as e:
            print(f"[!] Warning: Could not load SFace model from {model_path}: {e}")
            self.recognizer = None

    def align(self, frame: np.ndarray, face_detection) -> np.ndarray:
        """
        Aligns the face based on the 5 YuNet landmarks for SFace input.
        face_detection is expected to be a FaceDetection object (or tuple of 15 elements as required by OpenCV).
        """
        if self.recognizer is None:
            return np.zeros((112, 112, 3), dtype=np.uint8)
            
        # cv2.FaceRecognizerSF.alignCrop expects the exact output format of YuNet.
        # Construct the 15-element array: [x, y, w, h, lx, ly, rx, ry, nx, ny, lmx, lmy, rmx, rmy, conf]
        face_arr = np.zeros(15, dtype=np.float32)
        face_arr[0:4] = face_detection.bbox
        
        # Landmarks: Right Eye, Left Eye, Nose, Right Mouth, Left Mouth
        # YuNet outputs: 
        # 0: Right Eye
        # 1: Left Eye
        # 2: Nose
        # 3: Right Mouth
        # 4: Left Mouth
        for i, pt in enumerate(face_detection.landmarks):
            face_arr[4 + i * 2] = pt[0]
            face_arr[4 + i * 2 + 1] = pt[1]
            
        face_arr[14] = face_detection.confidence
        
        aligned_face = self.recognizer.alignCrop(frame, face_arr)
        return aligned_face

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extracts the 128-dimensional embedding from the aligned face.
        Returns L2 normalized embedding.
        """
        if self.recognizer is None:
            return np.zeros((1, 128), dtype=np.float32)
            
        feature = self.recognizer.feature(aligned_face)
        # SFace outputs a (1, 128) array
        # Normalize the embedding using L2 norm
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
            
        return feature
        
    def match(self, embed1: np.ndarray, embed2: np.ndarray, dis_type: int = cv2.FaceRecognizerSF_FR_COSINE) -> float:
        """
        Calculates similarity between two embeddings.
        Cosine similarity returns [-1, 1], where 1 is identical.
        """
        if self.recognizer is None:
            return 0.0
            
        # OpenCV natively supports L2 and Cosine distance evaluation
        score = self.recognizer.match(embed1, embed2, dis_type)
        return score
