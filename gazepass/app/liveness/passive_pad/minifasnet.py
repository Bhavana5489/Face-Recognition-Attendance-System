import cv2
import numpy as np

class MiniFASNetEnsemble:
    """
    Passive Presentation Attack Detection (PAD).
    Ensemble of MiniFASNetV1SE and MiniFASNetV2 models using ONNX Runtime or OpenCV DNN.
    """
    def __init__(self, model_v1_path: str, model_v2_path: str):
        self.model_v1_path = model_v1_path
        self.model_v2_path = model_v2_path
        
        try:
            self.net_v1 = cv2.dnn.readNetFromONNX(self.model_v1_path)
            self.net_v2 = cv2.dnn.readNetFromONNX(self.model_v2_path)
            self.is_loaded = True
        except Exception as e:
            print(f"[!] Warning: Could not load MiniFASNet models. {e}")
            self.is_loaded = False
            
    def _get_new_box(self, src_w, src_h, bbox, scale):
        x, y, box_w, box_h = bbox
        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))
        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y
        
        left_top_x = center_x-new_width/2
        left_top_y = center_y-new_height/2
        right_bottom_x = center_x+new_width/2
        right_bottom_y = center_y+new_height/2
        
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1
        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1
            
        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def _preprocess(self, frame: np.ndarray, bbox: tuple, scale: float) -> np.ndarray:
        src_h, src_w = frame.shape[:2]
        x1, y1, x2, y2 = self._get_new_box(src_w, src_h, bbox, scale)
        
        crop = frame[y1:y2+1, x1:x2+1]
        if crop.size == 0:
            return None
            
        crop = cv2.resize(crop, (80, 80))
        # MiniFASNet original test.py doesn't use swapRB.
        # CRITICAL DISCOVERY: Minivision's custom ToTensor in functional.py commented out .div(255)!
        # The PyTorch model was trained on 0-255 inputs, not 0-1!
        blob = cv2.dnn.blobFromImage(crop, 1.0, (80, 80), (0, 0, 0), swapRB=False, crop=False)
        return blob
        
    def evaluate(self, frame: np.ndarray, bbox: tuple) -> float:
        """
        Evaluates the liveness of the face.
        Returns a liveness probability [0.0 to 1.0]. 1.0 = Real, 0.0 = Spoof.
        """
        _, _, fusion_score = self.evaluate_detailed(frame, bbox)
        return fusion_score

    def evaluate_detailed(self, frame: np.ndarray, bbox: tuple) -> tuple:
        """
        Evaluates the liveness of the face using both V1SE and V2 models.
        Returns (score_v1, score_v2, fusion_score).
        """
        if not self.is_loaded:
            return 1.0, 1.0, 1.0
            
        # Inference V1SE (Scale 4.0)
        blob_v1 = self._preprocess(frame, bbox, 4.0)
        if blob_v1 is None: return 0.0, 0.0, 0.0
        self.net_v1.setInput(blob_v1)
        out_v1 = self.net_v1.forward()
        prob_v1 = np.exp(out_v1) / np.sum(np.exp(out_v1))
        score_v1 = float(prob_v1[0][1])
        
        # Inference V2 (Scale 2.7)
        blob_v2 = self._preprocess(frame, bbox, 2.7)
        if blob_v2 is None: return score_v1, 0.0, score_v1 / 2.0
        self.net_v2.setInput(blob_v2)
        out_v2 = self.net_v2.forward()
        prob_v2 = np.exp(out_v2) / np.sum(np.exp(out_v2))
        score_v2 = float(prob_v2[0][1])
        
        fusion_score = (score_v1 + score_v2) / 2.0
        return score_v1, score_v2, fusion_score
