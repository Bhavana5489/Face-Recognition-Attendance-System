import os
import cv2
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any

class TemporalPADCollector:
    """
    Data Collection Architecture for Temporal PAD (Phase 13).
    Saves temporal sequences of face crops to train a local 3D-CNN or RNN replay detector.
    """
    def __init__(self, storage_dir: str = "data/temporal_pad"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
    def save_sequence(self, track_id: str, frames: list, bboxes: list, metadata: Dict[str, Any]):
        """
        Saves a contiguous sequence of face crops for a tracked individual.
        frames: List of raw full-resolution frames
        bboxes: List of bounding boxes corresponding to frames
        """
        if len(frames) < 10:
            # Only save sequences long enough to capture temporal dynamics
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sequence_dir = os.path.join(self.storage_dir, f"{timestamp}_track_{track_id}")
        os.makedirs(sequence_dir, exist_ok=True)
        
        sequence_metadata = {
            "track_id": track_id,
            "timestamp": timestamp,
            "num_frames": len(frames),
            "label": metadata.get("label", "UNKNOWN"), # 'REAL' or 'SPOOF' if known during testing
            "bboxes": bboxes
        }
        
        with open(os.path.join(sequence_dir, "metadata.json"), "w") as f:
            json.dump(sequence_metadata, f, indent=4)
            
        for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
            x, y, w, h = bbox
            # Save cropped face with some context
            pad_x = int(w * 0.2)
            pad_y = int(h * 0.2)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                cv2.imwrite(os.path.join(sequence_dir, f"frame_{i:04d}.jpg"), crop)
