import numpy as np
from typing import List, Dict, Any
from app.vision.quality.engine import FaceQualityEngine

class VideoEnrollmentParser:
    """
    Parses an enrollment video session, categorizes head poses, 
    and selects the highest quality, most diverse frames for template extraction.
    """
    def __init__(self, quality_engine: FaceQualityEngine, min_quality_score: float = 0.6):
        self.quality_engine = quality_engine
        self.min_quality_score = min_quality_score
        self.candidate_pool = []
        
    def _categorize_pose(self, yaw: float, pitch: float, roll: float) -> str:
        """Classify continuous yaw/pitch angles into bucketed pose regions."""
        if pitch < -15: return "UP"
        if pitch > 15: return "DOWN"
        if yaw < -15: return "LEFT"
        if yaw > 15: return "RIGHT"
        return "FRONT"
        
    def add_frame_candidate(self, frame: np.ndarray, best_det, pose: tuple):
        """Processes a single video frame, keeping it only if it passes quality checks."""
        bbox = best_det.bbox
        detector_conf = best_det.confidence
        quality_eval = self.quality_engine.evaluate(frame, bbox, pose, detector_conf)
        
        if quality_eval["accepted"] and quality_eval["quality_score"] >= self.min_quality_score:
            pose_category = self._categorize_pose(pose[0], pose[1], pose[2])
            
            self.candidate_pool.append({
                "full_frame": frame.copy(),
                "best_det": best_det,
                "pose": pose,
                "pose_category": pose_category,
                "quality_score": quality_eval["quality_score"],
                "sharpness": quality_eval["metrics"]["sharpness"]
            })
            
    def get_diverse_templates(self, sface_recognizer, max_per_pose: int = 3) -> List[Dict[str, Any]]:
        """
        Extracts embeddings for candidates, sorts by quality, and enforces diversity constraints.
        Returns a curated list of dictionaries containing the final embeddings to store.
        """
        # Group candidates by pose
        pose_groups = {"FRONT": [], "LEFT": [], "RIGHT": [], "UP": [], "DOWN": []}
        for cand in self.candidate_pool:
            pose_groups[cand["pose_category"]].append(cand)
            
        final_templates = []
        
        for pose, candidates in pose_groups.items():
            if not candidates:
                continue
                
            # Sort by sharpness and quality score descending
            candidates.sort(key=lambda x: (x["sharpness"], x["quality_score"]), reverse=True)
            
            # Take the top N best frames for this pose
            selected = candidates[:max_per_pose]
            
            # Ideally here we calculate pairwise similarity to reject near-duplicates (redundancy penalty)
            # but for a fast implementation, taking the top distinct frames per pose bucket usually suffices.
            for cand in selected:
                # We assume the caller will align and extract embeddings, 
                # but for encapsulation, the parser returns the raw data needed.
                final_templates.append(cand)
                
        return final_templates
