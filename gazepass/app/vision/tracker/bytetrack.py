import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time

@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    age: int = 0
    time_since_update: int = 0
    hits: int = 1
    state: str = 'Tracked'  # Tracked, Lost, Removed

class ByteTracker:
    """
    Lightweight pure-Python implementation of ByteTrack logic for Face Tracking.
    It associates high-confidence detections first, then low-confidence detections
    to recover partially occluded or blurred faces.
    """
    def __init__(self, track_buffer: int = 30, track_thresh: float = 0.5, high_thresh: float = 0.6, match_thresh: float = 0.8):
        self.track_buffer = track_buffer
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        
        self.tracks: List[Track] = []
        self.next_id = 1
        
    def _iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0
            
        box1_area = bbox1[2] * bbox1[3]
        box2_area = bbox2[2] * bbox2[3]
        
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou
        
    def _iou_distance_matrix(self, tracks: List[Track], detections: List[Tuple]):
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1.0 - self._iou(track.bbox, det[0])
        return cost_matrix
        
    def _linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
            
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches, unmatched_a, unmatched_b = [], [], []
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > thresh:
                unmatched_a.append(r)
                unmatched_b.append(c)
            else:
                matches.append((r, c))
                
        # Add remaining unmatched rows/cols
        unmatched_a.extend(list(set(range(cost_matrix.shape[0])) - set(row_ind)))
        unmatched_b.extend(list(set(range(cost_matrix.shape[1])) - set(col_ind)))
        
        return np.array(matches), unmatched_a, unmatched_b

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float]]) -> List[Track]:
        """
        detections: list of (bbox, confidence)
        Returns list of active Tracks
        """
        high_dets = [d for d in detections if d[1] >= self.high_thresh]
        low_dets = [d for d in detections if self.track_thresh <= d[1] < self.high_thresh]
        
        # 1. First association with high score detections
        cost_matrix = self._iou_distance_matrix(self.tracks, high_dets)
        matches_a, un_tracks_a, un_dets_a = self._linear_assignment(cost_matrix, self.match_thresh)
        
        for r, c in matches_a:
            self.tracks[r].bbox = high_dets[c][0]
            self.tracks[r].confidence = high_dets[c][1]
            self.tracks[r].time_since_update = 0
            self.tracks[r].hits += 1
            self.tracks[r].state = 'Tracked'
            
        # 2. Second association with low score detections
        remaining_tracks = [self.tracks[i] for i in un_tracks_a if self.tracks[i].state == 'Tracked']
        cost_matrix = self._iou_distance_matrix(remaining_tracks, low_dets)
        matches_b, un_tracks_b, un_dets_b = self._linear_assignment(cost_matrix, 0.5)
        
        for r, c in matches_b:
            orig_idx = un_tracks_a[r]
            self.tracks[orig_idx].bbox = low_dets[c][0]
            self.tracks[orig_idx].confidence = low_dets[c][1]
            self.tracks[orig_idx].time_since_update = 0
            self.tracks[orig_idx].state = 'Tracked'
            
        # Unmatched tracks from second association become Lost
        for r in un_tracks_b:
            orig_idx = un_tracks_a[r]
            if self.tracks[orig_idx].state == 'Tracked':
                self.tracks[orig_idx].state = 'Lost'
                
        # Unmatched tracks from first association that were already lost
        for r in un_tracks_a:
            if self.tracks[r].state == 'Lost':
                self.tracks[r].time_since_update += 1
                
        # 3. Deal with unmatched high_dets (New Tracks)
        for c in un_dets_a:
            new_track = Track(track_id=self.next_id, bbox=high_dets[c][0], confidence=high_dets[c][1])
            self.tracks.append(new_track)
            self.next_id += 1
            
        # 4. Remove old lost tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.track_buffer]
        
        for t in self.tracks:
            t.age += 1
            
        return [t for t in self.tracks if t.state == 'Tracked']
