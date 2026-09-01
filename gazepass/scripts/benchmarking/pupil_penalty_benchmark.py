import sys
import os
import cv2
import numpy as np
import glob

# Add roots to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.vision.detector.yunet import YuNetDetector
from app.liveness.active_challenge.gaze_estimator import FallbackGazeEstimator
import app.liveness.active_challenge.gaze_estimator as ge

def run_benchmark():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'detector', 'face_detection_yunet_2023mar.onnx'))
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print("[*] Initializing YuNet detector...")
    detector = YuNetDetector(model_path, conf_threshold=0.7)
    estimator = FallbackGazeEstimator()
    
    # Locate all jpg frames in the genuine dataset
    dataset_pattern = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets', 'recognition', 'genuine', '**', '*.jpg'))
    image_paths = glob.glob(dataset_pattern, recursive=True)
    if not image_paths:
        print("Error: No genuine images found for benchmarking.")
        return
        
    print(f"[*] Found {len(image_paths)} genuine images. Loading and running detections...")
    
    # Run face detection and crop eyes once to save time
    detections_and_crops = []
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        dets = detector.detect(frame)
        if not dets:
            continue
        best_det = max(dets, key=lambda d: d.confidence)
        
        # Crop left and right eyes
        landmarks = best_det.landmarks
        if landmarks is not None and len(landmarks) >= 2:
            right_pt = landmarks[0]
            left_pt  = landmarks[1]
            eye_dist = np.linalg.norm(np.array(left_pt) - np.array(right_pt))
            crop_w = int(eye_dist * 0.35)
            crop_h = int(eye_dist * 0.25)
            
            # Left Eye Crop
            lx, ly = int(left_pt[0]), int(left_pt[1])
            left_crop = frame[max(0, ly-crop_h):min(frame.shape[0], ly+crop_h),
                              max(0, lx-crop_w):min(frame.shape[1], lx+crop_w)]
                              
            # Right Eye Crop
            rx, ry = int(right_pt[0]), int(right_pt[1])
            right_crop = frame[max(0, ry-crop_h):min(frame.shape[0], ry+crop_h),
                               max(0, rx-crop_w):min(frame.shape[1], rx+crop_w)]
                               
            if left_crop.size > 0 and right_crop.size > 0:
                detections_and_crops.append((left_crop, right_crop))
                
    print(f"[*] Pre-processed {len(detections_and_crops)} valid face detections with eyes.")
    
    penalties = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    results = []
    
    for penalty in penalties:
        # Mock load_config to return current penalty
        def make_mock(p):
            def mock():
                conf = {
                    'center_distance_penalty': p
                }
                return conf
            return mock
        ge.load_config = make_mock(penalty)
        
        detected_pupils = 0
        total_eyes = 0
        gaze_xs = []
        gaze_ys = []
        confidences = []
        
        for left_crop, right_crop in detections_and_crops:
            # Left eye
            lx, ly, l_conf = estimator._get_pupil_position_2d(left_crop)
            # Right eye
            rx, ry, r_conf = estimator._get_pupil_position_2d(right_crop)
            
            total_eyes += 2
            if l_conf > 0.0:
                detected_pupils += 1
                gaze_xs.append(lx)
                gaze_ys.append(ly)
                confidences.append(l_conf)
            if r_conf > 0.0:
                detected_pupils += 1
                gaze_xs.append(rx)
                gaze_ys.append(ry)
                confidences.append(r_conf)
                
        detection_rate = (detected_pupils / total_eyes) * 100 if total_eyes > 0 else 0.0
        mean_conf = np.mean(confidences) if confidences else 0.0
        std_x = np.std(gaze_xs) if gaze_xs else 0.0
        std_y = np.std(gaze_ys) if gaze_ys else 0.0
        
        results.append({
            "penalty": penalty,
            "detection_rate": detection_rate,
            "mean_confidence": mean_conf,
            "std_x": std_x,
            "std_y": std_y
        })
        
    print("\n" + "="*80)
    print("                 PUPIL PENALTY BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Penalty':<10} | {'Detection Rate (%)':<20} | {'Mean Confidence':<18} | {'Std Gaze X':<12} | {'Std Gaze Y':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['penalty']:<10.2f} | {r['detection_rate']:<20.2f} | {r['mean_confidence']:<18.4f} | {r['std_x']:<12.4f} | {r['std_y']:<12.4f}")
    print("="*80)
    
if __name__ == "__main__":
    run_benchmark()
