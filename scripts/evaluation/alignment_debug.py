import os
import sys
import cv2
import json
import numpy as np

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'gazepass'))

from app.vision.detector.yunet import YuNetDetector
from app.vision.recognition.sface import SFaceRecognizer

def main():
    print("="*50)
    print("  Gazepass V2: Alignment Debugger")
    print("="*50)

    dataset_root = os.path.join(BASE_DIR, 'datasets', 'recognition')
    if not os.path.exists(dataset_root):
        print(f"[!] Dataset directory not found: {dataset_root}")
        print("[!] Please capture some images first using capture.py")
        return

    # Load models
    models_dir = os.path.join(BASE_DIR, 'gazepass', 'models')
    detector = YuNetDetector(os.path.join(models_dir, 'detector', 'face_detection_yunet_2023mar.onnx'))
    sface = SFaceRecognizer(os.path.join(models_dir, 'recognition', 'face_recognition_sface_2021dec.onnx'))

    out_dir = os.path.join(BASE_DIR, 'reports', 'alignment_debug')
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Scanning {dataset_root} for images...")
    
    count = 0
    for root, _, files in os.walk(dataset_root):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, f)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                # Detect
                dets = detector.detect(frame)
                if not dets:
                    print(f"  [WARN] No face detected in {f}")
                    continue
                
                # Take largest face
                best_det = max(dets, key=lambda d: d.bbox[2] * d.bbox[3])
                
                # Draw landmarks on original
                vis_frame = frame.copy()
                for i in range(5):
                    pt = (int(best_det.landmarks[i][0]), int(best_det.landmarks[i][1]))
                    cv2.circle(vis_frame, pt, 2, (0, 255, 0), -1)
                
                # Align using SFace internal logic
                aligned_face = sface.align(frame, best_det)
                
                # Save contact sheet side-by-side
                # Resize original to match aligned height (112) for visualization
                h, w = aligned_face.shape[:2]
                orig_h, orig_w = vis_frame.shape[:2]
                scale = h / float(orig_h)
                resized_vis = cv2.resize(vis_frame, (int(orig_w * scale), h))
                
                # Pad if needed, or just stack horizontally
                contact_sheet = np.hstack((resized_vis, aligned_face))
                
                save_path = os.path.join(out_dir, f"align_{count:04d}_{f}")
                cv2.imwrite(save_path, contact_sheet)
                print(f"  [+] Saved {save_path}")
                count += 1

    print(f"[*] Alignment debug complete. Generated {count} contact sheets in {out_dir}")
    print("[*] Inspect these images to ensure the eyes are horizontal and the chin/forehead cropping is consistent.")

if __name__ == '__main__':
    main()
