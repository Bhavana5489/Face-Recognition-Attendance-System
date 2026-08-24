import os
import sys
import cv2
import json
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'gazepass'))

from app.vision.detector.yunet import YuNetDetector
from app.vision.recognition.sface import SFaceRecognizer

def main():
    print("="*50)
    print("  Gazepass V2: Recognition Benchmark")
    print("="*50)

    dataset_root = os.path.join(BASE_DIR, 'datasets', 'recognition')
    if not os.path.exists(dataset_root):
        print(f"[!] Dataset directory not found: {dataset_root}")
        print("[!] Please capture some images first using capture.py")
        return

    models_dir = os.path.join(BASE_DIR, 'gazepass', 'models')
    detector = YuNetDetector(os.path.join(models_dir, 'detector', 'face_detection_yunet_2023mar.onnx'))
    sface = SFaceRecognizer(os.path.join(models_dir, 'recognition', 'face_recognition_sface_2021dec.onnx'))

    # Load embeddings
    print("[*] Extracting embeddings...")
    embeddings = {} # student_id -> list of embeddings
    
    for category in ['enrollment', 'genuine', 'impostor']:
        cat_path = os.path.join(dataset_root, category)
        if not os.path.exists(cat_path): continue
        
        for student_id in os.listdir(cat_path):
            stu_path = os.path.join(cat_path, student_id)
            if not os.path.isdir(stu_path): continue
            
            if student_id not in embeddings:
                embeddings[student_id] = {'enrollment': [], 'test': []}
                
            for root, _, files in os.walk(stu_path):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(root, f)
                        frame = cv2.imread(img_path)
                        dets = detector.detect(frame)
                        if dets:
                            best_det = max(dets, key=lambda d: d.bbox[2] * d.bbox[3])
                            aligned = sface.align(frame, best_det)
                            emb = sface.get_embedding(aligned)
                            
                            if category == 'enrollment':
                                embeddings[student_id]['enrollment'].append(emb)
                            else:
                                embeddings[student_id]['test'].append({'emb': emb, 'is_genuine': category == 'genuine'})

    if not embeddings:
        print("[!] No embeddings extracted. Check dataset.")
        return

    print("[*] Running comparisons...")
    genuine_scores = []
    impostor_scores = []
    
    for student_id, data in embeddings.items():
        gallery = data['enrollment']
        if not gallery: continue
        
        # Test Genuine (same student test vs enrollment)
        for test_item in data['test']:
            if test_item['is_genuine']:
                # Compare against all gallery embeddings and take max similarity
                sims = [np.dot(test_item['emb'], g.T)[0][0] for g in gallery]
                genuine_scores.append(max(sims))
                
        # Test Impostors (this student's gallery vs other students' test embeddings)
        for other_student, other_data in embeddings.items():
            if student_id == other_student: continue
            for other_test in other_data['test']:
                sims = [np.dot(other_test['emb'], g.T)[0][0] for g in gallery]
                impostor_scores.append(max(sims))

    reports_dir = os.path.join(BASE_DIR, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    report = {
        "genuine_comparisons": len(genuine_scores),
        "impostor_comparisons": len(impostor_scores),
        "metrics": {}
    }
    
    if genuine_scores:
        report["metrics"]["genuine"] = {
            "mean": float(np.mean(genuine_scores)),
            "median": float(np.median(genuine_scores)),
            "min": float(np.min(genuine_scores)),
            "max": float(np.max(genuine_scores))
        }
        
    if impostor_scores:
        report["metrics"]["impostor"] = {
            "mean": float(np.mean(impostor_scores)),
            "median": float(np.median(impostor_scores)),
            "min": float(np.min(impostor_scores)),
            "max": float(np.max(impostor_scores))
        }
        
    report_path = os.path.join(reports_dir, 'recognition_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"[*] Benchmark complete. Report saved to {report_path}")
    print(json.dumps(report, indent=4))

if __name__ == '__main__':
    main()
