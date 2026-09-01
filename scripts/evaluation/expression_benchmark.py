import os
import sys
import cv2
import json
import sqlite3
import numpy as np

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'gazepass'))

from app.database.db_core import DatabaseManager
from app.vision.detector.yunet import YuNetDetector
from app.vision.recognition.sface import SFaceRecognizer
from app.vision.quality.engine import FaceQualityEngine
from app.vision.pose.head_pose import HeadPoseEstimator

def enroll_student(db_manager, detector, sface, quality_engine, student_id, student_name, enrollment_dir):
    print(f"[*] Enrolling student {student_name} ({student_id})...")
    
    # Insert student if not exists
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM students WHERE id = ?", (student_id,))
    row = cursor.fetchone()
    if not row:
        cursor.execute(
            "INSERT INTO students (id, admission_number, name) VALUES (?, ?, ?)",
            (student_id, f"ADM-{student_id}", student_name)
        )
        conn.commit()
    conn.close()

    # Clear old templates to ensure clean state
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM biometric_templates WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()

    # Find frames
    templates_count = 0
    pose_estimator = None
    
    for root, _, files in os.walk(enrollment_dir):
        for f in sorted(files):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, f)
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                h_frame, w_frame = frame.shape[:2]
                if pose_estimator is None:
                    pose_estimator = HeadPoseEstimator((w_frame, h_frame))
                    
                dets = detector.detect(frame)
                if not dets:
                    continue
                
                best_det = max(dets, key=lambda d: d.bbox[2] * d.bbox[3])
                
                # Estimate pose
                yaw, pitch, roll = pose_estimator.estimate(best_det.landmarks)
                
                # Evaluate quality
                quality_eval = quality_engine.evaluate(frame, best_det.bbox, (yaw, pitch, roll), best_det.confidence)
                
                if quality_eval["accepted"]:
                    aligned = sface.align(frame, best_det)
                    emb = sface.get_embedding(aligned)
                    
                    # Validate embedding
                    norm = np.linalg.norm(emb)
                    if not (0.95 <= norm <= 1.05) or np.isnan(emb).any() or np.isinf(emb).any():
                        print(f"[!] Warning: Invalid embedding norm={norm} for {f}")
                        continue
                        
                    db_manager.save_template(student_id, emb, {
                        "model_id": "sface",
                        "model_version": "2021dec",
                        "quality_score": quality_eval["quality_score"],
                        "yaw": yaw,
                        "pitch": pitch,
                        "roll": roll,
                        "source": "enrollment"
                    })
                    templates_count += 1

    print(f"[*] Enrolled {student_name} with {templates_count} templates.")
    return templates_count

def run_benchmark():
    print("="*60)
    print("  Gazepass V2: Expression Similarity Benchmark")
    print("="*60)

    db_path = os.path.join(BASE_DIR, 'gazepass', 'gazepass.db')
    # Backup DB if it exists and has content, but it's 0 bytes so no need for safety warning
    db_manager = DatabaseManager(db_path)

    models_dir = os.path.join(BASE_DIR, 'gazepass', 'models')
    detector = YuNetDetector(os.path.join(models_dir, 'detector', 'face_detection_yunet_2023mar.onnx'))
    sface = SFaceRecognizer(os.path.join(models_dir, 'recognition', 'face_recognition_sface_2021dec.onnx'))
    quality_engine = FaceQualityEngine()

    # Enroll TEST-001
    enrollment_dir = os.path.join(BASE_DIR, 'datasets', 'recognition', 'enrollment', 'TEST-001')
    template_count = enroll_student(db_manager, detector, sface, quality_engine, "TEST-001", "TEST Student", enrollment_dir)

    if template_count == 0:
        print("[!] No templates enrolled. Benchmark aborted.")
        return

    # Load templates from DB
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT embedding FROM biometric_templates WHERE student_id = 'TEST-001' AND template_status = 'ACTIVE'")
    rows = cursor.fetchall()
    conn.close()
    
    templates = [np.frombuffer(row['embedding'], dtype=np.float32) for row in rows]
    print(f"[*] Loaded {len(templates)} templates from database.")

    # Load test frames (genuine vs impostor)
    genuine_dir = os.path.join(BASE_DIR, 'datasets', 'recognition', 'genuine', 'TEST-001')
    impostor_dir = os.path.join(BASE_DIR, 'datasets', 'recognition', 'impostor', 'TEST-002')

    results = []

    # Helper function to process directory
    def process_test_dir(directory, is_genuine, expression_label):
        pose_estimator = None
        for root, _, files in os.walk(directory):
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, f)
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                    
                    h_frame, w_frame = frame.shape[:2]
                    if pose_estimator is None:
                        pose_estimator = HeadPoseEstimator((w_frame, h_frame))
                        
                    dets = detector.detect(frame)
                    if not dets:
                        continue
                    
                    best_det = max(dets, key=lambda d: d.bbox[2] * d.bbox[3])
                    
                    # Estimate pose
                    yaw, pitch, roll = pose_estimator.estimate(best_det.landmarks)
                    quality_eval = quality_engine.evaluate(frame, best_det.bbox, (yaw, pitch, roll), best_det.confidence)
                    
                    aligned = sface.align(frame, best_det)
                    emb = sface.get_embedding(aligned).flatten()
                    
                    # Compare to all templates
                    sims = [float(np.dot(emb, t)) for t in templates]
                    
                    if sims:
                        top1_sim = max(sims)
                        # We don't have second student templates, so top2_similarity can be calculated 
                        # as second best template similarity, or we use a constant if only 1 template
                        sorted_sims = sorted(sims, reverse=True)
                        top2_sim = sorted_sims[1] if len(sorted_sims) > 1 else 0.0
                        margin = top1_sim - top2_sim
                        
                        # Decide identity using T_rec = 0.70
                        identity_decision = "MATCH" if top1_sim >= 0.70 else "UNKNOWN"
                        
                        results.append({
                            "file": f,
                            "is_genuine": is_genuine,
                            "expression": expression_label,
                            "top1_similarity": top1_sim,
                            "top2_similarity": top2_sim,
                            "margin": margin,
                            "identity_decision": identity_decision,
                            "quality_score": quality_eval["quality_score"],
                            "template_count": len(templates)
                        })

    print("[*] Processing genuine test frames...")
    process_test_dir(genuine_dir, is_genuine=True, expression_label="variable_expressions")

    print("[*] Processing impostor test frames...")
    process_test_dir(impostor_dir, is_genuine=False, expression_label="impostor")

    # Generate report
    reports_dir = os.path.join(BASE_DIR, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, 'recognition_recovery.json')

    with open(report_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"[*] Benchmark complete. Report saved to {report_path}")
    
    # Summarize stats
    gen_sims = [r["top1_similarity"] for r in results if r["is_genuine"]]
    imp_sims = [r["top1_similarity"] for r in results if not r["is_genuine"]]
    
    print("\nSummary Statistics:")
    if gen_sims:
        print(f"  Genuine Similarity: Mean={np.mean(gen_sims):.4f}, Min={np.min(gen_sims):.4f}, Max={np.max(gen_sims):.4f}")
    if imp_sims:
        print(f"  Impostor Similarity: Mean={np.mean(imp_sims):.4f}, Min={np.min(imp_sims):.4f}, Max={np.max(imp_sims):.4f}")

if __name__ == "__main__":
    run_benchmark()
