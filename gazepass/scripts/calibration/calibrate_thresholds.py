import sys
import os
import numpy as np

# Add gazepass root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.database.db_core import DatabaseManager, VectorSearchEngine
from app.vision.recognition.sface import SFaceRecognizer

def calculate_far_frr(genuine_scores, impostor_scores, thresholds):
    """
    Calculates False Accept Rate (FAR) and False Reject Rate (FRR)
    across a range of thresholds.
    """
    results = []
    for t in thresholds:
        # FAR: Impostor score >= threshold
        far = np.sum(impostor_scores >= t) / len(impostor_scores) if len(impostor_scores) > 0 else 0.0
        
        # FRR: Genuine score < threshold
        frr = np.sum(genuine_scores < t) / len(genuine_scores) if len(genuine_scores) > 0 else 0.0
        
        results.append((t, far, frr))
    return results

def run_calibration(db_path: str):
    db = DatabaseManager(db_path)
    engine = VectorSearchEngine(db)
    
    student_ids, gallery = engine.student_ids, engine.gallery
    
    if len(gallery) < 2:
        print("[!] Not enough templates in the database to run calibration.")
        print("    Please enroll multiple students with multiple poses first.")
        return
        
    print(f"[*] Loaded {len(gallery)} active templates across {len(set(student_ids))} students.")
    
    genuine_scores = []
    impostor_scores = []
    
    # Generate pairwise scores (This is O(N^2), fine for calibration dataset size)
    for i in range(len(gallery)):
        for j in range(i + 1, len(gallery)):
            # Cosine similarity
            score = np.dot(gallery[i], gallery[j].T).item()
            
            if student_ids[i] == student_ids[j]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)
                
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    print(f"[*] Generated {len(genuine_scores)} genuine pairs and {len(impostor_scores)} impostor pairs.")
    
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        print("[!] Need both genuine and impostor pairs to calculate FAR/FRR.")
        return
        
    thresholds = np.linspace(0.0, 1.0, 100)
    rates = calculate_far_frr(genuine_scores, impostor_scores, thresholds)
    
    # Find Equal Error Rate (EER) where FAR == FRR
    eer_diff = [abs(far - frr) for t, far, frr in rates]
    eer_idx = np.argmin(eer_diff)
    eer_threshold, eer_far, eer_frr = rates[eer_idx]
    
    # Find Operating Point for FAR <= 0.001 (0.1%)
    op_idx = next((i for i, (t, far, frr) in enumerate(rates) if far <= 0.001), eer_idx)
    op_threshold, op_far, op_frr = rates[op_idx]
    
    print("\n=== CALIBRATION RESULTS ===")
    print(f"Equal Error Rate (EER): {((eer_far + eer_frr)/2)*100:.2f}% at Threshold: {eer_threshold:.3f}")
    print(f"High Security Operating Point (FAR <= 0.1%):")
    print(f"  -> Recommended Threshold: {op_threshold:.3f}")
    print(f"  -> Expected FRR at this threshold: {op_frr*100:.2f}%")
    print(f"  -> Expected FAR at this threshold: {op_far*100:.2f}%")
    print("===========================\n")
    print("Update the FRONTEND or CONFIG with the Recommended Threshold.")

if __name__ == "__main__":
    run_calibration("gazepass.db")
