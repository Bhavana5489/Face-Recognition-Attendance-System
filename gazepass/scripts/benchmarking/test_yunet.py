import cv2
import time
import sys
import os

# Add gazepass root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.vision.detector.yunet import YuNetDetector

def test_camera():
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'detector', 'face_detection_yunet_2023mar.onnx')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("[*] Initializing YuNet detector...")
    detector = YuNetDetector(model_path, conf_threshold=0.7)
    
    # Try CAP_DSHOW for Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[!] Failed to open camera using CAP_DSHOW. Falling back to default.")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[Error] Cannot open camera.")
        return

    print("[*] Starting benchmark... Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        frame = cv2.flip(frame, 1)
        
        # Benchmark detection
        t1 = time.time()
        detections = detector.detect(frame)
        latency = (time.time() - t1) * 1000  # ms
        
        frame_count += 1
        if frame_count % 10 == 0:
            fps = frame_count / (time.time() - start_time)
            
        # Draw results
        cv2.putText(frame, f"YuNet FPS: {fps:.1f} | Latency: {latency:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{det.confidence:.2f}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw landmarks
            for lx, ly in det.landmarks:
                cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)

        cv2.imshow('YuNet Benchmark', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Benchmark complete.")

if __name__ == "__main__":
    test_camera()
