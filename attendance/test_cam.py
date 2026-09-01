import cv2
print("Testing OpenCV camera access...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(f"Is Opened with DSHOW: {cap.isOpened()}")
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    print(f"Is Opened without DSHOW: {cap.isOpened()}")

if cap.isOpened():
    ret, frame = cap.read()
    print(f"Frame Read Success: {ret}")
    cap.release()
else:
    print("Failed to open camera.")
