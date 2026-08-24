# Gazepass 2026: Enterprise Biometric Architecture

## 1. Vision Pipeline
- **Detector**: OpenCV Zoo YuNet (2023mar) ONNX - Ultra-fast 5-point face detection.
- **Tracker**: Custom Python ByteTrack - Bipartite Hungarian matching for rigid object tracking, preventing redundant detection runs.
- **Landmarks**: FaceX/OpenCV Dense Landmarker - 98-point mesh extraction for gaze and pose.
- **Quality Engine**: Evaluates size, sharpness (Laplacian variance), brightness, and yaw/pitch deviation.

## 2. Recognition Engine
- **Model**: OpenCV Zoo SFace (2021dec) ONNX.
- **Alignment**: Deterministic 5-point affine transform to 112x112 standard crop.
- **Database**: SQLite3. Embeddings stored as raw BLOBs (numpy float32).
- **Search Engine**: In-memory NumPy cosine similarity matrix multiplication (Top-K exact retrieval).

## 3. Liveness Fusion (PAD)
- **Passive PAD**: Ensemble of MiniFASNet V1SE and V2 (ONNX). Pre-processes frames with contextual background scaling.
- **Active Challenge**: Generates random trajectory states (Look Left, Look Right, Blink, Look Up, Look Down) and verifies against FaceX geometric outputs and Eye Aspect Ratio (EAR).
- **Temporal PAD**: Dedicated data collection pipeline (`app/liveness/temporal`) to build future 3D-CNN replay detectors based on multi-frame dynamics.

## 4. Attendance State Machine
- **Flow**: Quality Gate -> Liveness Gate (Passive/Active) -> Identity Gate -> Temporal Aggregation -> Idempotency Check -> DB Insertion.
- **Temporal Aggregation**: Requires consensus (median similarity > threshold) across a minimum of 5 consecutive frames for a single track ID before marking attendance.
- **Idempotency**: SQLite `UNIQUE(student_id, attendance_date)` ensures users cannot be marked twice in one day.

## 5. Security & Template Adaptation
- **Adaptation Strategy**: High-confidence verification crops are quarantined. Only after a student acquires multiple quarantine templates spanning separate, distinct days will the medoid template be promoted into the active gallery. This eliminates model poisoning attacks from single-day presentation variations.
- **Observability**: `JSONFormatter` logging architecture for direct ingestion into Prometheus/ELK stacks.
