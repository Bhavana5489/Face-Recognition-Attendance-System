# Gazepass V1 Baseline

## Environment
* **Python version**: Record before update
* **OpenCV version**: Record before update
* **ONNX Runtime version**: Record before update
* **NumPy version**: Record before update
* **Flask version**: Record before update
* **SQLite schema version**: MVP baseline
* **Camera resolution**: Standard (usually 640x480 or 1280x720)
* **Hardware**: CPU / GPU details

## Current Behavior Observations (V1)
- **YuNet**: Working well.
- **ByteTrack**: Working well.
- **FaceX landmarks**: Working.
- **SQLite**: Sufficient for current scale.
- **Vector search**: Working.
- **SFace recognition**: Needs improvement. High uncertainty/overlap observed in edge cases.
- **MiniFASNet PAD**: Bad / High false rejection rate on legitimate live users.
- **Active liveness**: Needs rework. Simple gaze/blink is not enough or too strict on single frames.
- **Enrollment**: Working but needs upgrade (currently static 3-pose image selection).

## Pipeline Timings (To be populated)
- Capture:
- Detection:
- Tracking:
- Landmarks:
- Quality:
- SFace:
- PAD:
- Gaze:
- Decision:
