# Gazepass V2 Data Collection Protocol

This document specifies the strict protocols for acquiring the Development, Validation, and Test datasets used for calibrating and benchmarking the recognition and liveness pipelines.

## Dataset Types & Sizes

1. **Development Dataset (Dev)**
   - **Size**: 10–20 individuals
   - **Purpose**: Rapid iteration, debugging, and initial pipeline sanity checks.

2. **Validation Dataset (Val)**
   - **Size**: 20–50 individuals
   - **Purpose**: Threshold selection (Recognition cosine/margin, PAD bounds). Never used for training or initial debugging.

3. **Test Dataset (Test)**
   - **Size**: Separate locked cohort.
   - **Purpose**: Final unbiased reporting.

## Capture Environment Specifications

To ensure evaluation consistency, the following parameters must be strictly adhered to and recorded for every sample:
- **Camera**: Specific webcam ID / model must be logged.
- **Resolution**: Native 720p or 1080p, explicitly recorded.
- **FPS**: 30 FPS minimum.
- **Lighting**: Normal indoor, Low light, Backlight, or Bright Light.
- **Distance**: Arm's length (~50-60cm) standard, varying for specific test conditions.

## Recognition Datasets

### 1. Enrollment
- **Protocol**: One complete video sequence (FRONT -> LEFT -> RIGHT -> UP -> DOWN -> CENTER).
- **Goal**: Used to build the "Gallery" for the student.

### 2. Genuine
- **Protocol**: Same person, *different* session/day or strictly different lighting/clothing than enrollment.
- **Goal**: To evaluate False Rejection Rate (FRR) of legitimate users.

### 3. Impostor
- **Protocol**: Different person (ideally similar demographics/uniforms) facing the camera.
- **Goal**: To evaluate False Acceptance Rate (FAR) between distinct individuals.

## Presentation Attack Detection (PAD) Datasets

### 1. Real
- **Protocol**: Live, genuine person performing standard facial movements.
- **Variations**: Glasses on/off, varying expressions, different distances.

### 2. Print Attack (`print`)
- **Protocol**: A high-resolution printed photograph of a face, held steadily in front of the camera, slightly moved to simulate breathing.

### 3. Phone Photo Attack (`phone_photo`)
- **Protocol**: A static image displayed on a smartphone screen, facing the camera.

### 4. Phone Video Attack (`phone_video`)
- **Protocol**: A pre-recorded video of a face playing on a smartphone.

### 5. Tablet/Laptop Replay (`tablet`, `laptop`)
- **Protocol**: Replaying a video on a larger screen to test context/border detection.

## Active Liveness Datasets

### 1. Genuine Challenge
- **Protocol**: Real person accurately following the randomized prompt sequence (e.g., Look Left -> Blink).

### 2. Replay Challenge
- **Protocol**: Replaying a video of a person performing a challenge, to verify that random sequence generation defeats static video replays.

## Execution via Script
Always use the `scripts/dataset/capture.py` tool. The tool will automatically structure the directory and enforce the collection of required metadata (`student_id`, `session_id`, `lighting`, etc.).
