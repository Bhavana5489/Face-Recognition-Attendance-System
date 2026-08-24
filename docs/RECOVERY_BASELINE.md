# RECOVERY BASELINE

**Date:** 2026-08-24  
**Branch:** gazepass-v2-correction  
**Recovery snapshot commit:** `e6dee59`  
**Last known-working commit:** `d545628` (UI/UX overhaul — recognition + liveness working at single-expression)

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.13.3 (MSC v.1943, 64-bit) |
| OpenCV | 5.0.0 |
| NumPy | 2.3.2 |
| ONNX Runtime | 1.28.0 |
| Flask | latest |
| OS | Windows 11 |

---

## Active Model Files

| Model | Path | Size |
|-------|------|------|
| YuNet face detector | `gazepass/models/detector/face_detection_yunet_2023mar.onnx` | 232 KB |
| SFace recognizer | `gazepass/models/recognition/face_recognition_sface_2021dec.onnx` | present |
| MiniFASNet V1SE | `gazepass/models/liveness/minifasnet_v1se.onnx` | present |
| MiniFASNet V2 | `gazepass/models/liveness/minifasnet_v2.onnx` | present |
| FaceX landmark (98-pt) | `gazepass/models/landmarks/facex_landmark.onnx` | **MISSING** |

**FaceX model is absent.** System operates in DEGRADED gaze mode (YuNet 5-pt + pupil contour).

---

## Database

| Item | Value |
|------|-------|
| DB path | `gazepass/gazepass.db` |
| Schema version | v1 (no migrations table) |
| Students | xyz (×2), shravan |
| Biometric templates | loaded per student from `biometric_templates` table |
| Template status field | `ACTIVE` / `QUARANTINE` |

**Schema must not be altered without a backup + migration path.**

---

## Server Entry Point

```
attendance/server.py
```

Flask dev server, port 5000, all addresses.

---

## Camera

| Setting | Value |
|---------|-------|
| Device | CAP_DSHOW (index 0) |
| Resolution | Auto-negotiated (typically 1280×720 or 640×480) |
| FPS | Native camera FPS (~30) |
| Flip | Horizontal (cv2.flip(frame, 1)) |

---

## Confirmed Broken Symptoms (as of snapshot e6dee59)

### Symptom 1 — Recognition instability across expressions
- Same student smiling or changing expression → `UNCERTAIN` or `RECOGNITION_PENDING` instead of stable identity
- **Traced root cause:** `get_consensus_identity()` `sim_variance > 0.05` gate rejects legitimate intra-person expression variation

### Symptom 2 — Challenge dot fixed at center
- Dot never moves to LEFT/RIGHT/UP/DOWN positions
- **Traced root cause:** `challenge_engine.verify_observation()` is only called after identity + PAD gates pass. During the 5-frame identity accumulation window the engine stays in `WAITING` state forever. Dot renders at (50%, 50%) for `WAITING` condition.

### Symptom 3 (historical, fixed) — Premature FAILED on every verification
- Fixed: `get_challenge_result()` now only called on terminal states

### Symptom 4 (historical, fixed) — XSS student name injection
- Fixed: admin.html uses `textContent` + DOM methods; 3 malicious entries deleted from DB

---

## Recovery Milestone Order

```
MILESTONE 1 (this session):
  SFace recognizes same student across expressions
  +
  LEFT/RIGHT/UP/DOWN dots actually move

MILESTONE 2 (next session):
  Reconnect full V2 liveness pipeline
  (baseline → directional gaze → AttendanceEngine gates)

MILESTONE 3:
  Re-enable template adaptation
  Calibrate thresholds with genuine/impostor dataset
```

---

## Rule 39 — Recovery Safety

1. Create timestamped backup before modifying recognition or liveness code.  
2. Preserve current `gazepass.db` — never run destructive migrations automatically.  
3. Never overwrite or bulk-delete biometric templates during debugging.  
4. Diagnostic bypasses are **development-only**, clearly isolated from production attendance path.  
5. After each recovery phase, run corresponding regression tests before proceeding.  
6. Record the exact commit hash associated with every known-working state.  
7. If a regression is traced to a recent change still in Git, prefer restoring the smallest known-working implementation and then applying V2 corrections incrementally.
