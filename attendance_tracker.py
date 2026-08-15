# -*- coding: utf-8 -*-
"""
=============================================================
  Gaze-Tracking Attendance System
  File: d:/projects/face_detection/attendance_tracker.py
  Run: python attendance_tracker.py   (use anti_spoofing venv)
=============================================================
  How it works:
  1. Detects your face -> draws rectangle
  2. A red dot moves to 2 random positions over 4.0 seconds
  3. Tracks your pupil direction at each dot position
  4. If you followed 2/2 positions -> PRESENT
  Press SPACE to start check | Q to quit | R to reset list
=============================================================
"""

import cv2
import numpy as np
import time
import random
import os

# ── Cascades (bundled with OpenCV) ────────────────────────
HC = cv2.data.haarcascades
face_cascade  = cv2.CascadeClassifier(HC + "haarcascade_frontalface_default.xml")
eye_cascade   = cv2.CascadeClassifier(HC + "haarcascade_eye.xml")

# ── Config ────────────────────────────────────────────────
CAMERA_ID          = 0
TRACK_DURATION     = 4.0       # seconds for dot sequence (2.0s per stop)
WAYPOINTS_COUNT    = 2         # dot stops
SETTLE_TIME        = 1.30      # seconds to wait after dot moves before checking pupil
CHECK_TIME         = 0.13      # seconds to sample pupil after settling
PASS_THRESHOLD     = 2         # correct waypoints needed out of 2
DOT_RADIUS         = 18
GAZE_MARGIN        = 0.07      # how far off-center pupil can be and still count
FONT               = cv2.FONT_HERSHEY_SIMPLEX
FONT2              = cv2.FONT_HERSHEY_DUPLEX

# Colours
RED    = (30,  30,  220)
GREEN  = (30,  210, 30)
YELLOW = (0,   210, 220)
CYAN   = (220, 200, 0)
WHITE  = (255, 255, 255)
DARK   = (20,  20,  20)
GRAY   = (160, 160, 160)
ORANGE = (0,   140, 255)


# ─────────────────────────────────────────────────────────
#  PUPIL DETECTION
# ─────────────────────────────────────────────────────────
def get_pupil_x_ratio(eye_roi_bgr):
    """
    Returns normalised horizontal pupil position (0=left, 1=right)
    using adaptive darkest-pixel center estimation.
    """
    if eye_roi_bgr is None or eye_roi_bgr.size == 0:
        return None
    gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Crop middle section to filter out dark eyelash and corner skin artifacts
    margin_x = int(w * 0.17)
    margin_y = int(h * 0.20)
    crop = gray[margin_y:h-margin_y, margin_x:w-margin_x]
    if crop.size == 0:
        return None

    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(crop)

    cx = min_loc[0] + margin_x
    return float(cx / w)


def sample_pupil_direction(frame, face_bbox):
    """
    Returns 'left', 'center', or 'right' based on average pupil position
    in both detected eyes within the face region.
    Returns None if no eyes found.
    """
    fx, fy, fw, fh = face_bbox
    face_roi = frame[fy: fy + fh, fx: fx + fw]
    if face_roi.size == 0:
        return None

    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(
        face_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)
    )
    if len(eyes) == 0:
        return None

    ratios = []
    for (ex, ey, ew, eh) in eyes[:2]:
        # Upper half of eye region is more reliable
        eye_roi = face_roi[ey: ey + int(eh * 0.65), ex: ex + ew]
        r = get_pupil_x_ratio(eye_roi)
        if r is not None:
            ratios.append(r)

    if not ratios:
        return None

    avg = np.mean(ratios)
    if avg < (0.5 - GAZE_MARGIN):
        return "left"
    elif avg > (0.5 + GAZE_MARGIN):
        return "right"
    else:
        return "center"


# ─────────────────────────────────────────────────────────
#  DOT PATH GENERATOR
# ─────────────────────────────────────────────────────────
def generate_waypoints(frame_w, frame_h, n=6):
    """
    Create n positions the dot will visit.
    Guaranteed to include left, right, and center zones.
    """
    margin = 80
    zones = [
        # (x_min, x_max, y_min, y_max, zone_label)
        (margin,           frame_w // 3,       margin, frame_h - margin, "left"),
        (frame_w * 2 // 3, frame_w - margin,   margin, frame_h - margin, "right"),
        (frame_w // 3,     frame_w * 2 // 3,   margin, frame_h // 2,     "center"),
        (margin,           frame_w // 3,       margin, frame_h // 2,     "left"),
        (frame_w * 2 // 3, frame_w - margin,   frame_h // 2, frame_h - margin, "right"),
        (frame_w // 3,     frame_w * 2 // 3,   frame_h // 2, frame_h - margin, "center"),
    ]
    random.shuffle(zones)
    waypoints = []
    for (xmin, xmax, ymin, ymax, label) in zones[:n]:
        x = random.randint(int(xmin), int(xmax))
        y = random.randint(int(ymin), int(ymax))
        waypoints.append((x, y, label))
    return waypoints


# ─────────────────────────────────────────────────────────
#  ATTENDANCE STATE
# ─────────────────────────────────────────────────────────
class AttendanceSession:
    IDLE      = "idle"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"

    def __init__(self, frame_w, frame_h):
        self.frame_w    = frame_w
        self.frame_h    = frame_h
        self.state      = self.IDLE
        self.waypoints  = []
        self.wp_idx     = 0
        self.wp_start   = 0.0
        self.results    = []      # list of True/False per waypoint
        self.dot_pos    = (frame_w // 2, frame_h // 2)
        self.dot_alpha  = 1.0    # for pulse animation
        self.name       = ""
        self.wp_duration = TRACK_DURATION / WAYPOINTS_COUNT

    def start(self, name=""):
        self.name      = name if name else "Person"
        self.waypoints = generate_waypoints(self.frame_w, self.frame_h,
                                            WAYPOINTS_COUNT)
        self.wp_idx    = 0
        self.wp_start  = time.time()
        self.results   = []
        self.dot_start_pos = (self.frame_w // 2, self.frame_h // 2)
        self.dot_pos   = self.dot_start_pos
        self.state     = self.RUNNING

    def update(self, frame, face_bbox):
        """
        Call every frame. Returns True when session completes.
        """
        if self.state != self.RUNNING:
            return False

        now     = time.time()
        elapsed = now - self.wp_start

        # Current waypoint
        wp = self.waypoints[self.wp_idx]
        target_x, target_y, zone_label = wp

        # Smooth linear movement over 1.2 seconds (out of 2.0s total stop)
        slide_duration = 1.2
        if elapsed < slide_duration:
            t = elapsed / slide_duration
            self.dot_pos = (
                int(self.dot_start_pos[0] + (target_x - self.dot_start_pos[0]) * t),
                int(self.dot_start_pos[1] + (target_y - self.dot_start_pos[1]) * t)
            )
        else:
            self.dot_pos = (target_x, target_y)

        # Pulse alpha
        self.dot_alpha = 0.7 + 0.3 * abs(np.sin(now * 6))

        # After dot settles, sample pupil
        if elapsed >= SETTLE_TIME and face_bbox is not None:
            gaze = sample_pupil_direction(frame, face_bbox)
            if gaze is not None:
                correct = (gaze == zone_label)
            else:
                correct = False
        else:
            correct = None   # still settling

        # Move to next waypoint after wp_duration
        if elapsed >= self.wp_duration:
            if correct is not None:
                self.results.append(correct)
            self.wp_idx  += 1
            self.wp_start = now
            self.dot_start_pos = self.dot_pos  # Update start position for next leg
            if self.wp_idx >= len(self.waypoints):
                # Session complete
                score = sum(self.results)
                if score >= PASS_THRESHOLD:
                    self.state = self.SUCCESS
                else:
                    self.state = self.FAILED
                return True

        return False

    @property
    def progress(self):
        return self.wp_idx / WAYPOINTS_COUNT

    @property
    def score(self):
        return sum(self.results)


# ─────────────────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────────────────
def draw_dot(frame, pos, alpha=1.0):
    """Draw animated pulsing red dot with glow."""
    x, y = pos
    # Outer glow
    glow_r = int(DOT_RADIUS * 1.8)
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), glow_r, (50, 50, 230), -1)
    cv2.addWeighted(overlay, 0.3 * alpha, frame, 1 - 0.3 * alpha, 0, frame)
    # Main dot
    cv2.circle(frame, (x, y), DOT_RADIUS, RED, -1, cv2.LINE_AA)
    # Bright centre
    cv2.circle(frame, (x, y), DOT_RADIUS // 3, (120, 120, 255), -1, cv2.LINE_AA)


def draw_face_box(frame, bbox, color=GREEN):
    x, y, w, h = bbox
    # Corner-style rectangle (not full rect)
    t = 4       # thickness
    L = 22      # corner length
    pts = [
        ((x, y), (x + L, y), (x, y + L)),
        ((x + w, y), (x + w - L, y), (x + w, y + L)),
        ((x, y + h), (x + L, y + h), (x, y + h - L)),
        ((x + w, y + h), (x + w - L, y + h), (x + w, y + h - L)),
    ]
    for corner in pts:
        c, h1, v1 = corner
        cv2.line(frame, c, h1, color, t, cv2.LINE_AA)
        cv2.line(frame, c, v1, color, t, cv2.LINE_AA)


def draw_hud(frame, session, fps, attendance_list, current_gaze):
    h, w = frame.shape[:2]

    # ── Top bar ───────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), DARK, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.putText(frame, f"Gaze Attendance  |  FPS:{fps:.0f}  |  GAZE: {current_gaze}  |  SPACE=Start",
                (10, 27), FONT, 0.50, GRAY, 1, cv2.LINE_AA)

    # ── State banner ──────────────────────────────────────
    if session.state == session.IDLE:
        msg, color = "Press SPACE to start gaze check", CYAN
    elif session.state == session.RUNNING:
        wp_left = WAYPOINTS_COUNT - session.wp_idx
        msg, color = (f"TRACKING...  Follow the dot!  "
                      f"[{session.wp_idx}/{WAYPOINTS_COUNT}]"), YELLOW
    elif session.state == session.SUCCESS:
        msg, color = f"PRESENT  {session.name}  ({session.score}/{WAYPOINTS_COUNT} correct)", GREEN
    else:
        msg, color = f"FAILED ({session.score}/{WAYPOINTS_COUNT}) - Press SPACE to retry", RED

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 52), (w, h - 10), DARK, -1)
    cv2.addWeighted(overlay2, 0.70, frame, 0.30, 0, frame)
    cv2.putText(frame, msg, (12, h - 22), FONT2, 0.72, color, 2, cv2.LINE_AA)

    # ── Progress bar (during tracking) ───────────────────
    if session.state == session.RUNNING:
        bar_w = int(w * session.progress)
        cv2.rectangle(frame, (0, h - 10), (bar_w, h), color, -1)
        # Dot index markers
        for i in range(WAYPOINTS_COUNT):
            mx = int(w * (i / WAYPOINTS_COUNT))
            mc = GREEN if i < len(session.results) and session.results[i] else GRAY
            cv2.circle(frame, (mx + 12, h - 5), 4, mc, -1)

    # ── Attendance list panel (right side) ────────────────
    if attendance_list:
        panel_x = w - 220
        panel_h = 50 + len(attendance_list) * 28
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (panel_x - 8, 48),
                      (w - 4, 48 + panel_h), DARK, -1)
        cv2.addWeighted(overlay3, 0.72, frame, 0.28, 0, frame)
        cv2.putText(frame, "PRESENT", (panel_x, 72),
                    FONT2, 0.58, GREEN, 1, cv2.LINE_AA)
        for i, entry in enumerate(attendance_list[-8:]):   # show last 8
            cv2.putText(frame, f"  {i+1}. {entry}",
                        (panel_x, 100 + i * 28),
                        FONT, 0.55, WHITE, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_ID}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    ret, test_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read from camera")
        return

    frame_h, frame_w = test_frame.shape[:2]
    session         = AttendanceSession(frame_w, frame_h)
    attendance_list = []
    face_bbox       = None
    prev_time       = time.time()
    counter         = 1        # auto-name counter

    print("=" * 55)
    print("  Gaze Attendance Tracker")
    print("  SPACE = Start check  |  Q = Quit  |  R = Reset list")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame so left/right movements align naturally with the screen
        frame = cv2.flip(frame, 1)

        # ── Detect face ───────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        current_gaze = "N/A"
        if len(faces) > 0:
            # Largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            face_bbox = faces[0].tolist()
            box_color = (GREEN if session.state == session.SUCCESS
                         else RED if session.state == session.FAILED
                         else CYAN)
            draw_face_box(frame, face_bbox, color=box_color)

            # Detect and draw yellow rectangles on eyes to verify tracking
            fx, fy, fw, fh = face_bbox
            face_gray = gray[fy:fy+fh, fx:fx+fw]
            eyes = eye_cascade.detectMultiScale(
                face_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)
            )
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(frame, (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh), YELLOW, 1)

            # Estimate real-time gaze direction
            gaze = sample_pupil_direction(frame, face_bbox)
            if gaze is not None:
                current_gaze = gaze.upper()
        else:
            face_bbox = None

        # ── Update session ────────────────────────────────
        if session.state == session.RUNNING:
            done = session.update(frame, face_bbox)
            # Draw the dot
            draw_dot(frame, session.dot_pos, session.dot_alpha)

            if done:
                if session.state == session.SUCCESS:
                    entry = f"{session.name}"
                    if entry not in attendance_list:
                        attendance_list.append(entry)
                    print(f"  PRESENT: {session.name}  "
                          f"({session.score}/{WAYPOINTS_COUNT} waypoints)")
                else:
                    print(f"  FAILED:  {session.name}  "
                          f"({session.score}/{WAYPOINTS_COUNT} waypoints)")

        # ── FPS ───────────────────────────────────────────
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # ── HUD ───────────────────────────────────────────
        draw_hud(frame, session, fps, attendance_list, current_gaze)

        cv2.imshow("Gaze Attendance  |  SPACE=Start  Q=Quit  R=Reset", frame)

        # ── Key handling ──────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

        elif key == ord(' '):
            if session.state in (session.IDLE,
                                 session.SUCCESS,
                                 session.FAILED):
                name = f"Person {counter}"
                counter += 1
                session.start(name)
                print(f"  Starting check for: {name}")

        elif key in (ord('r'), ord('R')):
            attendance_list.clear()
            session.state = session.IDLE
            counter = 1
            print("  Attendance list reset.")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 55)
    print("  Final Attendance:")
    for i, name in enumerate(attendance_list, 1):
        print(f"    {i}. {name}")
    if not attendance_list:
        print("    (none)")
    print("=" * 55)


if __name__ == "__main__":
    main()
