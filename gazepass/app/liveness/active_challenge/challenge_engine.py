import numpy as np
import time
import random
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
from app.liveness.active_challenge.gaze_estimator import GazeObservation, load_config

class ChallengeState(Enum):
    WAITING = "WAITING"
    BASELINE_ACQUISITION = "BASELINE_ACQUISITION"
    BASELINE_LOCKED = "BASELINE_LOCKED"
    CHALLENGE_STARTED = "CHALLENGE_STARTED"
    WAITING_REACTION = "WAITING_REACTION"
    TRACKING_TARGET = "TRACKING_TARGET"
    TARGET_ACQUIRED = "TARGET_ACQUIRED"
    ADVANCE = "ADVANCE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    INCONCLUSIVE = "INCONCLUSIVE"

class ChallengeTarget:
    def __init__(self, command: str, screen_x: int, screen_y: int, expected_direction: str):
        self.command = command
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.expected_direction = expected_direction

class ChallengeResult:
    def __init__(self, completed_steps: int, total_steps: int, gaze_score: float, challenge_score: float, decision: str, failure_reason: Optional[str] = None):
        self.completed_steps = completed_steps
        self.total_steps = total_steps
        self.gaze_score = gaze_score
        self.challenge_score = challenge_score
        self.decision = decision             # "PASS", "FAIL", "INCONCLUSIVE"
        self.failure_reason = failure_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "gaze_score": self.gaze_score,
            "challenge_score": self.challenge_score,
            "decision": self.decision,
            "failure_reason": self.failure_reason
        }

class ActiveChallengeEngine:
    """
    Generates and verifies random challenge trajectories to defeat pre-recorded video replays.
    Enforces baseline acquisition, relative 2D direction validation, and continuous stability.
    """
    def __init__(self, num_challenges: int = 3, step_timeout: float = 3.5):
        self.num_challenges = num_challenges
        self.step_timeout = step_timeout
        self.active_session = None

    def generate_session(self, session_id: str, liveness_mode: str = "DEGRADED") -> Dict[str, Any]:
        """
        Generates a sequence of random directional commands bound to a session.
        DEGRADED mode enforces 3 directional steps, no blinks.
        """
        pool = ["LEFT", "RIGHT", "UP", "DOWN"]
        if liveness_mode == "FULL":
            pool.append("BLINK")

        sequence_targets = []
        last_command = None

        # Build constrained sequence of targets
        steps_count = self.num_challenges if liveness_mode == "FULL" else max(3, self.num_challenges)
        
        while len(sequence_targets) < steps_count:
            cmd = random.choice(pool)
            if cmd == last_command:
                continue
            if cmd == "BLINK" and len(sequence_targets) == 0:
                # Avoid starting with blink
                continue
                
            # Map command to UI screen coordinates (percentage) and expected direction
            if cmd == "LEFT":
                target = ChallengeTarget("LEFT", 25, 50, "LEFT")
            elif cmd == "RIGHT":
                target = ChallengeTarget("RIGHT", 75, 50, "RIGHT")
            elif cmd == "UP":
                target = ChallengeTarget("UP", 50, 25, "UP")
            elif cmd == "DOWN":
                target = ChallengeTarget("DOWN", 50, 75, "DOWN")
            else:  # BLINK
                target = ChallengeTarget("BLINK", 50, 50, "BLINK")

            sequence_targets.append(target)
            last_command = cmd

        # Load configurable baseline collection constraints
        config = load_config()
        gaze_cfg = config.get('gaze', {}).get('baseline', {})
        min_samples = gaze_cfg.get('min_samples', 10)
        max_samples = gaze_cfg.get('max_samples', 20)
        max_sigma_x = gaze_cfg.get('max_robust_sigma_x', 0.10)
        max_sigma_y = gaze_cfg.get('max_robust_sigma_y', 0.10)
        directional_ratio = config.get('directional_ratio', 1.5)

        self.active_session = {
            "session_id": session_id,
            "liveness_mode": liveness_mode,
            "sequence": sequence_targets,
            "current_index": 0,
            "state": ChallengeState.WAITING,
            "state_entered_time": time.time(),
            
            # Gaze tracking metrics
            "gaze_history": [],                     # Rolling window of last 10 delta_x, delta_y
            "baseline_observations": [],            # Collected center observations
            
            # Locked baseline metrics
            "baseline_x": 0.0,
            "baseline_y": 0.0,
            "sigma_x": 0.0,
            "sigma_y": 0.0,
            
            # Configuration
            "min_samples": min_samples,
            "max_samples": max_samples,
            "max_sigma_x": max_sigma_x,
            "max_sigma_y": max_sigma_y,
            "directional_ratio": directional_ratio,
            
            # Adaptive Thresholds (CALIBRATION_DEFAULT_ONLY parameters as fallback)
            "T_x": 0.15,
            "T_y": 0.15,
            
            # Timers
            "step_start_time": 0.0,
            "stability_start_time": None,
            "completed_steps": 0,
            "failure_reason": None
        }

        return {
            "session_id": session_id,
            "sequence": [{"command": t.command, "screen_x": t.screen_x, "screen_y": t.screen_y} for t in sequence_targets],
            "liveness_mode": liveness_mode,
            "timeout_sec": self.step_timeout * steps_count
        }

    def verify_observation(self, observation: GazeObservation, face_quality: float) -> str:
        """
        Processes a single GazeObservation and updates the challenge state machine.
        Returns the new state string.
        """
        if not self.active_session:
            return "INACTIVE"

        session = self.active_session
        now = time.time()

        # Helper to compute robust dispersion using Median Absolute Deviation (MAD)
        def compute_robust_dispersion(vals: List[float], med: float) -> float:
            if not vals:
                return 0.0
            ad = [abs(v - med) for v in vals]
            mad = float(np.median(ad))
            return 1.4826 * mad

        # Run sequential transitions loop to immediately process transient states
        max_transitions = 5
        for _ in range(max_transitions):
            old_state = session["state"]

            if session["state"] == ChallengeState.WAITING:
                session["state"] = ChallengeState.BASELINE_ACQUISITION
                session["state_entered_time"] = now

            elif session["state"] == ChallengeState.BASELINE_ACQUISITION:
                # Enforce Quality & Stability gates on baseline observations
                # User must look approximately straight (center head pose)
                if face_quality >= 0.40 and observation.confidence >= 0.30:
                    if abs(observation.yaw) <= 15.0 and abs(observation.pitch) <= 15.0:
                        session["baseline_observations"].append(observation)
                        if len(session["baseline_observations"]) > session["max_samples"]:
                            session["baseline_observations"].pop(0)

                # Check if enough samples gathered
                min_s = session["min_samples"]
                if len(session["baseline_observations"]) >= min_s:
                    gxs = [obs.gaze_x for obs in session["baseline_observations"]]
                    gys = [obs.gaze_y for obs in session["baseline_observations"]]
                    
                    med_x = float(np.median(gxs))
                    med_y = float(np.median(gys))
                    
                    sigma_x_rob = compute_robust_dispersion(gxs, med_x)
                    sigma_y_rob = compute_robust_dispersion(gys, med_y)

                    # Stability check: dispersion must be small
                    if sigma_x_rob <= session["max_sigma_x"] and sigma_y_rob <= session["max_sigma_y"]:
                        # Lock baseline parameters
                        session["baseline_x"] = med_x
                        session["baseline_y"] = med_y
                        session["sigma_x"] = sigma_x_rob
                        session["sigma_y"] = sigma_y_rob
                        
                        # Compute adaptive thresholds (CALIBRATION_DEFAULT_ONLY defaults as bounds)
                        session["T_x"] = max(0.15, 3.0 * sigma_x_rob)
                        session["T_y"] = max(0.15, 3.0 * sigma_y_rob)
                        
                        session["state"] = ChallengeState.BASELINE_LOCKED
                        session["state_entered_time"] = now
                
                # Timeout during baseline acquisition (15 seconds maximum to lock center)
                if now - session["state_entered_time"] > 15.0 and session["state"] == ChallengeState.BASELINE_ACQUISITION:
                    session["state"] = ChallengeState.INCONCLUSIVE
                    session["failure_reason"] = "Baseline acquisition timeout"

            elif session["state"] == ChallengeState.BASELINE_LOCKED:
                session["state"] = ChallengeState.CHALLENGE_STARTED
                session["state_entered_time"] = now

            elif session["state"] == ChallengeState.CHALLENGE_STARTED:
                session["step_start_time"] = now
                session["gaze_history"].clear()
                session["stability_start_time"] = None
                session["state"] = ChallengeState.WAITING_REACTION
                session["state_entered_time"] = now

            elif session["state"] == ChallengeState.WAITING_REACTION:
                # Ignore target success for a short reaction window
                reaction_window = 0.4
                
                # Still record history during reaction delay
                dx = observation.gaze_x - session["baseline_x"]
                dy = observation.gaze_y - session["baseline_y"]
                session["gaze_history"].append((dx, dy))
                if len(session["gaze_history"]) > 10:
                    session["gaze_history"].pop(0)
                    
                if now - session["step_start_time"] >= reaction_window:
                    session["state"] = ChallengeState.TRACKING_TARGET
                    session["state_entered_time"] = now

            elif session["state"] == ChallengeState.TRACKING_TARGET:
                # Check step timeout
                if now - session["step_start_time"] > self.step_timeout:
                    session["state"] = ChallengeState.TIMEOUT
                    session["failure_reason"] = f"Timeout waiting for step {session['current_index'] + 1}"
                else:
                    # Calculate displacement from baseline
                    dx = observation.gaze_x - session["baseline_x"]
                    dy = observation.gaze_y - session["baseline_y"]
                    
                    # Rolling window filter (last 10 observations)
                    session["gaze_history"].append((dx, dy))
                    if len(session["gaze_history"]) > 10:
                        session["gaze_history"].pop(0)
                        
                    # Compute filtered coordinate shift
                    med_dx = float(np.median([h[0] for h in session["gaze_history"]]))
                    med_dy = float(np.median([h[1] for h in session["gaze_history"]]))

                    # Evaluate direction conditions
                    current_target = session["sequence"][session["current_index"]]
                    cmd = current_target.command
                    
                    T_x = session["T_x"]
                    T_y = session["T_y"]
                    R = session.get("directional_ratio", 1.5)

                    condition_met = False
                    
                    if observation.confidence > 0.15:
                        if cmd == "LEFT":
                            condition_met = (med_dx <= -T_x) and (abs(med_dy) <= T_y) and (abs(med_dx) >= R * abs(med_dy))
                        elif cmd == "RIGHT":
                            condition_met = (med_dx >= T_x) and (abs(med_dy) <= T_y) and (abs(med_dx) >= R * abs(med_dy))
                        elif cmd == "UP":
                            condition_met = (med_dy <= -T_y) and (abs(med_dx) <= T_x) and (abs(med_dy) >= R * abs(med_dx))
                        elif cmd == "DOWN":
                            condition_met = (med_dy >= T_y) and (abs(med_dx) <= T_x) and (abs(med_dy) >= R * abs(med_dx))
                        elif cmd == "BLINK":
                            # Check blink status in observation
                            condition_met = getattr(observation, 'is_blinking', False)

                    if condition_met:
                        if session["stability_start_time"] is None:
                            session["stability_start_time"] = now
                        elif now - session["stability_start_time"] >= 0.30:  # Continuous stability gate
                            session["state"] = ChallengeState.TARGET_ACQUIRED
                            session["state_entered_time"] = now
                    else:
                        # Reset stability timer on target violation
                        session["stability_start_time"] = None

            elif session["state"] == ChallengeState.TARGET_ACQUIRED:
                session["completed_steps"] += 1
                session["state"] = ChallengeState.ADVANCE
                session["state_entered_time"] = now

            elif session["state"] == ChallengeState.ADVANCE:
                session["current_index"] += 1
                if session["current_index"] >= len(session["sequence"]):
                    session["state"] = ChallengeState.COMPLETED
                else:
                    session["state"] = ChallengeState.CHALLENGE_STARTED
                session["state_entered_time"] = now

            # Break if no transition happened
            if session["state"] == old_state:
                break

        return session["state"].value

    def get_challenge_result(self) -> ChallengeResult:
        """
        Constructs and returns the final ChallengeResult based on session outcome.
        """
        if not self.active_session:
            return ChallengeResult(0, 0, 0.0, 0.0, "FAIL", "No active session")

        session = self.active_session
        state = session["state"]
        total_steps = len(session["sequence"])
        completed_steps = session["completed_steps"]

        gaze_score = float(completed_steps) / total_steps if total_steps > 0 else 0.0
        
        if state == ChallengeState.COMPLETED and completed_steps == total_steps:
            decision = "PASS"
            challenge_score = 1.0
        elif state in [ChallengeState.TIMEOUT, ChallengeState.FAILED]:
            decision = "FAIL"
            challenge_score = 0.0
        elif state == ChallengeState.INCONCLUSIVE:
            decision = "INCONCLUSIVE"
            challenge_score = 0.0
        else:
            # Session is still in progress — not a failure
            decision = "IN_PROGRESS"
            challenge_score = 0.0

        return ChallengeResult(
            completed_steps=completed_steps,
            total_steps=total_steps,
            gaze_score=gaze_score,
            challenge_score=challenge_score,
            decision=decision,
            failure_reason=session.get("failure_reason")
        )
