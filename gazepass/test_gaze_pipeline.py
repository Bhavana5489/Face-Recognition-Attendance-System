import sys
import os
import unittest
import numpy as np
import time

# Insert gazepass path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.liveness.active_challenge.challenge_engine import ActiveChallengeEngine, ChallengeState, ChallengeResult
from app.liveness.active_challenge.gaze_estimator import GazeObservation, FallbackGazeEstimator

class TestGazePipeline(unittest.TestCase):
    def setUp(self):
        self.engine = ActiveChallengeEngine(num_challenges=3, step_timeout=2.0)
        self.session_info = self.engine.generate_session("test_session_123", liveness_mode="DEGRADED")
        self.session = self.engine.active_session

    def test_sequence_generation(self):
        self.assertEqual(self.session_info["session_id"], "test_session_123")
        self.assertEqual(self.session_info["liveness_mode"], "DEGRADED")
        sequence = self.session_info["sequence"]
        self.assertGreaterEqual(len(sequence), 3)
        # Ensure no adjacent duplicates
        for i in range(len(sequence) - 1):
            self.assertNotEqual(sequence[i]["command"], sequence[i+1]["command"])

    def test_baseline_acquisition_and_locking(self):
        # State starts as WAITING
        self.assertEqual(self.session["state"], ChallengeState.WAITING)
        
        # Feed one observation to transition to BASELINE_ACQUISITION
        obs = GazeObservation(time.time(), 0.05, -0.02, 0.9, "DEGRADED", 1.0, -1.0, 0.0)
        state = self.engine.verify_observation(obs, face_quality=0.85)
        self.assertEqual(state, ChallengeState.BASELINE_ACQUISITION.value)

        # Feed 15 stable center observations
        for _ in range(15):
            obs = GazeObservation(time.time(), 0.05, -0.02, 0.9, "DEGRADED", 1.0, -1.0, 0.0)
            state = self.engine.verify_observation(obs, face_quality=0.85)

        # Verify that state changes all the way to WAITING_REACTION (instant transition flow)
        self.assertEqual(self.session["state"], ChallengeState.WAITING_REACTION)
        self.assertAlmostEqual(self.session["baseline_x"], 0.05)
        self.assertAlmostEqual(self.session["baseline_y"], -0.02)
        self.assertEqual(self.session["sigma_x"], 0.0)
        self.assertEqual(self.session["sigma_y"], 0.0)
        self.assertEqual(self.session["T_x"], 0.15)
        self.assertEqual(self.session["T_y"], 0.15)

    def test_incorrect_vertical_movement_rejection(self):
        """
        Tests that vertical eye movements are rejected when a horizontal target is active.
        """
        # Lock baseline first
        self.test_baseline_acquisition_and_locking()
        
        # Simulate reaction window delay of 0.4s
        time.sleep(0.45)
        
        # Feed one observation to transition from WAITING_REACTION to TRACKING_TARGET
        obs = GazeObservation(time.time(), 0.05, -0.02, 0.9, "DEGRADED", 1.0, -1.0, 0.0)
        state = self.engine.verify_observation(obs, face_quality=0.85)
        self.assertEqual(state, ChallengeState.TRACKING_TARGET.value)
        
        # Get active command
        current_target = self.session["sequence"][0]
        active_command = current_target.command
        
        # If active command is horizontal (LEFT/RIGHT), feed vertical movements and make sure stability does not lock
        if active_command in ["LEFT", "RIGHT"]:
            for _ in range(10):
                obs = GazeObservation(time.time(), 0.05, 0.50, 0.9, "DEGRADED", 1.0, -1.0, 0.0)
                state = self.engine.verify_observation(obs, face_quality=0.85)
                # State should remain TRACKING_TARGET, and stability_start_time should be None
                self.assertEqual(state, ChallengeState.TRACKING_TARGET.value)
                self.assertIsNone(self.session["stability_start_time"])

    def test_successful_challenge_path(self):
        """
        Tests that matching correct directional trajectories with stability locks each step and passes the liveness test.
        """
        self.test_baseline_acquisition_and_locking()
        
        # Run through the sequence
        total_steps = len(self.session["sequence"])
        
        for i in range(total_steps):
            # Handle starting and reaction window transitions
            if self.session["state"] == ChallengeState.CHALLENGE_STARTED:
                obs = GazeObservation(time.time(), 0.05, -0.02, 0.9, "DEGRADED", 1.0, -1.0, 0.0)
                state = self.engine.verify_observation(obs, face_quality=0.85)
                
            if self.session["state"] == ChallengeState.WAITING_REACTION:
                time.sleep(0.45)
                obs = GazeObservation(time.time(), 0.05, -0.02, 0.9, "DEGRADED", 1.0, -1.0, 0.0)
                state = self.engine.verify_observation(obs, face_quality=0.85)
                
            self.assertEqual(self.session["state"], ChallengeState.TRACKING_TARGET)
            
            # Determine correct gaze value based on target
            target = self.session["sequence"][i]
            cmd = target.command
            baseline_x = self.session["baseline_x"]
            baseline_y = self.session["baseline_y"]
            
            if cmd == "LEFT":
                gx, gy = baseline_x - 0.35, baseline_y
            elif cmd == "RIGHT":
                gx, gy = baseline_x + 0.35, baseline_y
            elif cmd == "UP":
                gx, gy = baseline_x, baseline_y - 0.35
            elif cmd == "DOWN":
                gx, gy = baseline_x, baseline_y + 0.35
            else:
                gx, gy = baseline_x, baseline_y
                
            # Feed correct observations to satisfy the 0.3s stability duration (allow up to 1.5s total loop)
            start_correct = time.time()
            while time.time() - start_correct < 1.5:
                obs = GazeObservation(time.time(), gx, gy, 0.95, "DEGRADED", 1.0, -1.0, 0.0)
                state = self.engine.verify_observation(obs, face_quality=0.85)
                if state in [ChallengeState.TARGET_ACQUIRED.value, ChallengeState.ADVANCE.value, ChallengeState.CHALLENGE_STARTED.value, ChallengeState.COMPLETED.value]:
                    break
                time.sleep(0.05)
                
            # If there are more steps, state should eventually progress, else COMPLETED
            if i < total_steps - 1:
                self.assertIn(self.session["state"], [ChallengeState.CHALLENGE_STARTED, ChallengeState.WAITING_REACTION, ChallengeState.TRACKING_TARGET])
            else:
                self.assertEqual(self.session["state"], ChallengeState.COMPLETED)
                
        # Check final result
        res = self.engine.get_challenge_result()
        self.assertEqual(res.decision, "PASS")
        self.assertEqual(res.completed_steps, total_steps)
        self.assertEqual(res.challenge_score, 1.0)
        self.assertEqual(res.gaze_score, 1.0)

    def test_directional_rejection(self):
        """
        Tests that incorrect directions (e.g. looking RIGHT when command is LEFT) are rejected.
        """
        self.test_baseline_acquisition_and_locking()
        
        # Simulate reaction window delay of 0.4s
        time.sleep(0.45)
        
        # Feed one observation to transition from WAITING_REACTION to TRACKING_TARGET
        obs = GazeObservation(time.time(), 0.05, -0.02, 0.9, "DEGRADED", 1.0, -1.0, 0.0)
        state = self.engine.verify_observation(obs, face_quality=0.85)
        self.assertEqual(state, ChallengeState.TRACKING_TARGET.value)
        
        # Get active command
        target = self.session["sequence"][0]
        cmd = target.command
        baseline_x = self.session["baseline_x"]
        baseline_y = self.session["baseline_y"]
        
        # Feed the OPPOSITE direction of the target command
        if cmd == "LEFT":
            gx, gy = baseline_x + 0.35, baseline_y  # RIGHT
        elif cmd == "RIGHT":
            gx, gy = baseline_x - 0.35, baseline_y  # LEFT
        elif cmd == "UP":
            gx, gy = baseline_x, baseline_y + 0.35  # DOWN
        else: # DOWN or BLINK
            gx, gy = baseline_x, baseline_y - 0.35  # UP
            
        # Feed incorrect direction multiple times
        for _ in range(10):
            obs = GazeObservation(time.time(), gx, gy, 0.95, "DEGRADED", 1.0, -1.0, 0.0)
            state = self.engine.verify_observation(obs, face_quality=0.85)
            # Should remain in TRACKING_TARGET (or other non-terminal / non-acquired state)
            self.assertEqual(state, ChallengeState.TRACKING_TARGET.value)
            self.assertIsNone(self.session["stability_start_time"])

if __name__ == "__main__":
    unittest.main()
