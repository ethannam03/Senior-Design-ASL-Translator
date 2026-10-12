"""
run_motion_interpreter.py

Standalone real-time interpreter for the motion letters only (J, Z). No
static model involved -- this is purely for testing/debugging the motion
classifier in isolation before combining it with static letters.

How it works:
  - Every frame, normalized landmarks get pushed into a rolling buffer
    (MOTION_WINDOW frames long).
  - most_active_fingertip_motion() gives a cheap "how much is moving right
    now" score. Only once that crosses MOTION_ENERGY_THRESHOLD do we
    bother running the (more expensive) motion classifier on the buffer --
    most frames nothing is happening, so this keeps CPU low.
  - The classifier's "NONE" class is filtered out -- it exists only so
    the model has something to predict when nothing is actually
    happening; it should never itself become an accepted letter.
  - After accepting J or Z, we require BOTH the cooldown timer to expire
    AND the motion energy to drop back down before accepting again --
    otherwise one gesture could get detected multiple times as the
    sliding window keeps re-triggering on the same movement.

Keys:
  BACKSPACE  delete last letter
  SPACE      add a space
  0          clear word
  ESC        quit
"""

import time
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

from hand_features import (
    create_hand_landmarker, draw_landmarks,
    landmarks_to_array, normalize_landmarks, motion_vector,
    FINGERTIPS,
)

MOTION_WINDOW = 20           # ~0.6-0.8s of frames, tune to your camera fps
MOTION_ENERGY_THRESHOLD = 0.35
MOTION_CONFIDENCE_THRESHOLD = 0.65

COOLDOWN = 0.8                # minimum seconds between accepted gestures
ENERGY_RELEASE_THRESHOLD = 0.15  # energy must drop below this (hand settled)
                                  # before a new gesture can be accepted


def most_active_fingertip_motion(buffer):
    """Cheap motion-energy estimate -- just enough to decide whether to
    bother running the classifier this frame."""
    if len(buffer) < 2:
        return 0.0
    seq = np.array(buffer)
    best = 0.0
    for idx in FINGERTIPS:
        traj = seq[:, idx, :2]
        diffs = np.diff(traj, axis=0)
        total = np.sum(np.linalg.norm(diffs, axis=1))
        best = max(best, total)
    return float(best)


def main():
    motion_model = joblib.load("motion_only_model.joblib")
    landmarker = create_hand_landmarker()

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    motion_buffer = deque(maxlen=MOTION_WINDOW)

    word = ""
    state = "READY"
    cooldown_until = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Could not read frame from webcam")
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        now = time.time()
        display_letter = None
        display_confidence = 0.0
        energy = 0.0

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)

            raw = landmarks_to_array(hand_landmarks)
            normalized = normalize_landmarks(raw)
            motion_buffer.append(normalized)

            energy = most_active_fingertip_motion(motion_buffer)

            # ---- state machine ----
            # Release cooldown only once BOTH the timer has expired AND
            # the hand has actually settled down (low energy) -- this is
            # what stops one J or Z from being accepted multiple times as
            # the window keeps sliding across the same motion.
            if state == "COOLDOWN" and now >= cooldown_until and energy < ENERGY_RELEASE_THRESHOLD:
                state = "READY"

            if (state == "READY"
                    and energy >= MOTION_ENERGY_THRESHOLD
                    and len(motion_buffer) == MOTION_WINDOW):
                feat = motion_vector(list(motion_buffer)).reshape(1, -1)
                probabilities = motion_model.predict_proba(feat)[0]
                best = int(np.argmax(probabilities))
                motion_letter = motion_model.classes_[best]
                motion_confidence = float(probabilities[best])

                display_letter = motion_letter
                display_confidence = motion_confidence

                # NONE must never be treated as an accepted letter -- it's
                # only there so the classifier has something to predict
                # when nothing real is happening.
                if motion_letter != "NONE" and motion_confidence >= MOTION_CONFIDENCE_THRESHOLD:
                    word += motion_letter
                    state = "COOLDOWN"
                    cooldown_until = now + COOLDOWN
                    motion_buffer.clear()
                    print(f"Accepted letter: {motion_letter}, current word: {word}")
        else:
            motion_buffer.clear()

        # ---------------- keys ----------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 8 or key == 127:  # backspace
            word = word[:-1]
        elif key == ord(" "):
            word += " "
        elif key == ord("0"):
            word = ""

        # ---------------- HUD ----------------
        hud1 = f"letter: {display_letter} - confidence: {display_confidence:.1%} - energy: {energy:.2f}"
        hud2 = f"word: {word}"
        cv2.putText(frame, hud1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
        cv2.putText(frame, hud2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.imshow("Motion-Only Interpreter", frame)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()