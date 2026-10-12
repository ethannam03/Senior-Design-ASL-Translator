"""
collect_motion_data.py

Standalone data collection for the motion letters only (J, Z) plus a NONE
class (ordinary hand movement/jitter that should NOT be mistaken for J or
Z). No static letters here at all -- this file's only job is building
motion_only_data.csv.

How to use:
  1. Press '1', '2', or '3' to pick which class you're about to record
     (1 = J, 2 = Z, 3 = NONE).
  2. Press ENTER to start a recording window (MOTION_BUFFER_SECONDS long).
  3. Perform the gesture during that window:
       - J: hold the "I" hand shape (pinky up, rest curled) then hook the
         pinky down and slightly forward.
       - Z: extend the index finger (like "D") and trace a Z shape in the
         air -- right, diagonal down-left, right again.
       - NONE: anything that ISN'T J or Z -- hold a static letter and
         wiggle slightly, move your hand around idly, adjust your grip,
         etc. This class is what teaches the model "this is just normal
         movement, not a real gesture."
  4. Repeat at least 60-100 times per class, varying speed and size of the
     gesture, before moving to train_motion_model.py.

Output: motion_only_data.csv

Keys (all non-letter, consistent with the rest of this project):
  1/2/3   set current label to J / Z / NONE
  ENTER   record one motion buffer
  ESC     quit
"""

import csv
import os
import time

import cv2
import mediapipe as mp

from hand_features import (
    create_hand_landmarker, draw_landmarks,
    landmarks_to_array, normalize_landmarks, motion_vector,
    MOTION_FEATURE_NAME_ORDER,
)

MOTION_CSV = "motion_only_data.csv"
MOTION_BUFFER_SECONDS = 1

KEY_ESC = 27
KEY_ENTER = 13
KEY_ENTER_ALT = 10

LABEL_KEYS = {
    ord("1"): "J",
    ord("2"): "Z",
    ord("3"): "NONE",
}


def ensure_csv(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def append_row(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def main():
    ensure_csv(MOTION_CSV, MOTION_FEATURE_NAME_ORDER + ["label"])

    landmarker = create_hand_landmarker()
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    current_label = None
    recording = False
    buffer = []
    record_start = 0.0
    counts = {}

    print("Started. 1=J 2=Z 3=NONE, ENTER=record, ESC=quit.")

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
        hand_landmarks = None
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)

        # ---------------- recording ----------------
        if recording:
            if hand_landmarks is not None:
                raw = landmarks_to_array(hand_landmarks)
                normalized = normalize_landmarks(raw)
                buffer.append(normalized)
            if now - record_start >= MOTION_BUFFER_SECONDS:
                if len(buffer) >= 5 and current_label is not None:
                    feat = motion_vector(buffer)
                    append_row(MOTION_CSV, list(feat) + [current_label])
                    counts[current_label] = counts.get(current_label, 0) + 1
                    print(f"Saved '{current_label}' ({len(buffer)} frames) "
                          f"-> total {counts[current_label]}")
                else:
                    print("Buffer too short / no label set, discarded.")
                recording = False
                buffer = []

        # ---------------- key handling ----------------
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_ESC:
            break
        elif key in LABEL_KEYS:
            current_label = LABEL_KEYS[key]
            print(f"Label set to '{current_label}'")
        elif key in (KEY_ENTER, KEY_ENTER_ALT) and not recording:
            if current_label is None:
                print("Set a label first (1=J, 2=Z, 3=NONE)")
            else:
                recording = True
                buffer = []
                record_start = now
                print(f"Recording '{current_label}'...")

        # ---------------- HUD ----------------
        hud = f"Label: {current_label} | ENTER to record"
        if recording:
            hud += " | RECORDING..."
        hud += f" | counts: {counts}"
        cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2)
        cv2.imshow("Motion Data Collection", frame)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()