"""
Keys:
  BACKSPACE  delete last letter
  SPACE      add a space
  0          clear word
  ENTER      speak the current word/sentence aloud (text-to-speech)
  ESC        quit

Note on ENTER: a bare SHIFT press can't reliably be read through
cv2.waitKey() -- it's a modifier key, and OpenCV's keyboard polling only
sees actual character codes, not raw key-down states for modifiers alone.
ENTER was used instead since it works the same reliable way as every
other key here.
"""

import time
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3

from hand_features import (
    create_hand_landmarker, draw_landmarks, feature_vector,
    landmarks_to_array, normalize_landmarks, motion_vector,
    FINGERTIPS,
)

#static letter info
STATIC_WINDOW = 20
STATIC_CONFIDENCE_THRESHOLD = 0.70
STATIC_CONSISTENCY_THRESHOLD = 0.75

#motion (J/Z) 
MOTION_WINDOW = 20
MOTION_ENERGY_THRESHOLD = 0.35
MOTION_CONFIDENCE_THRESHOLD = 0.65
ENERGY_RELEASE_THRESHOLD = 0.15  # motion energy must settle below this too


COOLDOWN = 3
RESET_CONFIDENCE_DROP = 0.70  # static confidence must drop below this

#results window appearance
RESULTS_WIDTH = 640
RESULTS_HEIGHT = 300


def most_active_fingertip_motion(buffer):
    """Cheap motion-energy estimate -- just enough to decide whether to
    bother running the motion classifier this frame."""
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


def build_results_frame(static_letter, static_confidence, motion_letter, motion_confidence, word):

    canvas = np.zeros((RESULTS_HEIGHT, RESULTS_WIDTH, 3), dtype=np.uint8)

    static_text = (f"Static:  {static_letter}  ({static_confidence:.1%})"
                   if static_letter else "Static:  -")
    motion_text = (f"Motion:  {motion_letter}  ({motion_confidence:.1%})"
                   if motion_letter else "Motion:  -")
    word_text = f"Word: {word}"

    cv2.putText(canvas, static_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)
    cv2.putText(canvas, motion_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 165, 255), 2)
    cv2.putText(canvas, word_text, (20, 220), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 2)
    cv2.putText(canvas, "ESC quit - SPACE space - BACKSPACE delete - 0 clear",
                (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    return canvas


def main():
    static_model = joblib.load("static_model.joblib")
    motion_model = joblib.load("motion_only_model.joblib")
    landmarker = create_hand_landmarker()

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    static_buffer = deque(maxlen=STATIC_WINDOW)
    motion_buffer = deque(maxlen=MOTION_WINDOW)

    word = ""
    state = "READY"
    cooldown_until = 0.0
    last_accepted_letter = None

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

        static_letter = None
        static_confidence = 0.0
        motion_letter = None
        motion_confidence = 0.0
        energy = 0.0

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)  

            raw = landmarks_to_array(hand_landmarks)
            normalized = normalize_landmarks(raw)
            motion_buffer.append(normalized)

            #  static prediction
            vector = feature_vector(hand_landmarks).reshape(1, -1)
            probabilities = static_model.predict_proba(vector)[0]
            best = int(np.argmax(probabilities))
            static_letter = static_model.classes_[best]
            static_confidence = float(probabilities[best])
            static_buffer.append((static_letter, static_confidence))

            # motion check
            energy = most_active_fingertip_motion(motion_buffer)
            if energy >= MOTION_ENERGY_THRESHOLD and len(motion_buffer) == MOTION_WINDOW:
                m_vector = motion_vector(list(motion_buffer)).reshape(1, -1)
                m_probabilities = motion_model.predict_proba(m_vector)[0]
                m_best = int(np.argmax(m_probabilities))
                m_label = motion_model.classes_[m_best]
                m_confidence = float(m_probabilities[m_best])
                if m_label != "NONE":
                    motion_letter, motion_confidence = m_label, m_confidence

            # state machine
            if state == "COOLDOWN" and now >= cooldown_until:
                if static_letter != last_accepted_letter:
                    
                    state = "READY"
                elif static_confidence < RESET_CONFIDENCE_DROP and energy < ENERGY_RELEASE_THRESHOLD:
                   
                    state = "READY"

            if state == "READY":
                accepted = None

                # motion letters take priority when detected 
                if motion_letter is not None and motion_confidence >= MOTION_CONFIDENCE_THRESHOLD:
                    accepted = motion_letter

                elif len(static_buffer) == STATIC_WINDOW:
                    letters = [l for l, c in static_buffer if c >= STATIC_CONFIDENCE_THRESHOLD]
                    if len(letters) >= STATIC_WINDOW * STATIC_CONSISTENCY_THRESHOLD:
                        most_common, count = Counter(letters).most_common(1)[0]
                        if count >= STATIC_WINDOW * STATIC_CONSISTENCY_THRESHOLD:
                            accepted = most_common

                if accepted is not None:
                    word += accepted
                    last_accepted_letter = accepted
                    state = "COOLDOWN"
                    cooldown_until = now + COOLDOWN
                    static_buffer.clear()
                    motion_buffer.clear()
                    print(f"Accepted letter: {accepted}, current word: {word}")
        else:
            static_buffer.clear()
            motion_buffer.clear()

        #  keys 
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 8 or key == 127:  # backspace
            word = word[:-1]
        elif key == ord(" "):
            word += " "
        elif key == ord("0"):
            word = ""
        elif key in (13, 10):  # ENTER -- speak the current word/sentence
            if word.strip():
                speak_engine = pyttsx3.init()
                speak_engine.say(word)
                speak_engine.runAndWait()  # blocks until speech finishes --
                                            # camera feed pauses briefly
                speak_engine.stop()
                del speak_engine

        
        cv2.imshow("Camera", frame)  

        results_frame = build_results_frame(static_letter, static_confidence,
                                             motion_letter, motion_confidence, word)
        cv2.imshow("Results", results_frame)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()