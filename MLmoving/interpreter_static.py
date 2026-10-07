import time
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
import numpy as np

from hand_features import create_hand_landmarker, feature_vector, draw_landmarks

STATIC_WINDOW = 8
STATIC_CONFIDENCE_THRESHOLD = 0.7
STATIC_CONSISTENCY_THRESHOLD = 0.7

COOLDOWN = 3  # seconds to wait after accepting a letter before accepting again
RESET_CONFIDENCE_DROP = 0.3  # confidence must drop below this to record a new letter early


def main():
    static_model = joblib.load("static_model.joblib")
    landmarker = create_hand_landmarker()

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    static_buffer = deque(maxlen=STATIC_WINDOW) #always keeps most recent 8 frames of letters and confidences

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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        time_stamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, time_stamp_ms)

        now = time.time()
        display_letter = None
        display_confidence = 0.0

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)

            
            vector = feature_vector(hand_landmarks).reshape(1, -1)# reshape to 1 row for sklearn
            probabilities = static_model.predict_proba(vector)[0] #get probabilities for each class(each letter)
            best = int(np.argmax(probabilities))#index of most likely letter
            static_letter = static_model.classes_[best]
            static_confidence = float(probabilities[best])
            static_buffer.append((static_letter, static_confidence))

            display_letter = static_letter
            display_confidence = static_confidence

            # state machine
        
            if state == "COOLDOWN":
                if now >= cooldown_until: #ready to accept a new letter after cooldown
                    state = "READY"
                elif static_letter != last_accepted_letter and static_confidence < RESET_CONFIDENCE_DROP:
                    # pose changed before the timer even elapsed, safe to release early
                    state = "READY"

            if state == "READY" and len(static_buffer) == STATIC_WINDOW:
                letters = [l for l, c in static_buffer if c >= STATIC_CONFIDENCE_THRESHOLD]
                if len(letters) >= STATIC_WINDOW * STATIC_CONSISTENCY_THRESHOLD:
                    most_common, count = Counter(letters).most_common(1)[0]
                    if count >= STATIC_WINDOW * STATIC_CONSISTENCY_THRESHOLD:
                        word += most_common
                        last_accepted_letter = most_common
                        state = "COOLDOWN"
                        cooldown_until = now + COOLDOWN
                        static_buffer.clear()
                        print(f"Accepted letter: {most_common}, current word: {word}")
        else:
            static_buffer.clear()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 8 or key == 127:  # backspace
            word = word[:-1]
        elif key == ord(" "):
            word += " "
        elif key == ord("0"):
            word = ""

        hud1 = f"letter: {display_letter} - confidence: {display_confidence:.1%}"
        hud2 = f"word: {word}"
        cv2.putText(frame, hud1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0, 0), 2)
        cv2.putText(frame, hud2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0,0), 2)
        cv2.imshow("ASL Live Interpreter - Static Only", frame)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()