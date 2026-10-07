

import time
from collections import deque, Counter
 
import cv2
import joblib
import mediapipe as mp
import numpy as np

from hand_features import create_hand_landmarker, feature_vector, motion_vector, draw_landmarks,normalize_landmarks,landmarks_to_array, FINGERTIPS


STATIC_WINDOW = 8

STATIC_CONFIDENCE_THRESHOLD = 0.7
STAITC_CONSISTENCY_THRESHOLD = 0.75

MOTION_WINDOW = 18
MOTION_ENERGY_THRESHOLD = 0.35
MOTION_CONFIDENCE_THRESHOLD = 0.65

COOLDOWN = 3 #time to begin looking for new gesture
RESET_CONFIDENCE_DROP = 0.55 #confidence drop below this to release static cooldown

def most_active_fingertip(buffer):
    if len(buffer)<2:
        return 0.0
    seq = np.array(buffer)
    best=0
    for i in FINGERTIPS:
        traj = seq[:,i,:2]
        diffs = np.diff(traj,axis=0)
        total = np.sum(np.linalg.norm(diffs,axis=1))
        best = max(best,total)
    return float(best)

def main():
    static_model = joblib.load("static_model.joblib")
    motion_model = joblib.load("motion_model.joblib")

    landmarker = create_hand_landmarker()

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    static_buffer = deque(maxlen=STATIC_WINDOW)
    motion_buffer = deque(maxlen=MOTION_WINDOW)

    word = ""
    state="READY"

    cooldown = 0.0
    last_accepted_letter = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Could not read frame from webcam")
            break
        frame = cv2.flip(frame, 1)  # Mirror the frame
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

            raw = landmarks_to_array(hand_landmarks)
            normalized = normalize_landmarks(raw)
            motion_buffer.append(normalized)

            #static letter detection
            vector = normalized.flatten().reshape(1,-1)
            probabilities = static_model.predict_proba(vector)[0]
            best = int(np.argmax(probabilities))
            static_letter = static_model.classes_[best]
            static_confidence = float(probabilities[best])
            static_buffer.append((static_letter,static_confidence))
            display_letter = static_letter
            display_confidence = static_confidence

            #motion detection
            energy = most_active_fingertip(motion_buffer)
            motion_letter = None
            motion_confidence = 0.0
            if energy >= MOTION_ENERGY_THRESHOLD and len(motion_buffer)==MOTION_WINDOW:
                m_vector = motion_vector(list(motion_buffer)).reshape(1,-1)
                m_probabilities = motion_model.predict_proba(m_vector)[0]
                best = int(np.argmax(m_probabilities))
                motion_letter = motion_model.classes_[best]
                motion_confidence = float(m_probabilities[best])
                if motion_letter != "NONE":
                    display_letter = motion_letter
                    display_confidence = motion_confidence


            #state machine
            if state=="COOLDOWN":
                if now - cooldown >= COOLDOWN:
                    state="READY"
                elif static_letter != last_accepted_letter and static_confidence < RESET_CONFIDENCE_DROP:
                    state="READY"

            if state=="READY":
                accepted = None

                if motion_letter is not None and motion_confidence >= MOTION_CONFIDENCE_THRESHOLD:
                    accepted = motion_letter

                elif len(static_buffer)==STATIC_WINDOW:
                    letters = [l for l,c in static_buffer if c >= STATIC_CONFIDENCE_THRESHOLD]
                    if len(letters)>=STATIC_WINDOW*STAITC_CONSISTENCY_THRESHOLD:
                        most_common, count = Counter(letters).most_common(1)[0]
                        if count >= STATIC_WINDOW*STAITC_CONSISTENCY_THRESHOLD:
                            accepted = most_common

            if accepted is not None:
                word += accepted
                last_accepted_letter = accepted
                cooldown = now + COOLDOWN
                state="COOLDOWN"
                static_buffer.clear()
                motion_buffer.clear()
                print(f"Accepted letter: {accepted}, Current word: {word}")
        else:
            static_buffer.clear()
            motion_buffer.clear()
            if state=="COOLDOWN" and now - cooldown >= COOLDOWN:
                state="READY"


        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == 8 or key == 127:  # Backspace key to delete last letter
            word = word[:-1]
        elif key == ord(' '): # Space key to add a space
            word += " "
        elif key == ord('0'):
            word = ""



        hud1 = f"letter:{display_letter} - confidence: {display_confidence:.1%} - state: {state}"
        hud2 = f"word: {word}"
        cv2.putText(frame, hud1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, hud2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)   
        cv2.imshow("ASL Live Interpreter", frame)


    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()
        