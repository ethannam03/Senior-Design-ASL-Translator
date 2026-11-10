
"""
    Keys (all non-letter, so every letter key stays free to be a label):
    TAB      toggle STATIC / MOTION mode
    ESC      quit
    SPACE    (STATIC mode, hold) record frames
    1/2/3    (MOTION mode) set current motion label to J / Z / NONE
    ENTER    (MOTION mode) record one motion buffer
    a-y      (STATIC mode only) set current label to that letter(not j)

File will output 2 csv files:
    1. static_data.csv - contains all the static frames recorded with their labels
    2. motion_data.csv - contains all the motion buffers recorded with their labels

 """

import csv
import os
import time
 
import cv2
import mediapipe as mp
 
from hand_features import create_hand_landmarker, feature_vector, STATIC_FEATURE_NAMES, MOTION_FEATURE_NAME_ORDER, motion_vector, draw_landmarks,normalize_landmarks,landmarks_to_array



STATIC_CSV = 'static_data.csv'
MOTION_CSV = 'motion_data.csv'

MOTION_BUF_SEC = 0.7
MOTION_BUF_FPS = 24
STATIC_INTRVL = 0.1


KEY_ESC = 27
KEY_TAB = 9
KEY_ENTER = 13

MOTION_LABELS = {
    ord('1'): 'J',
    ord('2'): 'Z',
    ord('3'): 'NONE'
}

def check_csv(filepath,header):
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def append_csv(filepath,row):
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def main():
    check_csv(STATIC_CSV, STATIC_FEATURE_NAMES+['label'])
    check_csv(MOTION_CSV, MOTION_FEATURE_NAME_ORDER+['label'])

    landmarker = create_hand_landmarker()
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    mode = "STATIC"
    static_label = None
    motion_label = None
    last_static_record = 0.0

    recording_motion = False
    motion_buffer = []
    motion_record_start = 0.0

    static_count = {}
    motion_count = {}

    print("Press TAB to toggle between STATIC and MOTION mode, ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error reading frame from camera")
            break
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally to not hurt my brain
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        time_stamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, time_stamp_ms) 

        hand_landmarks = None
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks)
        now = time.time()

        key = cv2.waitKey(1) & 0xFF

        if mode == "STATIC" and hand_landmarks is not None:
            if key == ord(' ') and static_label is not None:
                if now - last_static_record >= STATIC_INTRVL:
                    features = feature_vector(hand_landmarks)
                    append_csv(STATIC_CSV, list(features) + [static_label])
                    last_static_record = now
                    static_count[static_label] = static_count.get(static_label, 0) + 1
                    print(f"Recorded STATIC sample for label '{static_label}'. Total: {static_count[static_label]}")
                    last_static_record = now

        if mode == "MOTION" and recording_motion:
            if hand_landmarks is not None:
                raw = landmarks_to_array(hand_landmarks)
                normalized = normalize_landmarks(raw)
                motion_buffer.append(normalized)
            if now - motion_record_start >= MOTION_BUF_SEC:
                if len(motion_buffer) >= 5 and motion_label is not None:
                    features = motion_vector(motion_buffer)
                    append_csv(MOTION_CSV, list(features) + [motion_label])
                    motion_count[motion_label] = motion_count.get(motion_label, 0) + 1
                    print(f"Recorded MOTION sample for label '{motion_label}'. Total: {motion_count[motion_label]}")
                else:
                    print("Motion buffer too short or no label set, sample not recorded.")

                recording_motion = False
                motion_buffer = []

        if key == KEY_ESC:
            break
        elif key == KEY_TAB:
            mode = "MOTION" if mode == "STATIC" else "STATIC"
            print(f"Switched to {mode} mode")
        elif mode == "STATIC" and 97 <= key <= 122 and key != 106:  # a-y except j
            static_label = chr(key).upper()
            print(f"Set STATIC label to '{static_label}'")
        elif mode == "MOTION" and key in MOTION_LABELS:
            motion_label = MOTION_LABELS[key]
            print(f"Set MOTION label to '{motion_label}'")
        elif mode == "MOTION" and key == KEY_ENTER and not recording_motion:
            if motion_label is not None:
                recording_motion = True
                motion_buffer = []
                motion_record_start = now
                print(f"Started recording MOTION sample for label '{motion_label}'")
            else:
                print("No MOTION label set. Press 1, 2, or 3 to set a label before recording.")

        hud = f"Mode: {mode} - "
        if mode == "STATIC":
            hud += f" Label: {static_label} - Press SPACE to record"
            
        else:
            hud += f" Label: {motion_label} - Press ENTER to record"
            if recording_motion:
                hud += f" Recording..."
            hud += f" count:{motion_count}"
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    main()


   
