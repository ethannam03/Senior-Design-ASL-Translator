"""
camera_toggle.py

Standalone script, no models or hand detection involved -- just a webcam
feed you can turn on and off with ENTER.

How it works (power-saving version):
  - "Off" actually calls cap.release() -- this really shuts down the
    camera's video stream at the driver/hardware level, not just stops
    reading frames in Python. That's what cuts the sensor's own power
    draw, not just CPU usage from decoding frames nobody looks at.
  - "On" reopens the device with cv2.VideoCapture(0) again. This has a
    noticeable delay (often 0.3-1+ seconds depending on the webcam) while
    it reinitializes -- that's the tradeoff for genuinely powering the
    camera down in between, rather than just pausing the display.

Keys:
  ENTER   toggle camera on/off (release / reopen the device)
  ESC     quit
"""

import cv2
import numpy as np

KEY_ESC = 27
KEY_ENTER = 13
KEY_ENTER_ALT = 10  # some terminals/OSes report 10 (LF) instead of 13 (CR)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def main():
    camera_on = True
    cap = open_camera()

    # pre-built "off" frame -- just a solid dark screen with text, reused
    # every loop instead of rebuilding it each time
    off_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(off_frame, "Camera Off - press ENTER to resume",
                (30, FRAME_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)

    print("Started. ENTER = toggle camera, ESC = quit.")

    while True:
        if camera_on:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: Could not read frame from webcam")
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Camera On - press ENTER to pause",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
        else:
            # Camera is actually released at this point (see toggle logic
            # below) -- nothing to read, just show the "off" placeholder.
            frame = off_frame.copy()

        cv2.imshow("Camera Toggle", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == KEY_ESC:
            break
        elif key in (KEY_ENTER, KEY_ENTER_ALT):
            camera_on = not camera_on
            if camera_on:
                cap = open_camera()  # reinitializes the hardware -- the slow part
                print("Camera ON (reopening device...)")
            else:
                cap.release()  # actually powers down the video stream
                print("Camera OFF (device released)")

    if camera_on:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()