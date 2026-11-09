"""
ASL translator (ML models) - Raspberry Pi 5 version.

Keys:
  ENTER (once)   confirm the current letter - adds it to the word
  ENTER (twice)  speak the current word aloud (text-to-speech)
  SPACE          add a space
  BACKSPACE      delete last character
  0              clear word
  a              toggle AUTO-accept (letters are added automatically, as in
                 the original version) / MANUAL (you confirm with ENTER)
  ESC            quit
(Key presses only register when a video window is focused.)

"Current letter" is what the screens show as "Letter:" - a motion letter
(J/Z) that was just detected, which stays available for a couple of
seconds, or otherwise the steadiest recent static prediction. Neither needs
to reach the confidence thresholds (those only apply in AUTO mode), so
ENTER confirms whatever the model currently says. A single ENTER is
acted on after a short pause (DOUBLE_ENTER_SECONDS) so the program can tell
whether a second ENTER is coming; the letter captured is the one showing
when you pressed ENTER.

The same information is shown in the "Results" window and on the ILI9341
TFT (if attached).

RASPBERRY PI NOTES
  Setup (once):
      sudo apt update
      sudo apt install -y python3-venv python3-opencv python3-picamera2 espeak-ng
      python3 -m venv --system-site-packages .venv
      source .venv/bin/activate
      pip install mediapipe scikit-learn joblib pyttsx3
  Files expected next to this script:
      hand_features.py, tft_display.py (optional), static_model.joblib,
      motion_only_model.joblib
      (hand_landmarker.task is downloaded automatically if missing)

  Environment variables:
      ASL_CAMERA=usb|picamera|auto   camera source (default auto)
      ASL_CAM_INDEX=0                USB webcam index
      ASL_RES=640x480                capture size, WIDTHxHEIGHT
      ASL_DISPLAY=off                run without the ILI9341 TFT
      ASL_AUTO=on                    start in AUTO-accept mode (default off)
      ASL_CONFIRM_MIN=0              lowest confidence ENTER will confirm
                                     (default 0 = confirm anything)
      ASL_DOUBLE_ENTER=0.4           seconds allowed between the two ENTERs
  Example:
      ASL_CAMERA=usb python translator_pi.py

  ACCURACY CAVEATS when moving to a new machine:
   - Landmark x/y are normalized by image width and height separately, so
     the camera's aspect ratio affects the features. If you collected your
     training data on a 16:9 camera (most laptop webcams), capture at a
     16:9 size here too, e.g. ASL_RES=640x360 or 1280x720.
   - The J/Z motion model looks at the last MOTION_WINDOW *frames*. If the
     Pi's frame rate differs a lot from the machine you trained on, the
     motion features (path length, speeds) shift. Check the FPS shown on
     the camera window; retrain on the Pi if J/Z are unreliable.
   - The .joblib files must be loaded with a compatible scikit-learn
     version (see the error message if loading fails).
"""

import os
import sys

# Raspberry Pi OS (Bookworm) runs Wayland by default, and Qt-based OpenCV
# builds can fail to open windows there. Fall back to X11 (XWayland) unless
# the user already chose a platform. Must be set before cv2 is imported.
if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY") \
        and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
import threading
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
import numpy as np

try:
    import pyttsx3
except Exception:  # not installed / no speech backend
    pyttsx3 = None

from hand_features import (
    create_hand_landmarker, draw_landmarks, feature_vector,
    landmarks_to_array, normalize_landmarks, motion_vector,
    FINGERTIPS,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_MODEL_PATH = os.path.join(BASE_DIR, "static_model.joblib")
MOTION_MODEL_PATH = os.path.join(BASE_DIR, "motion_only_model.joblib")

# ---------------------------------------------------------------------------
# Camera / platform settings
# ---------------------------------------------------------------------------
CAMERA_SOURCE = os.environ.get("ASL_CAMERA", "auto").lower()
CAM_INDEX = int(os.environ.get("ASL_CAM_INDEX", "0"))


def _parse_res(text):
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 640, 480


FRAME_W, FRAME_H = _parse_res(os.environ.get("ASL_RES", "640x480"))

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

# "Current letter" (what ENTER confirms)
CURRENT_LETTER_WINDOW = 6       # recent static frames that get a majority vote
MOTION_DISPLAY_SECONDS = 2.0    # how long a detected J/Z stays confirmable
# Confirming ignores the STATIC/MOTION confidence thresholds above: ENTER
# takes whatever the model currently predicts, however sure it is. Set
# ASL_CONFIRM_MIN (e.g. 0.3) to ignore only the very unsure readings.
# AUTO mode still uses the full thresholds.
CONFIRM_MIN_CONFIDENCE = float(os.environ.get("ASL_CONFIRM_MIN", "0"))

# ENTER once = confirm, ENTER twice within this many seconds = speak
DOUBLE_ENTER_SECONDS = float(os.environ.get("ASL_DOUBLE_ENTER", "0.4"))

# Start in AUTO-accept mode (the original behavior) instead of MANUAL?
AUTO_ACCEPT_DEFAULT = os.environ.get("ASL_AUTO", "off").lower() == "on"

#results window appearance
RESULTS_WIDTH = 640
RESULTS_HEIGHT = 300


# ---------------------------------------------------------------------------
# Camera: Pi Camera Module (Picamera2) or USB webcam (OpenCV/V4L2)
# ---------------------------------------------------------------------------
class Camera:
    """One read() -> (ok, BGR frame) whether frames come from a ribbon-cable
    Pi camera (needs Picamera2 on the Pi 5) or a USB webcam."""

    def __init__(self, source="auto"):
        self.picam = None
        self.cap = None
        self.kind = None

        if source in ("auto", "picamera"):
            self._try_picamera()
        if self.kind is None and source in ("auto", "usb"):
            self._try_usb()
        if self.kind is None:
            sys.exit(
                "No camera could be opened.\n"
                "  - Pi Camera Module: check the ribbon cable, and that "
                "python3-picamera2 is installed (apt) and the venv was made "
                "with --system-site-packages.\n"
                "  - USB webcam: check it's plugged in (ls /dev/video*).\n"
                "  - Force one with ASL_CAMERA=usb or ASL_CAMERA=picamera."
            )
        print(f"Using camera: {self.kind}")

    def _try_picamera(self):
        try:
            from picamera2 import Picamera2
            if not Picamera2.global_camera_info():
                raise RuntimeError("no Pi camera detected")
            cam = Picamera2()
            # "RGB888" in Picamera2 is laid out as B,G,R in memory, which is
            # exactly the channel order OpenCV expects - no conversion.
            config = cam.create_preview_configuration(
                main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
            )
            cam.configure(config)
            cam.start()
            self.picam = cam
            self.kind = "Pi Camera (Picamera2)"
        except Exception as e:
            print(f"Picamera2 not used ({e}).")
            self.picam = None

    def _try_usb(self):
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        cap = cv2.VideoCapture(CAM_INDEX, backend)
        if not cap.isOpened():
            cap.release()
            return
        # MJPG lets most USB webcams deliver 30 fps; small buffer = less lag.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.kind = "USB webcam (OpenCV)"

    def read(self):
        if self.picam is not None:
            try:
                return True, self.picam.capture_array()
            except Exception:
                return False, None
        return self.cap.read()

    def release(self):
        if self.picam is not None:
            try:
                self.picam.stop()
                self.picam.close()
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------------------
# Text-to-speech, run on a background thread so the camera never freezes
# ---------------------------------------------------------------------------
class Speaker:
    """pyttsx3's runAndWait() blocks until speech finishes, which used to
    freeze the camera feed. Here it runs on its own thread, and a second
    request while it's still talking is ignored instead of piling up."""

    def __init__(self):
        self._busy = threading.Lock()
        if pyttsx3 is None:
            print("Speech disabled: pyttsx3 not installed "
                  "(pip install pyttsx3; sudo apt install espeak-ng).")

    def say(self, text):
        if pyttsx3 is None:
            print(f"(speech unavailable) {text}")
            return
        if not self._busy.acquire(blocking=False):
            return  # already speaking
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _run(self, text):
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"Speech failed: {e} "
                  "(is espeak-ng installed, and is the speaker set up as the "
                  "default audio output? try: aplay -l)")
        finally:
            self._busy.release()


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


def stable_static_letter(static_buffer):
    """Majority vote over the last few static predictions (any confidence
    above CONFIRM_MIN_CONFIDENCE), so the 'current letter' doesn't flicker
    frame to frame."""
    recent = [l for l, c in list(static_buffer)[-CURRENT_LETTER_WINDOW:]
              if c >= CONFIRM_MIN_CONFIDENCE]
    if not recent:
        return None
    letter, count = Counter(recent).most_common(1)[0]
    if count >= max(2, len(recent) // 2):
        return letter
    return None


def make_ui_state(static_letter, static_confidence, motion_letter, motion_confidence,
                  current_letter, confirmed_letter, auto_mode, word):
    """Everything the Results window and the TFT display show."""
    return {
        "static": (static_letter, round(float(static_confidence), 3)),
        "motion": (motion_letter, round(float(motion_confidence), 3)),
        "current": current_letter,
        "confirmed": confirmed_letter,
        "mode": "AUTO" if auto_mode else "MANUAL",
        "word": word,
    }


def build_results_frame(s):
    canvas = np.zeros((RESULTS_HEIGHT, RESULTS_WIDTH, 3), dtype=np.uint8)

    static_letter, static_confidence = s["static"]
    motion_letter, motion_confidence = s["motion"]

    static_text = (f"Static:  {static_letter}  ({static_confidence:.1%})"
                   if static_letter else "Static:  -")
    motion_text = (f"Motion:  {motion_letter}  ({motion_confidence:.1%})"
                   if motion_letter else "Motion:  -")
    status_text = (f"Letter: {s['current'] or '-'}    "
                   f"Last confirmed: {s['confirmed'] or '-'}    [{s['mode']}]")
    word_text = f"Word: {s['word']}"

    cv2.putText(canvas, static_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)
    cv2.putText(canvas, motion_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 165, 255), 2)
    cv2.putText(canvas, status_text, (20, 135), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 200, 0), 2)
    cv2.putText(canvas, word_text, (20, 210), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 2)
    cv2.putText(canvas, "ENTER confirm - ENTER x2 speak - BACKSPACE delete - 0 clear",
                (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    cv2.putText(canvas, "SPACE space - a auto/manual - ESC quit",
                (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    return canvas


def load_models():
    try:
        static_model = joblib.load(STATIC_MODEL_PATH)
        motion_model = joblib.load(MOTION_MODEL_PATH)
    except FileNotFoundError as e:
        sys.exit(f"Model file not found: {e.filename}\n"
                 "Copy static_model.joblib and motion_only_model.joblib into "
                 f"{BASE_DIR}")
    except Exception as e:
        sys.exit(
            f"Could not load the .joblib models: {e}\n"
            "This usually means the scikit-learn version on the Pi differs "
            "from the one that trained them. Compare 'pip show scikit-learn' "
            "on both machines and install the training version on the Pi "
            "(pip install scikit-learn==X.Y.Z), or retrain on the Pi."
        )
    return static_model, motion_model


def main():
    static_model, motion_model = load_models()
    landmarker = create_hand_landmarker()
    camera = Camera(CAMERA_SOURCE)
    speaker = Speaker()

    # Optional TFT display; the translator runs fine without it.
    tft = None
    if os.environ.get("ASL_DISPLAY", "on").lower() != "off":
        try:
            from tft_display import TFTDisplay
            tft = TFTDisplay()
            print("TFT display ready.")
        except Exception as e:
            print(f"TFT display not available ({e}). Continuing without it.")

    start_time = time.monotonic()
    last_timestamp_ms = -1
    last_frame_time = start_time
    fps = 0.0

    static_buffer = deque(maxlen=STATIC_WINDOW)
    motion_buffer = deque(maxlen=MOTION_WINDOW)

    word = ""
    state = "READY"
    cooldown_until = 0.0
    last_accepted_letter = None      # used by the AUTO-accept cooldown logic

    auto_mode = AUTO_ACCEPT_DEFAULT  # AUTO: letters added automatically
    last_confirmed_letter = None     # shown as "Last confirmed"
    pinned_motion_letter = None      # a J/Z that stays confirmable for a while
    pinned_motion_until = 0.0

    # ENTER handling: first press is held for DOUBLE_ENTER_SECONDS to see
    # whether a second press follows (-> speak) or not (-> confirm).
    pending_enter_at = None
    pending_enter_letter = None

    print(f"Mode: {'AUTO' if auto_mode else 'MANUAL'} "
          "(press 'a' to switch; ENTER confirms, ENTER twice speaks)")

    if tft:
        tft.update(make_ui_state(None, 0.0, None, 0.0, None, None, auto_mode, word))

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            print("ERROR: Could not read frame from camera")
            break
        frame = cv2.flip(frame, 1)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Monotonic clock: the Pi's wall clock can jump when it syncs with
        # the network, which would break MediaPipe's increasing timestamps
        # and the cooldown / double-ENTER timing below.
        now = time.monotonic()
        timestamp_ms = int((now - start_time) * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        dt = now - last_frame_time
        last_frame_time = now
        if dt > 0:
            fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 * (1.0 / dt)

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

            # state machine (only used in AUTO mode, but kept up to date)
            if state == "COOLDOWN" and now >= cooldown_until:
                if static_letter != last_accepted_letter:
                    
                    state = "READY"
                elif static_confidence < RESET_CONFIDENCE_DROP and energy < ENERGY_RELEASE_THRESHOLD:
                   
                    state = "READY"

            if auto_mode and state == "READY":
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
                    last_confirmed_letter = accepted
                    state = "COOLDOWN"
                    cooldown_until = now + COOLDOWN
                    static_buffer.clear()
                    motion_buffer.clear()
                    print(f"Accepted letter: {accepted}, current word: {word}")
        else:
            static_buffer.clear()
            motion_buffer.clear()

        # -- "current letter": what ENTER would confirm right now ----------
        # No confidence threshold here (see CONFIRM_MIN_CONFIDENCE): a motion
        # letter is any non-NONE motion prediction, and the static letter is
        # the steadiest recent prediction, falling back to this frame's.
        if motion_letter is not None and motion_confidence >= CONFIRM_MIN_CONFIDENCE:
            pinned_motion_letter = motion_letter
            pinned_motion_until = now + MOTION_DISPLAY_SECONDS
        if now < pinned_motion_until:
            current_letter = pinned_motion_letter
        else:
            current_letter = stable_static_letter(static_buffer)
            if (current_letter is None and static_letter is not None
                    and static_confidence >= CONFIRM_MIN_CONFIDENCE):
                current_letter = static_letter

        # -- single ENTER: window expired with no second press -> confirm --
        if pending_enter_at is not None and (now - pending_enter_at) > DOUBLE_ENTER_SECONDS:
            if pending_enter_letter:
                word += pending_enter_letter
                last_confirmed_letter = pending_enter_letter
                print(f"Confirmed letter: {pending_enter_letter}, current word: {word}")
            pending_enter_at = None
            pending_enter_letter = None

        #  keys 
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 8 or key == 127:  # backspace
            word = word[:-1]
            last_confirmed_letter = word[-1] if word and word[-1] != " " else None
        elif key == ord(" "):
            word += " "
        elif key == ord("0"):
            word = ""
            last_confirmed_letter = None
        elif key == ord("a"):
            auto_mode = not auto_mode
            state = "READY"
            static_buffer.clear()
            motion_buffer.clear()
            print(f"Mode: {'AUTO' if auto_mode else 'MANUAL'}")
        elif key in (13, 10):  # ENTER
            if pending_enter_at is not None and (now - pending_enter_at) <= DOUBLE_ENTER_SECONDS:
                # second press: speak the word instead of confirming a letter
                pending_enter_at = None
                pending_enter_letter = None
                if word.strip():
                    speaker.say(word)  # background thread; camera keeps going
                else:
                    print("Nothing to speak yet.")
            else:
                # first press: remember the letter showing right now and
                # wait briefly for a possible second press
                pending_enter_at = now
                pending_enter_letter = current_letter

        
        cv2.putText(frame, f"{fps:4.1f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        cv2.imshow("Camera", frame)  

        ui_state = make_ui_state(static_letter, static_confidence,
                                 motion_letter, motion_confidence,
                                 current_letter, last_confirmed_letter,
                                 auto_mode, word)
        cv2.imshow("Results", build_results_frame(ui_state))

        # The TFT shows exactly what the Results window shows.
        if tft:
            tft.update(ui_state)

    if tft:
        tft.close()
    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()