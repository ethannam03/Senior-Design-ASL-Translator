"""
ASL Fingerspelling Reader (Raspberry Pi 5 / desktop) — recognizes all 26 ASL
alphabet handshapes using MediaPipe's Tasks API (HandLandmarker), and shows
the live letter plus the confirmed word on an Adafruit ILI9341 TFT.

SETUP ON A RASPBERRY PI 5 (one time):
  1. Raspberry Pi OS 64-bit (Bookworm), then:
         sudo apt update
         sudo apt install -y python3-venv python3-opencv python3-picamera2
         python3 -m venv --system-site-packages .venv
         source .venv/bin/activate
         pip install mediapipe
     (--system-site-packages is what lets the venv see the apt-installed
      OpenCV and Picamera2.)
  2. The hand model is downloaded automatically on first run if it's
     missing. To do it by hand:
         wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  3. For the TFT display (optional - the script runs fine without it):
         sudo raspi-config nonint do_spi 0        # then reboot
         sudo apt install -y fonts-dejavu-core python3-pil
         pip install adafruit-circuitpython-rgb-display adafruit-blinka
     The display wiring (SPI0) and the board's IM1/IM2/IM3 = 3.3V, IM0 = GND
     setting must already be working (your display test shows colors).
  4. Run from the Pi's desktop (HDMI, VNC, or Pi Connect screen sharing):
         python MediaPipeASL.py

ENVIRONMENT VARIABLES:
  ASL_CAMERA=usb|picamera|auto   which camera to use (default auto: Pi
                                 Camera Module first, then USB webcam)
  ASL_CAM_INDEX=0                USB webcam index (default 0)
  ASL_DISPLAY=off                run without the TFT display
  Example:  ASL_CAMERA=usb python MediaPipeASL.py

  - 24 letters (everything except J and Z) are recognized from a single
    still handshape (finger-bend angles, thumb position/depth, fingertip
    spacing, and hand orientation).
  - J and Z are NOT static handshapes — they're defined by a short motion
    trace — so they're recognized separately by watching the fingertip's
    path over the last ~2 seconds and matching it against the expected
    shape:
        J: hold the "I" handshape (pinky up, thumb tucked) and draw a
           small hook/J in the air.
        Z: hold a pointing "index only" handshape and draw a Z (right,
           diagonal down-left, right) in the air.


Controls:
  - Show your hand to the camera, palm facing the camera works best.
  - For J/Z: hold the starting handshape, then draw the letter's motion
    in the air in front of the camera; hold briefly at the end.
  - The letter currently detected is shown live on screen.
  - Press SPACE to confirm the letter currently shown - it's locked in as
    "Last confirmed" and appended to the running "Word" on screen, both of
    which stay visible even after your hand moves or leaves the frame.
  - Press BACKSPACE to undo the last confirmed letter.
  - Press 'd' to toggle a debug overlay showing the raw angles/gaps/motion
    numbers (and FPS) behind the current decision - use this to tune
    thresholds for your own camera and hand (see TUNING NOTES below).
  - Press 't' to start/stop an ACCURACY TEST: it prompts you through the
    alphabet one letter at a time; sign the prompted letter and press
    SPACE to log whether the live reading matched. Tracks per-letter
    accuracy and what it misread each miss as, and auto-saves a CSV
    summary (handy for a report/presentation) when the test finishes, or
    any time you press 's' while in test mode.
  - Press 'n' (while testing) to skip the current letter without logging it.
  - Press 'q' to quit.
  (Key presses only register when the video window is focused.)


IMPORTANT ACCURACY NOTE:
Real ASL recognition systems are normally trained on thousands of labeled
hand images / landmark sets. This version instead uses hand-coded geometric
rules, which is inherently less accurate. Letters that share very similar
handshapes are still the most error-prone, especially:
  - The "closed fist" family: A, E, M, N, S, T (differ only in exact thumb
    placement/depth relative to the curled fingers). This version adds the
    landmark z-coordinate (rough relative depth) to help separate these,
    but MediaPipe's depth estimate is noisy on a single camera, so this
    remains the least reliable part of the classifier.
  - R (crossed fingers — hard to see reliably in a 2D projection).
  - G vs Q and K vs P (same finger shape, distinguished only by whether the
    hand is oriented pointing up or down).
  - J and Z depend on a clean, deliberate motion — fast or tiny motions,
    or moving the whole arm instead of just the hand/wrist, may not
    trigger reliably.
For best results: face your palm toward the camera, keep your hand well
lit, hold static shapes steady for a second, keep fingers you're NOT using
clearly curled into the palm, and for J/Z draw the motion at a natural,
deliberate pace roughly in place (don't walk the whole hand across frame).


TUNING NOTES:
Every threshold in this file (STRAIGHT/BENT below, the gap/distance
numbers inside classify_static_letter, and min_len/dy/dx_total inside
detect_hook_motion / detect_zigzag_motion) was picked without seeing your
camera, lighting, or hand - so if a specific letter or gesture consistently
misses, don't just re-read the code, watch the numbers:
  1. Press 'd' to turn on the debug overlay.
  2. Hold the shape (or run the gesture) and watch which number is sitting
     right on a threshold boundary, or on the wrong side of it entirely.
  3. Nudge that one constant in the direction that fixes it, save, rerun.
This is inherent to a hand-coded (not ML-trained) classifier: it will
never generalize perfectly to a new camera/hand without this kind of
per-setup calibration.


ACCURACY TEST MODE (press 't'):
Walks through all 26 letters in random order. For each, you sign the
prompted letter and press SPACE when ready; it logs whether the live
reading matched. Press 'n' to skip a letter without logging it. When you
finish the alphabet (or stop early with 't'), it writes two CSVs next to
this script:
  - asl_accuracy_summary_<timestamp>.csv: one row per letter with
    attempts, correct, accuracy %, and the most common thing it misread
    that letter as - the file you want for a chart or table in a report.
  - asl_accuracy_log_<timestamp>.csv: every single attempt with a
    timestamp - useful if you want to build your own confusion matrix or
    show results over time.
Press 's' any time during a test to save early without stopping it.
"""

import os
import sys

# Raspberry Pi OS (Bookworm) runs Wayland by default, and Qt-based OpenCV
# builds can fail to open windows there. Fall back to X11 (XWayland) unless
# the user already chose a platform. Must be set before cv2 is imported.
if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY") \
        and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time
import math
import csv
import random
import threading
import urllib.request
from collections import deque, Counter
from enum import IntEnum

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Frame size used for capture, processing, and display. 640x480 is a good
# fit for a Pi 5: hand landmarks don't need more resolution, and a smaller
# frame means less work per loop. All overlay positions scale from these.
FRAME_W, FRAME_H = 640, 480

# "auto" (Pi camera first, then USB), "usb", or "picamera".
CAMERA_SOURCE = os.environ.get("ASL_CAMERA", "auto").lower()
CAM_INDEX = int(os.environ.get("ASL_CAM_INDEX", "0"))

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# --- TFT display (ILI9341 on SPI0) ---
TFT_ENABLED = os.environ.get("ASL_DISPLAY", "on").lower() != "off"
TFT_CS = None          # None = CE0 (pin 24) driven automatically by the kernel.
                       # Or a Blinka pin name like "D5" if CS is on a plain GPIO.
TFT_DC = "D25"         # GPIO25, pin 22
TFT_RST = "D24"        # GPIO24, pin 18
TFT_ROTATION = 90      # 90 = 320x240 landscape; try 270 if upside down
TFT_BAUDRATE = 8_000_000   # raise toward 24_000_000 if your wiring is clean
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

if TFT_ENABLED:
    try:
        import board
        import digitalio
        from PIL import Image, ImageDraw, ImageFont
        from adafruit_rgb_display import ili9341
    except Exception as e:  # missing packages, Blinka platform problems, ...
        print(f"TFT display disabled (could not import display libraries: {e})")
        TFT_ENABLED = False


# ---------------------------------------------------------------------------
# Setup: MediaPipe Tasks HandLandmarker
# ---------------------------------------------------------------------------
def ensure_model():
    """Download the hand model next to the script if it isn't there yet."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    print(f"Model not found at {MODEL_PATH}; downloading (~8 MB)...")
    tmp_path = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        os.replace(tmp_path, MODEL_PATH)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        sys.exit(
            f"Could not download the model automatically ({e}).\n"
            "Download it manually with:\n"
            f"  wget -O hand_landmarker.task {MODEL_URL}"
        )


ensure_model()

landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)


class HandLandmark(IntEnum):
    """Same indices as the old mp.solutions HandLandmark enum, so the rest
    of the code (HL.WRIST.value, etc.) works unchanged."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_hand(image, lm):
    """Replacement for mp_drawing.draw_landmarks (Tasks API has no helper
    that works on plain landmark lists)."""
    h, w = image.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(image, p, 3, (0, 0, 255), -1)


# ---------------------------------------------------------------------------
# Camera: Pi Camera Module (Picamera2) or USB webcam (OpenCV/V4L2)
# ---------------------------------------------------------------------------
class Camera:
    """Gives the main loop one simple read() -> (ok, BGR frame) regardless
    of whether the frames come from a ribbon-cable Pi camera or a USB
    webcam. On a Pi 5 the ribbon cameras use libcamera, which OpenCV's
    VideoCapture can't read, so they have to go through Picamera2."""

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
            # "RGB888" in Picamera2 is laid out as B,G,R in memory, which
            # is exactly the channel order OpenCV expects - no conversion.
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
        # MJPG lets most USB webcams deliver 30 fps at 640x480 over USB;
        # raw YUYV often caps lower. Small buffer = less lag.
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


camera = Camera(CAMERA_SOURCE)


# ---------------------------------------------------------------------------
# TFT display (ILI9341) - drawn on a background thread
# ---------------------------------------------------------------------------
def _load_font(path, size):
    """TrueType font if available, otherwise PIL's tiny built-in font."""
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


class TFTDisplay:
    """Shows the live letter (big) and the confirmed word (bottom).

    A full-screen refresh over SPI takes on the order of 100+ ms, which
    would stall the camera loop if done inline. So update() only stores the
    latest (letter, word) and wakes a worker thread that does the drawing;
    if several updates arrive while it's busy, only the newest is drawn."""

    def __init__(self):
        def pin(name):
            return digitalio.DigitalInOut(getattr(board, name))

        cs = pin(TFT_CS) if TFT_CS else None
        self.disp = ili9341.ILI9341(
            board.SPI(), rotation=TFT_ROTATION, cs=cs,
            dc=pin(TFT_DC), rst=pin(TFT_RST), baudrate=TFT_BAUDRATE,
        )

        # With 90/270 rotation the library reports width/height swapped.
        if self.disp.rotation % 180 == 90:
            self.width, self.height = self.disp.height, self.disp.width
        else:
            self.width, self.height = self.disp.width, self.disp.height

        self.font_title = _load_font(FONT_BOLD, 18)
        self.font_letter = _load_font(FONT_BOLD, 140)
        self.font_word = _load_font(FONT_BOLD, 30)

        self._fill((0, 0, 0))  # also proves SPI works before the thread starts

        self._lock = threading.Lock()
        self._pending = None
        self._wake = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, letter, word):
        """Cheap and non-blocking; safe to call every frame."""
        with self._lock:
            self._pending = (letter, word)
        self._wake.set()

    def _fill(self, rgb):
        self.disp.image(Image.new("RGB", (self.width, self.height), rgb))

    def _run(self):
        last = None
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                return
            with self._lock:
                state = self._pending
            if state is None or state == last:
                continue
            try:
                self._draw(*state)
                last = state
            except Exception as e:
                print(f"TFT drawing stopped after an error: {e}")
                return

    def _draw(self, letter, word):
        img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        d = ImageDraw.Draw(img)

        # Title bar
        d.rectangle((0, 0, self.width, 28), fill=(30, 30, 90))
        d.text((8, 4), "ASL Reader", font=self.font_title, fill=(255, 255, 255))

        # Big centered letter ("?" in orange when nothing is detected)
        text = letter if letter else "?"
        color = (0, 255, 0) if letter else (255, 165, 0)
        l, t, r, b = d.textbbox((0, 0), text, font=self.font_letter)
        x = (self.width - (r - l)) // 2 - l
        y = 28 + (self.height - 28 - 50 - (b - t)) // 2 - t
        d.text((x, y), text, font=self.font_letter, fill=color)

        # Confirmed word along the bottom; drop the oldest letters if it
        # won't fit.
        shown = word
        while shown and d.textlength(shown, font=self.font_word) > self.width - 20:
            shown = shown[1:]
        d.line((0, self.height - 48, self.width, self.height - 48), fill=(80, 80, 80))
        d.text((10, self.height - 42), shown or "-", font=self.font_word, fill=(255, 200, 0))

        self.disp.image(img)

    def close(self):
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=2)
        try:
            self._fill((0, 0, 0))
        except Exception:
            pass


tft = None
if TFT_ENABLED:
    try:
        tft = TFTDisplay()
        print("TFT display ready.")
    except Exception as e:
        print(f"TFT display not available ({e}). Continuing without it.")
        tft = None


# Rolling buffer of recent classifications, used to smooth out single-frame
# jitter / misreads before we show or commit a letter.
RECENT_WINDOW = 6
recent_letters = deque(maxlen=RECENT_WINDOW)


# J/Z are one-off "events" from the motion detector, not a held pose, so
# they'd otherwise show for a single frame and vanish before you could
# react. This keeps whichever one just fired pinned on screen for a bit,
# overriding whatever the static classifier reads from your hand's
# resting position during the cooldown right after.
MOTION_DISPLAY_SECONDS = 1.5
last_motion_display = None
last_motion_display_until = 0.0




# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)




def angle_at(p1, p2, p3):
    """Angle (degrees) at point p2, between rays p2->p1 and p2->p3.
    ~180 degrees means the joint is straight; small angles mean it's bent."""
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 180.0
    cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))




def hand_orientation(wrist, middle_mcp):
    """Direction from wrist to the base of the fingers, relative to
    straight 'up' in the (already left-right flipped) image. Cleaner
    replacement for the previous fake-point trick.
    NOTE: this got widened to 45/135 in an earlier pass to make H easier
    to hit, but that made "down" harder to reach too - which quietly
    broke Q and P, since they're only distinguished from G/K by
    orientation=="down". Reverted to 55/125 here; H is fixed via the
    thumb check in its own rule instead (see classify_static_letter)."""
    vx = middle_mcp.x - wrist.x
    vy = middle_mcp.y - wrist.y
    mag = math.hypot(vx, vy) or 1e-6
    vx, vy = vx / mag, vy / mag
    # "up" in image coordinates is (0, -1) since y grows downward
    cos_a = max(-1.0, min(1.0, vx * 0 + vy * -1))
    angle = math.degrees(math.acos(cos_a))
    if angle < 55:
        return "up"
    elif angle > 125:
        return "down"
    return "side"




# Curl-angle thresholds (angle at the PIP joint of each finger)
STRAIGHT = 155   # angle above this => finger is essentially straight
BENT = 95        # angle below this => finger is essentially fully curled
# Between BENT and STRAIGHT => "half-curled" (hook / claw shapes)




def finger_curl(lm, mcp, pip, tip):
    return angle_at(lm[mcp], lm[pip], lm[tip])




def curl_state(angle):
    if angle > STRAIGHT:
        return "straight"
    if angle < BENT:
        return "bent"
    return "half"




def extract_features(hand_landmarks):
    # Tasks API returns a plain list of landmarks (no .landmark attribute)
    lm = hand_landmarks
    HL = HandLandmark


    wrist = lm[HL.WRIST.value]
    index_mcp = lm[HL.INDEX_FINGER_MCP.value]
    middle_mcp = lm[HL.MIDDLE_FINGER_MCP.value]
    ring_mcp = lm[HL.RING_FINGER_MCP.value]
    pinky_mcp = lm[HL.PINKY_MCP.value]


    hand_size = dist(wrist, middle_mcp) or 1e-6


    idx_angle = finger_curl(lm, HL.INDEX_FINGER_MCP.value, HL.INDEX_FINGER_PIP.value, HL.INDEX_FINGER_TIP.value)
    mid_angle = finger_curl(lm, HL.MIDDLE_FINGER_MCP.value, HL.MIDDLE_FINGER_PIP.value, HL.MIDDLE_FINGER_TIP.value)
    ring_angle = finger_curl(lm, HL.RING_FINGER_MCP.value, HL.RING_FINGER_PIP.value, HL.RING_FINGER_TIP.value)
    pinky_angle = finger_curl(lm, HL.PINKY_MCP.value, HL.PINKY_PIP.value, HL.PINKY_TIP.value)


    thumb_tip = lm[HL.THUMB_TIP.value]
    thumb_ip = lm[HL.THUMB_IP.value]
    thumb_mcp = lm[HL.THUMB_MCP.value]
    thumb_angle = angle_at(thumb_mcp, thumb_ip, thumb_tip)


    # Thumb "extended away from palm" check (sideways motion, not curl-based)
    thumb_extended = dist(thumb_tip, pinky_mcp) > dist(wrist, pinky_mcp) * 0.6


    index_tip = lm[HL.INDEX_FINGER_TIP.value]
    middle_tip = lm[HL.MIDDLE_FINGER_TIP.value]
    ring_tip = lm[HL.RING_FINGER_TIP.value]
    pinky_tip = lm[HL.PINKY_TIP.value]


    orientation = hand_orientation(wrist, middle_mcp)


    # Rough relative depth: is the thumb tip closer to the camera (more
    # "in front of") the curled fingertips, or farther away ("tucked
    # behind" them)? MediaPipe's z is relative to the wrist and noisy on
    # a single RGB camera, so this is only used as a secondary signal.
    fingertip_z_avg = (index_tip.z + middle_tip.z + ring_tip.z + pinky_tip.z) / 4.0
    thumb_in_front = (thumb_tip.z - fingertip_z_avg) < -0.015


    return {
        "lm": lm, "hand_size": hand_size,
        "idx": curl_state(idx_angle), "idx_angle": idx_angle,
        "mid": curl_state(mid_angle), "mid_angle": mid_angle,
        "ring": curl_state(ring_angle), "ring_angle": ring_angle,
        "pinky": curl_state(pinky_angle), "pinky_angle": pinky_angle,
        "thumb_angle": thumb_angle, "thumb_extended": thumb_extended,
        "thumb_in_front": thumb_in_front,
        "thumb_tip": thumb_tip, "index_tip": index_tip, "middle_tip": middle_tip,
        "ring_tip": ring_tip, "pinky_tip": pinky_tip,
        "index_mcp": index_mcp, "middle_mcp": middle_mcp, "ring_mcp": ring_mcp,
        "pinky_mcp": pinky_mcp, "wrist": wrist,
        "orientation": orientation,
    }




def gap(f, a, b):
    """Normalized distance between two fingertip keys, scaled by hand size."""
    return dist(f[a], f[b]) / f["hand_size"]




# H/U vs V/K/P is decided entirely by one number (the index-middle
# fingertip gap), and "fingers together" vs "fingers apart" is a
# continuum, not a clean binary - so a gap sitting near the old single
# cutoff (0.28) would flicker between H and V on ordinary frame noise.
# Splitting it into two cutoffs with a dead zone between them, and
# "sticking" with whichever side we last decided while inside that dead
# zone, stops that flicker.
H_V_GAP_LOW = 0.24   # below this: definitely together -> H/U branch
H_V_GAP_HIGH = 0.34  # at/above this: definitely apart -> V/K/P branch
_last_finger_gap_branch = "close"  # sticky state between frames




# ---------------------------------------------------------------------------
# Static ASL letter classification (24 letters: no J, no Z)
# ---------------------------------------------------------------------------
def classify_static_letter(f):
    global _last_finger_gap_branch
    idx, mid, ring, pinky = f["idx"], f["mid"], f["ring"], f["pinky"]
    thumb_ext = f["thumb_extended"]


    idx_straight = idx == "straight"
    mid_straight = mid == "straight"
    ring_straight = ring == "straight"
    pinky_straight = pinky == "straight"
    idx_bent = idx == "bent"
    mid_bent = mid == "bent"
    ring_bent = ring == "bent"
    pinky_bent = pinky == "bent"


    all_bent = idx_bent and mid_bent and ring_bent and pinky_bent
    all_straight = idx_straight and mid_straight and ring_straight and pinky_straight


    idx_mid_gap = gap(f, "index_tip", "middle_tip")
    thumb_idx_gap = gap(f, "thumb_tip", "index_tip")
    thumb_mid_gap = gap(f, "thumb_tip", "middle_tip")


    # --- Distinctive single-finger / spread shapes first (most reliable) ---


    # Y: thumb and pinky out, everything else curled ("hang loose")
    if thumb_ext and pinky_straight and idx_bent and mid_bent and ring_bent:
        return "Y"


    # I: pinky only, thumb tucked in
    if pinky_straight and idx_bent and mid_bent and ring_bent and not thumb_ext:
        return "I"


    # L: index up, thumb out roughly perpendicular to it
    if idx_straight and mid_bent and ring_bent and pinky_bent and thumb_ext:
        thumb_index_angle = angle_at(f["thumb_tip"], f["wrist"], f["index_tip"])
        if thumb_index_angle > 55:
            return "L"


    # G: index + thumb both extended, roughly parallel, others curled
    if idx_straight and mid_bent and ring_bent and pinky_bent and thumb_ext:
        return "Q" if f["orientation"] == "down" else "G"


    # R: index + middle straight but crossed. The reliable signal isn't
    # which way each finger leans from its own knuckle (that's mostly
    # just hand tilt, not crossing) - it's whether index and middle swap
    # left/right ORDER between their knuckles and their tips. If finger A
    # is left-of finger B at the base but right-of it at the tip, they've
    # crossed. Still the least reliable check in this set (2D projection
    # of two thin overlapping fingers), but this is a cleaner signal than
    # the per-finger tilt comparison it replaces.
    # NOTE: this check now runs BEFORE the H/U/V/K/P block below. It used
    # to come after it, but that block matches the same finger pattern and
    # always returns a letter, so R was unreachable.
    if idx_straight and mid_straight and ring_bent and pinky_bent and idx_mid_gap < 0.22:
        order_at_base = f["index_mcp"].x - f["middle_mcp"].x
        order_at_tip = f["index_tip"].x - f["middle_tip"].x
        # Require a real, measurable left-right gap at the base before
        # trusting the sign flip - if index_mcp and middle_mcp are
        # nearly the same x (hand viewed close to edge-on), the sign is
        # just noise and shouldn't be allowed to decide anything.
        if abs(order_at_base) > 0.01 and order_at_base * order_at_tip < 0:
            return "R"


    # H/U (fingers together) vs V/K/P (fingers apart). See H_V_GAP_LOW/HIGH
    # above: outside the dead zone we commit to a branch and remember it;
    # inside it, we stick with whatever we last committed to instead of
    # re-deciding from a borderline number every single frame.
    if idx_straight and mid_straight and ring_bent and pinky_bent:
        if idx_mid_gap < H_V_GAP_LOW:
            _last_finger_gap_branch = "close"
        elif idx_mid_gap >= H_V_GAP_HIGH:
            _last_finger_gap_branch = "spread"
        # else: ambiguous - leave _last_finger_gap_branch as it was


        if _last_finger_gap_branch == "close":
            # H and U are the SAME handshape and only differ by
            # orientation - that's the only reliable signal here.
            if f["orientation"] == "side":
                return "H"
            return "U"
        else:
            # K/P: thumb pokes up near the BASE of index & middle (close
            # to the middle knuckle), not near the middle FINGERTIP -
            # thumb_mid_gap measures tip-to-tip, which stays large here
            # since the middle finger is extended far away from where the
            # thumb actually sits. That mismatch was the bug behind "P
            # always reads as V." thumb_ext (meant for a thumb splayed
            # sideways, like in L/Y) doesn't fit either, since K/P's
            # thumb pokes forward through the gap, not out to the side -
            # dropped that requirement too.
            thumb_to_middle_mcp = dist(f["thumb_tip"], f["middle_mcp"]) / f["hand_size"]
            thumb_between = thumb_to_middle_mcp < 0.5
            if thumb_between:
                return "P" if f["orientation"] == "down" else "K"
            return "V"


    # W: index, middle, ring all straight & spread; pinky curled
    if idx_straight and mid_straight and ring_straight and pinky_bent:
        return "W"


    # D: index straight & up alone, thumb touches middle finger (small circle)
    if idx_straight and mid_bent and ring_bent and pinky_bent and thumb_mid_gap < 0.35:
        return "D"


    # F: thumb+index pinch together, middle/ring/pinky straight up
    if thumb_idx_gap < 0.35 and mid_straight and ring_straight and pinky_straight:
        return "F"


    # X: index hooked (half-curled) at the first knuckle, rest in a fist
    if idx == "half" and mid_bent and ring_bent and pinky_bent:
        return "X"


    # B: all four fingers straight & together, thumb folded across palm
    if all_straight and not thumb_ext:
        return "B"


    # C: fingers gently curved (half), thumb also curved, not touching tips
    if idx == "half" and mid == "half" and ring == "half" and pinky == "half":
        avg_thumb_gap = (thumb_idx_gap + thumb_mid_gap) / 2
        if avg_thumb_gap > 0.4:
            return "C"
        return "O"  # curved AND fingertips gathered near thumb -> closed circle


    # --- Fist family: A, E, M, N, S, T ---------------------------------
    # All six look like a closed fist and differ mainly in exactly where
    # the thumb sits. The previous version picked M/N/T by checking each
    # one's own distance threshold in a fixed order - which meant if N's
    # thumb position happened to also satisfy M's threshold (easy, since
    # they're adjacent knuckles), M always won just because it was
    # checked first. This version instead asks "which knuckle is the
    # thumb tip actually CLOSEST to" directly: that tells you how many
    # fingers are curled over the thumb (T = over just the index-row gap,
    # N = under index+middle, M = under index+middle+ring), which is a
    # comparison rather than an independent guess per letter. Depth (z)
    # still separates S (thumb wraps in FRONT of the curled fingers) from
    # the rest (thumb tucked BEHIND them). Still the least reliable part
    # of the classifier - treat results here as informed guesses.
    if all_bent:
        thumb_tip = f["thumb_tip"]
        thumb_in_front = f["thumb_in_front"]


        d_idx_mcp = dist(thumb_tip, f["index_mcp"]) / f["hand_size"]
        d_mid_mcp = dist(thumb_tip, f["middle_mcp"]) / f["hand_size"]
        d_ring_mcp = dist(thumb_tip, f["ring_mcp"]) / f["hand_size"]
        d_pinky_mcp = dist(thumb_tip, f["pinky_mcp"]) / f["hand_size"]


        # S and T both put the thumb tip near the index knuckle, so
        # position alone can't split them - the real difference is how
        # the thumb itself is held: T keeps it relatively straight,
        # poking through the index/middle gap; S curls it down flatter,
        # wrapped across the front of the fist. thumb_angle (straight
        # near 180, curled lower) captures that.
        if thumb_in_front and d_idx_mcp < 0.55 and f["thumb_angle"] < 130:
            # curled flat across the front of the curled fingers -> S
            return "S"


        # Nearest knuckle row wins, instead of first-match-wins.
        candidates = [("T", d_idx_mcp), ("N", d_mid_mcp), ("M", d_ring_mcp), ("M", d_pinky_mcp)]
        nearest_label, nearest_dist = min(candidates, key=lambda pair: pair[1])


        if nearest_dist < 0.5:
            if nearest_label == "T":
                # nearest the index knuckle and didn't get claimed by the
                # curled-flat S check above -> relatively straight thumb
                # poking through the index/middle gap -> T
                return "T"
            # Real M/N hold the thumb fairly STRAIGHT, tucked flat under
            # the fingers. E's thumb is CURLED/bent, with the fingertips
            # resting on top of it. Position alone (nearest knuckle) was
            # matching E's thumb to M/N just because it happened to land
            # near that knuckle - requiring a straighter thumb here stops
            # a curled E thumb from being claimed as M/N before E's own
            # check below ever gets a chance.
            if nearest_label in ("N", "M") and not thumb_in_front and f["thumb_angle"] > 110:
                # thumb tucked behind the index+middle (N) or
                # index+middle+ring (M) knuckles -> N / M
                return nearest_label


        if thumb_ext and f["thumb_angle"] > 140:
            # thumb resting alongside the fist, roughly straight -> A
            return "A"
        # fingertips curled down close to the thumb, thumb bent under -> E
        return "E"


    return None  # unclear / transitional handshape




# ---------------------------------------------------------------------------
# Motion tracking for J and Z
# ---------------------------------------------------------------------------
# Trace points older than this are dropped, so a long hold of the starting
# handshape can't push the trace past the max_duration checks below (which
# would make J/Z impossible to trigger until the shape broke and reset).
MAX_TRACE_SECONDS = 2.0


class MotionTracker:
    """Tracks a single fingertip's position (normalized by hand size,
    relative to its position when tracking started) over a short rolling
    time window, so we can look at the *shape* of the path it drew."""


    def __init__(self, maxlen=60):
        self.buffer = deque(maxlen=maxlen)  # (t, x, y, hand_size)


    def update(self, point, hand_size, t):
        self.buffer.append((t, point.x, point.y, hand_size))
        # Drop points that have aged out of the time window.
        while self.buffer and (t - self.buffer[0][0]) > MAX_TRACE_SECONDS:
            self.buffer.popleft()


    def clear(self):
        self.buffer.clear()


    def normalized_path(self):
        if not self.buffer:
            return []
        _, x0, y0, s0 = self.buffer[0]
        s0 = s0 or 1e-6
        return [((x - x0) / s0, (y - y0) / s0) for (_, x, y, _s) in self.buffer]


    def duration(self):
        if len(self.buffer) < 2:
            return 0.0
        return self.buffer[-1][0] - self.buffer[0][0]


    def path_length(self):
        pts = self.normalized_path()
        return sum(
            math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            for i in range(len(pts) - 1)
        )




def smooth_series(values, k=2):
    """Simple moving average. Raw per-frame landmark positions are noisy
    enough that looking at the *shape* of a path frame-to-frame (e.g. is
    it curving smoothly one way, or wiggling back and forth) picks up
    that noise as if it were real direction changes. Smoothing first
    fixes that - used by both hook and zigzag detection below."""
    out = []
    n = len(values)
    for i in range(n):
        window = values[max(0, i - k):i + k + 1]
        out.append(sum(window) / len(window))
    return out




def path_turn_stats(pts, eps=2e-3):
    """Returns (sign_changes, total_turn) for a path: how many times its
    curvature direction flipped, and net signed curvature overall. Call
    on an already-smoothed path. eps ignores near-zero cross products
    from essentially-straight segments, so leftover jitter after
    smoothing doesn't still get counted as a direction flip."""
    total_turn = 0.0
    sign_changes = 0
    last_sign = 0
    for i in range(1, len(pts) - 1):
        v1 = (pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        v2 = (pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        total_turn += cross
        if abs(cross) > eps:
            sign = 1 if cross > 0 else -1
            if last_sign != 0 and sign != last_sign:
                sign_changes += 1
            last_sign = sign
    return sign_changes, total_turn




def detect_hook_motion(tracker, min_len=0.4, min_duration=0.12, max_duration=3.0):
    """J: a single smooth hook - net downward travel, mostly one
    continuous turn (not a zigzag), enough path length to be a
    deliberate motion rather than jitter.
    The path is smoothed before judging its curvature (see
    smooth_series/path_turn_stats) - without that, ordinary per-frame
    landmark noise causes several spurious direction flips even during a
    genuinely clean hook, which is why this used to misfire so often.
    TUNE: if J still won't fire, lower min_len / the dy check below; if it
    fires on ordinary hand jitter, raise them. Check the debug overlay
    ('d' key) to see the live path_length/duration/dy/turns while you
    gesture."""
    pts = tracker.normalized_path()
    if len(pts) < 5:
        return False
    if not (min_duration <= tracker.duration() <= max_duration):
        return False
    if tracker.path_length() < min_len:
        return False


    dy = pts[-1][1] - pts[0][1]
    if dy < 0.08:  # hook must move downward overall
        return False


    xs = smooth_series([p[0] for p in pts], k=2)
    ys = smooth_series([p[1] for p in pts], k=2)
    smoothed = list(zip(xs, ys))


    sign_changes, total_turn = path_turn_stats(smoothed)
    return sign_changes <= 2 and abs(total_turn) > 0.03




def detect_zigzag_motion(tracker, min_len=0.5, min_duration=0.15, max_duration=3.2):
    """Z: two clear reversals in horizontal direction (right, diagonal
    down-left, right again), with net downward drift and enough
    horizontal travel to be deliberate.
    TUNE: same idea as detect_hook_motion - watch the debug overlay while
    drawing a Z and adjust min_len / dx_total / dy_total to match what you
    actually see for path_length and net displacement."""
    pts = tracker.normalized_path()
    if len(pts) < 6:
        return False
    if not (min_duration <= tracker.duration() <= max_duration):
        return False
    if tracker.path_length() < min_len:
        return False


    dx_total = pts[-1][0] - pts[0][0]
    dy_total = pts[-1][1] - pts[0][1]
    if abs(dx_total) < 0.15 or dy_total < 0.05:
        return False


    xs = [p[0] for p in pts]
    smoothed = smooth_series(xs, k=2)


    reversals = 0
    last_dir = 0
    for i in range(1, len(smoothed)):
        d = smoothed[i] - smoothed[i - 1]
        if abs(d) < 1e-3:
            continue
        direction = 1 if d > 0 else -1
        if last_dir != 0 and direction != last_dir:
            reversals += 1
        last_dir = direction


    return reversals >= 2




def is_j_candidate_shape(f):
    """Base 'I'-family shape (pinky up, rest curled) regardless of thumb
    position, since the thumb crosses during the J motion itself.
    Uses 'not straight' rather than requiring exactly 'bent' for the
    curled fingers - mid-motion blur often reads a curled finger as the
    in-between 'half' state rather than fully 'bent', and that shouldn't
    be treated as the hand having changed shape."""
    return f["pinky"] == "straight" and f["idx"] != "straight" and f["mid"] != "straight" and f["ring"] != "straight"




def is_z_candidate_shape(f):
    """Base pointing shape (index up, rest curled), regardless of thumb,
    since D/G/L/Q also start this way — motion (or lack of it) decides.
    Same 'not straight' tolerance as is_j_candidate_shape, for the same
    reason."""
    return f["idx"] == "straight" and f["mid"] != "straight" and f["ring"] != "straight" and f["pinky"] != "straight"




j_tracker = MotionTracker(maxlen=120)
z_tracker = MotionTracker(maxlen=120)
motion_cooldown_until = 0.0  # suppress re-triggering right after a J/Z fires
MOTION_COOLDOWN_SECONDS = 0.8


# How much real time (not frame count) we tolerate the hand reading as
# "wrong shape" mid-gesture before wiping the trace. Frame count doesn't
# work for this because it means something different on every camera -
# 8 frames is nothing on a slow camera and a long time on a fast one.
# Time is what actually matters: while your hand is really moving through
# a J or Z, its projected finger angles wobble from viewing-angle changes
# alone even though you never left the handshape. Without this tolerance
# the trace gets wiped before the gesture finishes and J/Z rarely fire.
MOTION_SHAPE_LEEWAY_SECONDS = 0.5
j_last_good_time = None
z_last_good_time = None




def classify_letter(f, now):
    """Full 26-letter classification: checks J/Z motion first (when the
    hand is in a candidate starting shape), otherwise falls back to the
    static 24-letter classifier."""
    global motion_cooldown_until, j_last_good_time, z_last_good_time


    in_cooldown = now < motion_cooldown_until


    if in_cooldown:
        j_tracker.clear()
        z_tracker.clear()
        j_last_good_time = None
        z_last_good_time = None
    else:
        # Feed the trackers based on current candidate shape, but tolerate
        # a short stretch of real time with mismatched frames mid-gesture
        # rather than wiping the whole trace on the first wobble.
        if is_j_candidate_shape(f):
            j_last_good_time = now
            j_tracker.update(f["pinky_tip"], f["hand_size"], now)
        else:
            if j_last_good_time is None or (now - j_last_good_time) > MOTION_SHAPE_LEEWAY_SECONDS:
                j_tracker.clear()
                j_last_good_time = None
            else:
                j_tracker.update(f["pinky_tip"], f["hand_size"], now)


        if is_z_candidate_shape(f):
            z_last_good_time = now
            z_tracker.update(f["index_tip"], f["hand_size"], now)
        else:
            if z_last_good_time is None or (now - z_last_good_time) > MOTION_SHAPE_LEEWAY_SECONDS:
                z_tracker.clear()
                z_last_good_time = None
            else:
                z_tracker.update(f["index_tip"], f["hand_size"], now)


    if not in_cooldown:
        if detect_zigzag_motion(z_tracker):
            z_tracker.clear()
            motion_cooldown_until = now + MOTION_COOLDOWN_SECONDS
            return "Z"
        if detect_hook_motion(j_tracker):
            j_tracker.clear()
            motion_cooldown_until = now + MOTION_COOLDOWN_SECONDS
            return "J"


    return classify_static_letter(f)




# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def stable_letter():
    """Majority vote over the recent-letters window; ignores None frames
    unless that's all we have."""
    non_none = [l for l in recent_letters if l is not None]
    if not non_none:
        return None
    most_common, count = Counter(non_none).most_common(1)[0]
    # require it to appear in at least half the non-empty recent frames
    if count >= max(2, len(non_none) // 2):
        return most_common
    return None




debug_mode = False  # toggle with 'd' - shows the raw numbers behind each decision
current_fps = 0.0   # smoothed frames/second, shown in the debug overlay


# Confirm-and-build state: SPACE locks in whatever letter is currently
# showing (so you control timing instead of it auto-committing), appends
# it to a running word, and keeps it visible even after your hand moves
# or leaves the frame. BACKSPACE undoes the last confirm.
last_confirmed_letter = None
confirmed_word = ""


# ---------------------------------------------------------------------------
# Accuracy test mode - prompts through the alphabet, logs whether the live
# reading matched the prompted letter each time you press SPACE, and can
# export a CSV summary (per-letter accuracy + what it got confused with).
# Built for quickly generating real numbers for a report/presentation
# rather than relying on a gut feel of "it works pretty well."
# ---------------------------------------------------------------------------
TEST_ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


test_mode = False
test_order = []          # shuffled per test run so repeats don't memorize order
test_index = 0
test_results = {}         # letter -> {"attempts": int, "correct": int, "confusions": Counter()}
test_log = []              # list of (timestamp, target, read_as, correct) for a full event log
test_start_time = None
test_last_saved_path = None
test_complete = False
test_last_feedback = None        # short-lived "CORRECT" / "Read as X" message
test_last_feedback_until = 0.0   # compared against time.monotonic()




def reset_test():
    global test_order, test_index, test_results, test_log, test_start_time
    global test_last_saved_path, test_complete, test_last_feedback
    test_order = TEST_ALPHABET[:]
    random.shuffle(test_order)
    test_index = 0
    test_results = {letter: {"attempts": 0, "correct": 0, "confusions": Counter()} for letter in TEST_ALPHABET}
    test_log = []
    test_start_time = time.time()
    test_last_saved_path = None
    test_complete = False
    test_last_feedback = None




def current_test_target():
    if 0 <= test_index < len(test_order):
        return test_order[test_index]
    return None




def log_test_attempt(target, read_as):
    """Records one attempt against the prompted letter, then advances to
    the next prompt. read_as can be None (nothing confidently detected)."""
    global test_index
    correct = (read_as == target)
    test_results[target]["attempts"] += 1
    if correct:
        test_results[target]["correct"] += 1
    else:
        test_results[target]["confusions"][read_as if read_as else "(nothing)"] += 1
    test_log.append((time.time(), target, read_as if read_as else "", correct))
    test_index += 1




def save_test_results():
    """Writes two CSVs: a per-letter summary (great for a bar chart of
    accuracy by letter) and a full attempt-by-attempt log (great for a
    confusion matrix or just showing your work). Returns the summary
    path, or None if there's nothing to save yet."""
    total_attempts = sum(r["attempts"] for r in test_results.values())
    if total_attempts == 0:
        return None


    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(out_dir, f"asl_accuracy_summary_{stamp}.csv")
    log_path = os.path.join(out_dir, f"asl_accuracy_log_{stamp}.csv")


    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["letter", "attempts", "correct", "accuracy_pct", "top_confusion", "top_confusion_count"])
        for letter in TEST_ALPHABET:
            r = test_results[letter]
            attempts, correct = r["attempts"], r["correct"]
            accuracy_pct = round(100 * correct / attempts, 1) if attempts else ""
            if r["confusions"]:
                top_label, top_count = r["confusions"].most_common(1)[0]
            else:
                top_label, top_count = "", ""
            writer.writerow([letter, attempts, correct, accuracy_pct, top_label, top_count])


        overall_attempts = total_attempts
        overall_correct = sum(r["correct"] for r in test_results.values())
        overall_pct = round(100 * overall_correct / overall_attempts, 1) if overall_attempts else 0
        writer.writerow([])
        writer.writerow(["OVERALL", overall_attempts, overall_correct, overall_pct, "", ""])


    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "target_letter", "read_as", "correct"])
        for ts, target, read_as, correct in test_log:
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)), target, read_as, correct])


    return summary_path




def _j_turn_debug_str():
    """Live curvature stats for the J tracker's current trace, shown in
    the debug overlay so you can see directly whether what you're
    drawing reads as one smooth turn (what detect_hook_motion wants) or
    several (what gets rejected), instead of guessing from whether J
    fired or not."""
    pts = j_tracker.normalized_path()
    if len(pts) < 5:
        return "n/a (too few points yet)"
    xs = smooth_series([p[0] for p in pts], k=2)
    ys = smooth_series([p[1] for p in pts], k=2)
    sign_changes, total_turn = path_turn_stats(list(zip(xs, ys)))
    dy = pts[-1][1] - pts[0][1]
    return f"sign_changes={sign_changes} total_turn={total_turn:.3f} dy={dy:.2f}"




def draw_debug_overlay(image, f):
    """Live readout of everything the classifier looked at, so you can
    tune the thresholds in classify_static_letter / detect_hook_motion /
    detect_zigzag_motion against your own camera and hand instead of
    guessing. Hold a shape or run a gesture and watch which numbers are
    near a threshold boundary - that's the one to nudge."""
    idx_mid_gap = gap(f, "index_tip", "middle_tip")
    thumb_idx_gap = gap(f, "thumb_tip", "index_tip")
    thumb_mid_gap = gap(f, "thumb_tip", "middle_tip")
    d_idx_mcp = dist(f["thumb_tip"], f["index_mcp"]) / f["hand_size"]
    d_mid_mcp = dist(f["thumb_tip"], f["middle_mcp"]) / f["hand_size"]
    d_ring_mcp = dist(f["thumb_tip"], f["ring_mcp"]) / f["hand_size"]
    d_pinky_mcp = dist(f["thumb_tip"], f["pinky_mcp"]) / f["hand_size"]
    order_at_base = f["index_mcp"].x - f["middle_mcp"].x
    order_at_tip = f["index_tip"].x - f["middle_tip"].x


    lines = [
        f"FPS {current_fps:4.1f}   camera: {camera.kind}   tft: {'on' if tft else 'off'}",
        f"idx {f['idx']:>7} {f['idx_angle']:5.0f}   mid {f['mid']:>7} {f['mid_angle']:5.0f}",
        f"ring{f['ring']:>7} {f['ring_angle']:5.0f}   pky {f['pinky']:>7} {f['pinky_angle']:5.0f}",
        f"thumb_angle {f['thumb_angle']:5.0f}  ext={f['thumb_extended']}  front={f['thumb_in_front']}",
        f"idx_mid_gap {idx_mid_gap:.2f} (branch={_last_finger_gap_branch})  thumb_idx {thumb_idx_gap:.2f}  thumb_mid {thumb_mid_gap:.2f}",
        f"thumb->midMCP (K/P check) {dist(f['thumb_tip'], f['middle_mcp']) / f['hand_size']:.2f}  (threshold 0.50)",
        f"thumb-> idxMCP {d_idx_mcp:.2f} midMCP {d_mid_mcp:.2f} ringMCP {d_ring_mcp:.2f} pkyMCP {d_pinky_mcp:.2f}",
        f"fist family: thumb_angle {f['thumb_angle']:.0f} (M/N need >110, A needs >140)  front={f['thumb_in_front']}",
        f"R cross: base {order_at_base:+.3f}  tip {order_at_tip:+.3f}  crossed={order_at_base * order_at_tip < 0}",
        f"orientation {f['orientation']}",
        f"J: shape={is_j_candidate_shape(f)} "
        f"len={j_tracker.path_length():.2f} dur={j_tracker.duration():.2f} pts={len(j_tracker.buffer)}",
        f"J turns: {_j_turn_debug_str()}",
        f"Z: shape={is_z_candidate_shape(f)} "
        f"len={z_tracker.path_length():.2f} dur={z_tracker.duration():.2f} pts={len(z_tracker.buffer)}",
    ]
    y = 160  # below the "Letter / Last confirmed / Word" lines
    for line in lines:
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        y += 22




def draw_test_overlay(image):
    """Panel in the top-right showing the current test prompt (or the
    final summary once the alphabet's been run through), plus running
    accuracy, drawn over a translucent box so it stays readable on top
    of the camera feed. Positioned relative to the frame size."""
    x0, y0, x1, y1 = FRAME_W - 340, 10, FRAME_W - 10, 190
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)


    total_attempts = sum(r["attempts"] for r in test_results.values())
    total_correct = sum(r["correct"] for r in test_results.values())
    acc_str = f"{100 * total_correct / total_attempts:.0f}%" if total_attempts else "--"


    lines = []
    if test_complete:
        lines.append(("TEST COMPLETE", (0, 255, 0)))
        lines.append((f"Accuracy: {acc_str} ({total_correct}/{total_attempts})", (0, 230, 255)))
        saved_name = os.path.basename(test_last_saved_path) if test_last_saved_path else "-"
        lines.append((f"Saved: {saved_name[:26]}", (0, 230, 255)))
        lines.append(("t = restart   s = re-save", (180, 180, 180)))
    else:
        target = current_test_target()
        lines.append((f"SIGN: {target}   ({test_index + 1}/{len(test_order)})", (0, 230, 255)))
        lines.append((f"Running: {acc_str} ({total_correct}/{total_attempts})", (0, 230, 255)))
        if test_last_feedback and time.monotonic() < test_last_feedback_until:
            feedback_color = (0, 255, 0) if test_last_feedback == "CORRECT" else (0, 100, 255)
            lines.append((test_last_feedback, feedback_color))
        lines.append(("SPACE=log  n=skip  t=stop", (180, 180, 180)))


    y = y0 + 30
    for text, color in lines:
        cv2.putText(image, text, (x0 + 12, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        y += 30




last_timestamp_ms = 0  # detect_for_video needs strictly increasing timestamps
last_frame_time = time.monotonic()

if tft:
    tft.update(None, confirmed_word)  # show the "?" idle screen right away

while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        break


    if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    frame = cv2.flip(frame, 1)


    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = frame.copy()  # BGR copy we draw overlays on


    # Monotonic clock, not time.time(): a Pi without a battery-backed clock
    # can have its wall time jump (e.g. when NTP syncs after boot), which
    # would corrupt the J/Z motion timing and break MediaPipe's strictly
    # increasing timestamps.
    now = time.monotonic()
    timestamp_ms = int(now * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms


    dt = now - last_frame_time
    last_frame_time = now
    if dt > 0:
        inst_fps = 1.0 / dt
        current_fps = inst_fps if current_fps == 0 else 0.9 * current_fps + 0.1 * inst_fps


    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    active_hand = result.hand_landmarks[0] if result.hand_landmarks else None


    current_letter = None
    if active_hand:
        draw_hand(image, active_hand)
        features = extract_features(active_hand)
        current_letter = classify_letter(features, now)
        if debug_mode:
            draw_debug_overlay(image, features)
    else:
        # No hand in frame: motion traces so far are meaningless, drop them.
        j_tracker.clear()
        z_tracker.clear()
        j_last_good_time = None
        z_last_good_time = None


    if current_letter in ("J", "Z"):
        # Pin this on screen for a while instead of the single frame it
        # would otherwise get, and don't let the static reads during the
        # cooldown right after (from your hand's resting position) pollute
        # the smoothing buffer.
        last_motion_display = current_letter
        last_motion_display_until = now + MOTION_DISPLAY_SECONDS
        recent_letters.clear()
    elif now >= last_motion_display_until:
        recent_letters.append(current_letter)
    # else: still inside the J/Z sticky window - ignore this frame's
    # static read entirely so it can't flicker back in early.


    if now < last_motion_display_until:
        shown_letter = last_motion_display
    else:
        shown_letter = stable_letter()


    # -- UI -------------------------------------------------------------
    if shown_letter:
        cv2.putText(image, f"Letter: {shown_letter}", (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 255, 0), 3)
    else:
        cv2.putText(image, "Letter: ...", (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1.1, (0, 165, 255), 2)


    # Last confirmed letter + the word built up so far. These stay on
    # screen regardless of what's currently in frame, since they reflect
    # what you've locked in, not what's live.
    confirmed_display = last_confirmed_letter if last_confirmed_letter else "-"
    cv2.putText(image, f"Last confirmed: {confirmed_display}", (10, 90),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 200, 0), 2)
    cv2.putText(image, f"Word: {confirmed_word}", (10, 125),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 200, 0), 2)


    if test_mode:
        draw_test_overlay(image)
        cv2.putText(image, "t = stop test   q = quit", (10, FRAME_H - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (180, 180, 180), 1)
    else:
        cv2.putText(image, "SPACE confirm | BKSP undo | d debug | t test | q quit",
                    (10, FRAME_H - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (180, 180, 180), 1)


    cv2.imshow("ASL Fingerspelling Reader", image)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('d'):
        debug_mode = not debug_mode


    if key == ord('t'):
        if test_mode:
            # Stopping mid-test: save whatever was collected, if anything.
            if sum(r["attempts"] for r in test_results.values()) > 0:
                test_last_saved_path = save_test_results()
            test_mode = False
        else:
            reset_test()
            test_mode = True


    if key == ord('s') and test_mode:
        saved = save_test_results()
        if saved:
            test_last_saved_path = saved


    if key == ord('n') and test_mode and not test_complete:
        # Skip the current prompt without logging it as an attempt.
        test_index += 1
        test_last_feedback = None
        if test_index >= len(test_order):
            test_complete = True
            test_last_saved_path = save_test_results()


    if key == 32:  # SPACE
        if test_mode:
            if not test_complete:
                target = current_test_target()
                if target is not None:
                    log_test_attempt(target, shown_letter)
                    _, _, read_as, correct = test_log[-1]
                    test_last_feedback = "CORRECT" if correct else f"Read as {read_as or '(nothing)'}"
                    test_last_feedback_until = time.monotonic() + 1.2
                    if test_index >= len(test_order):
                        test_complete = True
                        test_last_saved_path = save_test_results()
        else:
            # Normal mode: confirm whatever letter is currently shown
            if shown_letter:
                last_confirmed_letter = shown_letter
                confirmed_word += shown_letter


    if key in (8, 127) and not test_mode:  # BACKSPACE (varies by platform): undo last confirm
        confirmed_word = confirmed_word[:-1]
        last_confirmed_letter = confirmed_word[-1] if confirmed_word else None


    # Push the newest state to the TFT (non-blocking; drawn on its own thread).
    # Done after key handling so a just-confirmed letter shows up immediately.
    if tft:
        tft.update(shown_letter, confirmed_word)


if tft:
    tft.close()
camera.release()
landmarker.close()
cv2.destroyAllWindows()