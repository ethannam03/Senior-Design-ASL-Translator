"""
USB webcam test for Raspberry Pi.

Scans every /dev/video* node, reports which ones actually deliver frames,
measures the real frame rate of the first working one, saves a snapshot
(webcam_snapshot.jpg next to this script), and opens a live preview.

Run (inside your venv, from the Pi's desktop / screen sharing for the preview):
    python webcam_test.py
Press q in the preview window to quit.
"""

import glob
import os
import re
import time

import cv2

WIDTH, HEIGHT = 640, 480


def device_number(path):
    m = re.search(r"(\d+)$", path)
    return int(m.group(1)) if m else 9999


def try_device(path):
    """Return (capture, frame) if this node delivers real video, else None."""
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame = None
    for _ in range(10):  # first frames are often dark/empty while exposure settles
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    if frame is None:
        cap.release()
        return None
    return cap, frame


devices = sorted(glob.glob("/dev/video*"), key=device_number)
if not devices:
    raise SystemExit("No /dev/video* devices found. Is the webcam plugged in? Try: lsusb")

print("Found device nodes:", ", ".join(devices))
print("Testing each one for real video frames...\n")

working = None
for path in devices:
    result = try_device(path)
    if result is None:
        print(f"  {path}: no video (normal for metadata / codec / ISP nodes)")
        continue
    cap, frame = result
    h, w = frame.shape[:2]
    print(f"  {path}: WORKS  ({w}x{h})")
    if working is None:
        working = (path, cap, frame)
    else:
        cap.release()

if working is None:
    raise SystemExit(
        "\nNo node delivered frames. Check the cable/port, run 'lsusb' and "
        "'dmesg | tail -20', and try a different USB port (blue USB 3 ports "
        "are best for webcams on a Pi 5)."
    )

path, cap, frame = working
print(f"\nUsing {path}. Measuring frame rate over 60 frames...")
start = time.monotonic()
count = 0
while count < 60:
    ok, frame = cap.read()
    if ok:
        count += 1
elapsed = time.monotonic() - start
print(f"Measured about {count / elapsed:.1f} fps")

snap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webcam_snapshot.jpg")
cv2.imwrite(snap_path, frame)
print(f"Saved snapshot: {snap_path}")

index = device_number(path)
print(f"\nTo use this camera in OpenCV, open it with: cv2.VideoCapture({index}, cv2.CAP_V4L2)")

print("Opening live preview (press q to quit)...")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost the camera feed.")
            break
        cv2.imshow("Webcam test (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except cv2.error as e:
    print(f"Could not open a preview window ({e}). The snapshot above still shows the camera works.")

cap.release()
cv2.destroyAllWindows()