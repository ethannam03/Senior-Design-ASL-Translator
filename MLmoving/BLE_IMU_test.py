"""
Raspberry Pi side of the ESP32 IMU test.

Finds the ESP32 advertising as "ASL-IMU", connects over Bluetooth LE, and
prints each update it sends:  moving,accel,gyro   (e.g. "1,0.73,22.1").
It also keeps the same "how long has it been still?" timer the translator
uses, and shows what the camera WOULD do (off after IDLE_SECONDS of stillness).

Setup (inside your venv):
    pip install bleak
    (Pi Bluetooth must be on:  rfkill list / bluetoothctl show)
Run:
    python ble_imu_listener.py
Stop with Ctrl+C.
"""

import asyncio
import time

from bleak import BleakClient, BleakScanner

DEVICE_NAME = "ASL-IMU"
TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"   # Nordic UART TX (notify)
IDLE_SECONDS = 5.0                                       # same as the translator


class MotionTracker:
    """Turns the ESP32's text lines into 'seconds since motion'."""

    def __init__(self, idle_seconds=IDLE_SECONDS, clock=time.monotonic):
        self.idle_seconds = idle_seconds
        self.clock = clock
        self.last_motion = clock()
        self._buffer = ""

    def seconds_since_motion(self):
        return self.clock() - self.last_motion

    def feed(self, text):
        """Takes raw notification text and returns a list of lines to show.
        A line can arrive split across notifications, so partial text is
        kept until its newline arrives."""
        self._buffer += text
        shown = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 3 and parts[0] in ("0", "1"):
                if parts[0] == "1":
                    self.last_motion = self.clock()
                still = self.seconds_since_motion()
                state = ("camera would be OFF" if still >= self.idle_seconds
                         else "camera would be on")
                shown.append(f"moving={parts[0]}  accel={parts[1]:>6} m/s^2  "
                             f"gyro={parts[2]:>7} deg/s   "
                             f"still {still:4.1f}s -> {state}")
            else:
                shown.append(f"ESP32 message: {line}")
        return shown


async def run():
    print(f"Scanning for '{DEVICE_NAME}' (up to 15 s)...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=15.0)
    if device is None:
        print("Not found. Check that the ESP32 is powered and its sketch is "
              "running (the LED should blink), that nothing else is connected "
              "to it, and that the Pi's Bluetooth is on.")
        return

    print(f"Found {device.address}. Connecting...")
    tracker = MotionTracker()

    def on_notify(_sender, data):
        for line in tracker.feed(data.decode(errors="ignore")):
            print(line)

    async with BleakClient(device) as client:
        await client.start_notify(TX_CHAR_UUID, on_notify)
        print("Connected. Move the IMU - Ctrl+C to stop.\n")
        while client.is_connected:
            await asyncio.sleep(0.5)
        print("Disconnected.")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped.")