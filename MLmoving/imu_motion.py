"""
BNO055 motion monitor for the ASL translator (I2C, default address 0x28).

Built from imu_motion_test_bno055.py: same register setup and the same
"motion" rule (linear acceleration OR rotation rate above a threshold), but
it runs on a background thread and just remembers WHEN the IMU last moved,
so the translator can ask "how long has it been still?".

    imu = IMUMotionMonitor(addr=0x28)
    imu.seconds_since_motion()   # 0.0 right after movement, grows when still
    imu.healthy()                # False if the IMU stopped answering
    imu.poke()                   # count it as motion (e.g. a hand was seen)
    imu.close()

Needs:  pip install smbus2     and I2C enabled (sudo raspi-config nonint do_i2c 0)
"""

import math
import threading
import time

from smbus2 import SMBus

# BNO055 registers
CHIP_ID = 0x00          # should read 0xA0
GYRO_DATA = 0x14        # 6 bytes, 1 LSB = 1/16 deg/s
LIA_DATA = 0x28         # 6 bytes linear accel (gravity removed), 1 LSB = 0.01 m/s^2
UNIT_SEL = 0x3B
OPR_MODE = 0x3D
PWR_MODE = 0x3E
SYS_TRIGGER = 0x3F

MODE_CONFIG = 0x00
MODE_IMU = 0x08         # accel + gyro fusion (no magnetometer)


def to_signed(lsb, msb):
    v = (msb << 8) | lsb
    return v - 65536 if v > 32767 else v


def read_block(bus, addr, reg, n, retries=5):
    """Read with retries; the BNO055 occasionally NACKs on the Pi."""
    for attempt in range(retries):
        try:
            return bus.read_i2c_block_data(addr, reg, n)
        except OSError:
            time.sleep(0.01)
    raise OSError(f"I2C read failed after {retries} retries (reg 0x{reg:02X})")


def write_byte(bus, addr, reg, val, retries=5):
    for attempt in range(retries):
        try:
            bus.write_byte_data(addr, reg, val)
            return
        except OSError:
            time.sleep(0.01)
    raise OSError(f"I2C write failed after {retries} retries (reg 0x{reg:02X})")


def init_bno055(bus, addr):
    # Wait for chip to answer, then verify ID
    for _ in range(20):
        try:
            if read_block(bus, addr, CHIP_ID, 1)[0] == 0xA0:
                break
        except OSError:
            pass
        time.sleep(0.1)
    else:
        raise RuntimeError("BNO055 not responding (CHIP_ID != 0xA0)")
    print("BNO055 found (CHIP_ID = 0xA0)")

    write_byte(bus, addr, OPR_MODE, MODE_CONFIG)
    time.sleep(0.03)
    write_byte(bus, addr, SYS_TRIGGER, 0x20)   # reset
    time.sleep(0.7)
    for _ in range(20):                         # wait for it to come back
        try:
            if read_block(bus, addr, CHIP_ID, 1)[0] == 0xA0:
                break
        except OSError:
            pass
        time.sleep(0.1)
    time.sleep(0.05)

    write_byte(bus, addr, PWR_MODE, 0x00)       # normal power
    time.sleep(0.01)
    write_byte(bus, addr, SYS_TRIGGER, 0x00)    # use internal oscillator
    write_byte(bus, addr, UNIT_SEL, 0x00)       # m/s^2, deg/s, degrees
    write_byte(bus, addr, OPR_MODE, MODE_IMU)
    time.sleep(0.05)


def read_motion(bus, addr):
    """Return linear accel (m/s^2) and gyro (deg/s) magnitudes."""
    d = read_block(bus, addr, LIA_DATA, 6)
    lin = [to_signed(d[i], d[i + 1]) / 100.0 for i in (0, 2, 4)]
    g = read_block(bus, addr, GYRO_DATA, 6)
    gyr = [to_signed(g[i], g[i + 1]) / 16.0 for i in (0, 2, 4)]
    return (math.sqrt(sum(c * c for c in lin)),
            math.sqrt(sum(c * c for c in gyr)), lin, gyr)


class IMUMotionMonitor:
    """Samples the IMU on a background thread and tracks when it last moved.
    Raises if the IMU can't be found/initialized, so the caller can carry on
    without it."""

    def __init__(self, bus_num=1, addr=0x28, accel_thresh=0.5,
                 gyro_thresh=15.0, rate=50.0):
        self._addr = addr
        self._accel_thresh = accel_thresh
        self._gyro_thresh = gyro_thresh
        self._period = 1.0 / rate

        self._bus = SMBus(bus_num)
        try:
            init_bno055(self._bus, addr)
        except Exception:
            self._bus.close()
            raise

        now = time.monotonic()
        self._last_motion = now      # starts "just moved" so it doesn't sleep at once
        self._last_ok = now          # last successful IMU read
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                a_mag, g_mag, _lin, _gyr = read_motion(self._bus, self._addr)
            except OSError:
                time.sleep(0.1)
                continue
            now = time.monotonic()
            self._last_ok = now
            if a_mag > self._accel_thresh or g_mag > self._gyro_thresh:
                self._last_motion = now
            time.sleep(self._period)

    def seconds_since_motion(self):
        return time.monotonic() - self._last_motion

    def healthy(self, max_age=2.0):
        """False if the IMU hasn't answered recently. Callers should treat
        that as 'don't power anything down', not as 'still'."""
        return (time.monotonic() - self._last_ok) < max_age

    def poke(self):
        """Count this moment as motion (resets the idle timer)."""
        self._last_motion = time.monotonic()

    def close(self):
        self._stop = True
        self._thread.join(timeout=1)
        try:
            self._bus.close()
        except Exception:
            pass