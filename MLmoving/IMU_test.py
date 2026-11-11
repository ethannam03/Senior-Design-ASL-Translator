#!/usr/bin/env python3
"""
IMU motion detection test for Raspberry Pi (BNO055 over I2C, address 0x28).

Uses IMU fusion mode, where the sensor itself subtracts gravity and reports
linear acceleration, so no manual baseline calibration is needed.

Run:
    python3 imu_motion_test_bno055.py
    python3 imu_motion_test_bno055.py --accel-thresh 0.3 --gyro-thresh 15
"""

import argparse
import math
import time

from smbus2 import SMBus

# BNO055 registers
CHIP_ID = 0x00          # should read 0xA0
GYRO_DATA = 0x14        # 6 bytes, 1 LSB = 1/16 deg/s
LIA_DATA = 0x28         # 6 bytes linear accel (gravity removed), 1 LSB = 0.01 m/s^2
CALIB_STAT = 0x35
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bus", type=int, default=1)
    p.add_argument("--addr", type=lambda x: int(x, 0), default=0x28)
    p.add_argument("--accel-thresh", type=float, default=0.5,
                   help="linear accel (m/s^2) that counts as motion")
    p.add_argument("--gyro-thresh", type=float, default=15.0,
                   help="rotation rate (deg/s) that counts as motion")
    p.add_argument("--rate", type=float, default=50.0, help="sample rate in Hz")
    p.add_argument("--verbose", action="store_true",
                   help="print raw values every sample")
    args = p.parse_args()

    with SMBus(args.bus) as bus:
        init_bno055(bus, args.addr)
        print("Ready. Move the IMU! (Ctrl+C to quit)\n")

        moving = False
        last_motion = 0.0
        settle_time = 0.5
        period = 1.0 / args.rate

        try:
            while True:
                try:
                    a_mag, g_mag, lin, gyr = read_motion(bus, args.addr)
                except OSError as e:
                    print(f"read error: {e}")
                    time.sleep(0.1)
                    continue

                detected = a_mag > args.accel_thresh or g_mag > args.gyro_thresh
                now = time.time()

                if detected:
                    last_motion = now
                    if not moving:
                        moving = True
                        print(f"[{time.strftime('%H:%M:%S')}] MOTION DETECTED")
                    print(f"  linear accel: {a_mag:5.2f} m/s^2 | "
                          f"gyro: {g_mag:6.1f} deg/s")
                elif moving and now - last_motion > settle_time:
                    moving = False
                    print(f"[{time.strftime('%H:%M:%S')}] Motion stopped\n")

                if args.verbose:
                    print(f"lin=({lin[0]:+.2f},{lin[1]:+.2f},{lin[2]:+.2f}) "
                          f"gyr=({gyr[0]:+.1f},{gyr[1]:+.1f},{gyr[2]:+.1f})")

                time.sleep(period)
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()