"""
Runs ON THE RASPBERRY PI. Sends a few lines to the Pico over USB serial and
checks that it echoes them back.

Setup (inside your venv):
    pip install pyserial
Run (close Thonny first - only one program can use the port at a time):
    python pi_echo_test.py                 # uses /dev/ttyACM0
    python pi_echo_test.py /dev/ttyACM1    # if the port has another name

If you get "permission denied", run once:
    sudo usermod -a -G dialout $USER      # then log out and back in
"""

import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"


def read_reply(ser, wait=2.0):
    """Return the first line starting with 'echo:' (skips other output such
    as the startup message), or None if nothing arrives in time."""
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        line = ser.readline().decode(errors="replace").strip()
        if line.startswith("echo:"):
            return line
    return None


def main():
    print(f"Opening {PORT} ...")
    try:
        ser = serial.Serial(PORT, 115200, timeout=0.5)
    except serial.SerialException as e:
        sys.exit(f"Could not open {PORT}: {e}\n"
                 "Check 'ls /dev/ttyACM*', that Thonny is closed, and the "
                 "dialout group note at the top of this file.")

    with ser:
        time.sleep(1.0)            # let the Pico settle after the port opens
        ser.reset_input_buffer()

        good = 0
        for i in range(5):
            message = f"ping {i}"
            ser.write((message + "\n").encode())
            reply = read_reply(ser)
            print(f"sent: {message:8}  got: {reply or '(no reply)'}")
            if reply == f"echo: {message}":
                good += 1
            time.sleep(0.2)

    print(f"\n{good}/5 replies correct.")
    if good == 5:
        print("The Pico link works.")
    else:
        print("Some replies were missing. Check that main.py is saved on the "
              "Pico (it should print 'pico ready' in Thonny) and try again.")


if __name__ == "__main__":
    main()