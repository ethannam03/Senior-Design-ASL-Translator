#!/usr/bin/env python3
"""
Speaker test for a MAX98357A I2S amp on a Raspberry Pi.

Plays: (1) a 440 Hz beep, (2) a rising scale, (3) spoken text via espeak-ng.
Uses only the standard library plus the `aplay` and `espeak-ng` command-line tools.

Setup:
    sudo apt install -y alsa-utils espeak-ng
    # /boot/firmware/config.txt should contain:
    #   dtoverlay=hifiberry-dac
    #   dtoverlay=i2s-mmap
    # then reboot and confirm the card appears with:  aplay -l

Run:
    python3 speaker_test.py
    python3 speaker_test.py --text "Hello from the ASL translator" --volume 0.3
    python3 speaker_test.py --device plughw:0,0
"""

import argparse
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import wave

RATE = 48000


def make_tone_wav(path, freqs, seconds_each=0.4, volume=0.3):
    """Write a mono 16-bit WAV containing each frequency in sequence."""
    amp = int(32767 * volume)
    fade = int(RATE * 0.01)  # 10 ms fade in/out to avoid clicks
    frames = bytearray()
    for f in freqs:
        n = int(RATE * seconds_each)
        for i in range(n):
            env = min(1.0, i / fade, (n - i) / fade)
            sample = int(amp * env * math.sin(2 * math.pi * f * i / RATE))
            frames += struct.pack("<hh", sample, sample)  # left + right
        frames += struct.pack("<hh", 0, 0) * int(RATE * 0.05)  # short gap
    with wave.open(path, "wb") as w:
        w.setnchannels(2)  # I2S DACs require stereo
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(frames))


def find_i2s_device():
    """Look through `aplay -l` for the hifiberry/I2S DAC and return plughw:N,0."""
    try:
        out = subprocess.run(["aplay", "-l"], capture_output=True,
                             text=True).stdout
    except OSError:
        return None
    for line in out.splitlines():
        low = line.lower()
        if not line.startswith("card"):
            continue
        if any(skip in low for skip in ("hdmi", "vc4", "bcm2835", "headphones")):
            continue  # not the I2S amp
        card = line.split()[1].rstrip(":")
        return f"plughw:{card},0"
    return None


def play_wav(path, device):
    cmd = ["aplay", "-q"]
    if device:
        cmd += ["-D", device]
    cmd.append(path)
    subprocess.run(cmd, check=True)


def speak(text, device, volume):
    """Pipe espeak-ng's WAV output into aplay so we can pick the device."""
    amplitude = int(volume * 200)  # espeak-ng amplitude 0-200
    espeak = subprocess.Popen(
        ["espeak-ng", "-a", str(amplitude), "--stdout", text],
        stdout=subprocess.PIPE,
    )
    cmd = ["aplay", "-q"]
    if device:
        cmd += ["-D", device]
    subprocess.run(cmd, stdin=espeak.stdout, check=True)
    espeak.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None,
                   help="ALSA device, e.g. plughw:0,0 (default: system default)")
    p.add_argument("--text", default="Speaker test successful.",
                   help="text to speak")
    p.add_argument("--volume", type=float, default=0.3,
                   help="0.0 to 1.0 (start low!)")
    p.add_argument("--skip-speech", action="store_true")
    args = p.parse_args()

    if not 0.0 < args.volume <= 1.0:
        sys.exit("--volume must be between 0 and 1")
    if shutil.which("aplay") is None:
        sys.exit("aplay not found: sudo apt install alsa-utils")

    if args.device is None:
        args.device = find_i2s_device()
        if args.device:
            print(f"Using detected I2S device: {args.device}")
        else:
            print("No I2S card found in `aplay -l`. Check your config.txt "
                  "overlay and wiring, then reboot. Trying system default...")

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            print("1/3  440 Hz beep...")
            make_tone_wav(f.name, [440], seconds_each=1.0, volume=args.volume)
            play_wav(f.name, args.device)

            print("2/3  Rising scale (C4 to C5)...")
            scale = [262, 294, 330, 349, 392, 440, 494, 523]
            make_tone_wav(f.name, scale, seconds_each=0.3, volume=args.volume)
            play_wav(f.name, args.device)

        if args.skip_speech:
            print("Skipping speech test.")
        elif shutil.which("espeak-ng") is None:
            print("3/3  espeak-ng not installed, skipping: "
                  "sudo apt install espeak-ng")
        else:
            print(f'3/3  Speaking: "{args.text}"')
            speak(args.text, args.device, args.volume)

        print("Done.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Playback failed ({e}). Check `aplay -l` and your "
                 "/boot/firmware/config.txt overlay, or try --device plughw:0,0")


if __name__ == "__main__":
    main()