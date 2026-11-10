"""
Drop-in replacement for the pyttsx3-based Speaker class in the ASL translator.

Why: pyttsx3 plays through ALSA's *default* device, which on your Pi is an
HDMI port (card 0), not the MAX98357A amp. That's what caused the
"audio open error 524". This version pipes espeak-ng straight into aplay on
the I2S card instead. No pyttsx3 needed.

Environment variables (all optional):
    ASL_AUDIO_DEVICE=plughw:2,0   force a specific ALSA device
    ASL_VOLUME=0.5                0.0 to 1.0 (default 0.5)
    ASL_VOICE=en-us               espeak-ng voice
    ASL_SPEECH_SPEED=150          words per minute (espeak-ng default is 175)

Requires:  sudo apt install -y alsa-utils espeak-ng
"""

import os
import shutil
import subprocess
import threading


def find_i2s_device():
    """Return 'plughw:N,0' for the first non-HDMI, non-onboard sound card."""
    try:
        out = subprocess.run(["aplay", "-l"], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        low = line.lower()
        if not line.startswith("card"):
            continue
        if any(skip in low for skip in ("hdmi", "vc4", "bcm2835", "headphones")):
            continue
        card = line.split()[1].rstrip(":")
        return f"plughw:{card},0"
    return None


class Speaker:
    """Speaks text on a background thread so the camera never freezes.
    A request made while it is still talking is ignored instead of queued."""

    def __init__(self):
        self._busy = threading.Lock()
        self.device = os.environ.get("ASL_AUDIO_DEVICE") or find_i2s_device()
        try:
            vol = float(os.environ.get("ASL_VOLUME", "0.5"))
        except ValueError:
            vol = 0.5
        self.amplitude = int(max(0.0, min(1.0, vol)) * 200)  # espeak: 0-200
        self.voice = os.environ.get("ASL_VOICE", "en-us")
        self.speed = os.environ.get("ASL_SPEECH_SPEED", "150")

        self.ready = True
        for tool in ("espeak-ng", "aplay"):
            if shutil.which(tool) is None:
                print(f"Speech disabled: '{tool}' not found "
                      "(sudo apt install alsa-utils espeak-ng).")
                self.ready = False
        if self.ready:
            if self.device:
                print(f"Speech output: {self.device}")
            else:
                print("No I2S sound card found in `aplay -l`; speech will use "
                      "the system default (probably HDMI). Check "
                      "dtoverlay=hifiberry-dac in /boot/firmware/config.txt.")

    def say(self, text):
        if not self.ready:
            print(f"(speech unavailable) {text}")
            return
        if not self._busy.acquire(blocking=False):
            return  # already speaking
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _run(self, text):
        espeak = None
        try:
            espeak = subprocess.Popen(
                ["espeak-ng", "-v", self.voice, "-s", self.speed,
                 "-a", str(self.amplitude), "--stdout", text],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            cmd = ["aplay", "-q"]
            if self.device:
                cmd += ["-D", self.device]
            result = subprocess.run(cmd, stdin=espeak.stdout,
                                    capture_output=False,
                                    stderr=subprocess.PIPE, text=True)
            espeak.stdout.close()
            espeak.wait()
            if result.returncode != 0:
                print(f"Speech failed: {result.stderr.strip()} "
                      "(check `aplay -l` or set ASL_AUDIO_DEVICE=plughw:N,0)")
        except Exception as e:
            print(f"Speech failed: {e}")
        finally:
            if espeak is not None and espeak.poll() is None:
                espeak.kill()
            self._busy.release()