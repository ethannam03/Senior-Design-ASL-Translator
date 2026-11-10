"""
Optional ILI9341 TFT output for the ASL translator (Raspberry Pi 5).

Shows the same information as the translator's "Results" window:
    Static:  A  (95.0%)
    Motion:  J  (80.0%)
    Letter: A   Last: B   [MANUAL]
    Word: HELLO
    (key hints)

Importing this module raises an exception if the display libraries aren't
installed, and TFTDisplay() raises if the display can't be initialized -
the translator catches both and simply runs without a screen.

Needs (inside your venv):
    sudo raspi-config nonint do_spi 0        # then reboot
    sudo apt install -y fonts-dejavu-core python3-pil
    pip install adafruit-circuitpython-rgb-display adafruit-blinka
Wiring: SPI0, CS = CE0 (pin 24), D/C = GPIO25 (pin 22), RST = GPIO24 (pin 18),
and the board's IM1/IM2/IM3 = 3.3V, IM0 = GND (SPI mode).
"""

import threading
import time

import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
from adafruit_rgb_display import ili9341

TFT_CS = None          # None = CE0 (pin 24) driven automatically by the kernel.
                       # Or a Blinka pin name like "D5" if CS is on a plain GPIO.
TFT_DC = "D25"         # GPIO25, pin 22
TFT_RST = "D24"        # GPIO24, pin 18
TFT_ROTATION = 90      # 90 = 320x240 landscape; try 270 if upside down
TFT_BAUDRATE = 8_000_000   # raise toward 24_000_000 if your wiring is clean
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# The confidence percentages change every frame, so without a limit the
# screen would be redrawing constantly. At most one redraw per this many
# seconds (a faster TFT_BAUDRATE makes each redraw quicker, too).
MIN_REDRAW_INTERVAL = 0.2

# Same color scheme as the Results window (RGB here; OpenCV uses BGR).
COLOR_STATIC = (0, 255, 0)
COLOR_MOTION = (255, 165, 0)
COLOR_STATUS = (0, 200, 255)
COLOR_WORD = (255, 255, 255)
COLOR_HINT = (150, 150, 150)

HINT_LINES = (
    "ENTER confirm   ENTER x2 speak   BKSP delete",
    "SPACE space   0 clear   a auto/manual   ESC quit",
)


def _load_font(path, size):
    """TrueType font if available, otherwise PIL's tiny built-in font."""
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


class TFTDisplay:
    """Mirrors the Results window on the TFT.

    A full-screen refresh over SPI takes 100+ ms, which would stall the
    camera loop if done inline. So update() only stores the latest state
    and wakes a worker thread that does the drawing; if several updates
    arrive while it's busy, only the newest is drawn.

    The state is a dict with these keys:
        "static":    (letter or None, confidence 0-1)
        "motion":    (letter or None, confidence 0-1)
        "current":   letter currently shown / ready to confirm, or None
        "confirmed": last confirmed letter, or None
        "mode":      "AUTO" or "MANUAL"
        "word":      the word built so far
    """

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

        self.font_line = _load_font(FONT_BOLD, 20)
        self.font_status = _load_font(FONT_BOLD, 16)
        self.font_word = _load_font(FONT_BOLD, 30)
        self.font_hint = _load_font(FONT_REG, 12)

        self._fill((0, 0, 0))  # also proves SPI works before the thread starts

        self._lock = threading.Lock()
        self._pending = None
        self._wake = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, state):
        """Cheap and non-blocking; safe to call every frame."""
        with self._lock:
            self._pending = state
        self._wake.set()

    def _fill(self, rgb):
        self.disp.image(Image.new("RGB", (self.width, self.height), rgb))

    def _run(self):
        last_state = None
        last_draw = 0.0
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop:
                return

            # Rate limit: wait out the rest of the interval, then take
            # whatever the newest state is by then.
            remaining = MIN_REDRAW_INTERVAL - (time.monotonic() - last_draw)
            if remaining > 0:
                time.sleep(remaining)
            if self._stop:
                return

            with self._lock:
                state = self._pending
            if state is None or state == last_state:
                continue
            try:
                self._draw(state)
                last_state = state
                last_draw = time.monotonic()
            except Exception as e:
                print(f"TFT drawing stopped after an error: {e}")
                return

    def _wrap(self, draw, text, font, max_width):
        """Character-wrap text to fit max_width pixels."""
        lines, current = [], ""
        for ch in text:
            if draw.textlength(current + ch, font=font) <= max_width:
                current += ch
            else:
                lines.append(current)
                current = ch
        lines.append(current)
        return lines

    def _draw(self, s):
        img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        d = ImageDraw.Draw(img)

        static_letter, static_conf = s["static"]
        motion_letter, motion_conf = s["motion"]
        current = s["current"] or "-"
        confirmed = s["confirmed"] or "-"

        static_text = (f"Static:  {static_letter}  ({static_conf:.1%})"
                       if static_letter else "Static:  -")
        motion_text = (f"Motion:  {motion_letter}  ({motion_conf:.1%})"
                       if motion_letter else "Motion:  -")
        status_text = f"Letter: {current}   Last: {confirmed}   [{s['mode']}]"

        d.text((8, 6), static_text, font=self.font_line, fill=COLOR_STATIC)
        d.text((8, 34), motion_text, font=self.font_line, fill=COLOR_MOTION)
        d.text((8, 64), status_text, font=self.font_status, fill=COLOR_STATUS)

        # Word, wrapped; if it's too long, the newest 3 lines are shown.
        word_lines = self._wrap(d, f"Word: {s['word']}", self.font_word, self.width - 16)
        y = 94
        for line in word_lines[-3:]:
            d.text((8, y), line, font=self.font_word, fill=COLOR_WORD)
            y += 36

        # Key hints along the bottom
        y = self.height - 2 * 16 - 4
        for hint in HINT_LINES:
            d.text((8, y), hint, font=self.font_hint, fill=COLOR_HINT)
            y += 16

        self.disp.image(img)

    def close(self):
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=2)
        try:
            self._fill((0, 0, 0))
        except Exception:
            pass