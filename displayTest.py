import time, board, digitalio
from adafruit_rgb_display import ili9341

# Step 1: pulse RST by hand. On most modules the screen visibly changes
# (blanks or flickers) while RST is held low. This proves the RST wire works.
rst = digitalio.DigitalInOut(board.D24)
rst.switch_to_output(value=True)
print("Holding RST low for 2 seconds - watch the screen")
rst.value = False
time.sleep(2)
rst.value = True
time.sleep(0.5)
rst.deinit()

# Step 2: init the display at a very slow SPI speed.
# Use D5 here ONLY if your CS wire is on pin 29 (GPIO5).
# If CS is still on pin 24 (CE0), use cs=None instead.
cs = None
dc = digitalio.DigitalInOut(board.D25)
rst = digitalio.DigitalInOut(board.D24)

disp = ili9341.ILI9341(board.SPI(), cs=cs, dc=dc, rst=rst,
                       baudrate=500_000, rotation=90)
for name, color in (("red", 0xF800), ("green", 0x07E0), ("blue", 0x001F)):
    print("filling", name)
    disp.fill(color)
    time.sleep(1)
print("done")