"""Draw the app icon (icon.png, 1024 px): a speech bubble with sound bars. Needs Pillow.

uv run --no-project --with pillow python tools/gui/packaging/make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

S = 1024
OUT = Path(__file__).resolve().parents[1] / "src" / "gui" / "icon.png"

top, bottom = (79, 70, 229), (13, 148, 136)  # indigo -> teal
grad = Image.new("RGB", (S, S))
px = grad.load()
for y in range(S):
    t = y / (S - 1)
    c = tuple(round(a + (b - a) * t) for a, b in zip(top, bottom, strict=True))
    for x in range(S):
        px[x, y] = c

mask = Image.new("L", (S, S), 0)
m = 100  # macOS-style margin around the rounded square
ImageDraw.Draw(mask).rounded_rectangle((m, m, S - m, S - m), radius=185, fill=255)
icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
icon.paste(grad, (0, 0), mask)

d = ImageDraw.Draw(icon)
white = (255, 255, 255, 255)
d.rounded_rectangle((250, 270, 774, 650), radius=110, fill=white)
d.polygon([(340, 620), (300, 780), (470, 640)], fill=white)
bars = [90, 170, 250, 150, 210, 110]
w, gap = 34, 32
x0 = S // 2 - (len(bars) * w + (len(bars) - 1) * gap) // 2
for i, h in enumerate(bars):
    x = x0 + i * (w + gap)
    d.rounded_rectangle((x, 460 - h // 2, x + w, 460 + h // 2), radius=w // 2, fill=(79, 70, 229, 255))
icon.save(OUT)
print(OUT)
