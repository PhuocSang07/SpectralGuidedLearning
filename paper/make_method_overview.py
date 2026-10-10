"""Replace the three chip icons of the method overview with robot illustrations.

The overview was exported from HTML (Chromium), so its source is not in the
repo. Instead of rebuilding it, this script draws vector robots over the old
icons of ``figures/src/method_overview_base.pdf`` and writes
``figures/method_overview.pdf``. Coordinates are in 300-dpi pixels of the base
page (4613 x 1862) and are converted to points.
"""
import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

HERE = Path(__file__).parent
BASE = HERE / "figures/src/method_overview_base.pdf"
OUT = HERE / "figures/method_overview.pdf"
W, H = 1107.12, 446.88
PX = 300 / 72  # pixels per point of the reference rendering

INK = (0.19, 0.25, 0.36)
BODY = (0.95, 0.96, 0.98)
EYE = (0.44, 0.83, 0.90)
GOLD = (0.97, 0.78, 0.18)


def pt(x, y):
    return x / PX, H - y / PX


def rgb(c, col, fill=True):
    (c.setFillColorRGB if fill else c.setStrokeColorRGB)(*col)


def head(c, accent, cap=False, bulb=False, snow=False):
    """Robot head with glasses, unit size (head is 2 wide), centred at 0,0."""
    c.setLineWidth(0.09)
    c.setLineJoin(1)
    c.setLineCap(1)
    rgb(c, INK, False)
    # ears
    rgb(c, accent)
    for sx in (-1, 1):
        c.roundRect(sx * 1.0 - 0.1, -0.25, 0.2, 0.5, 0.08, fill=1, stroke=1)
    # antenna
    if not cap:
        c.line(0, 0.8, 0, 1.12)
        rgb(c, GOLD if bulb else accent)
        c.circle(0, 1.2 + (0.1 if bulb else 0), 0.17 if bulb else 0.1, fill=1, stroke=1)
    rgb(c, BODY)
    c.roundRect(-1, -0.78, 2, 1.58, 0.42, fill=1, stroke=1)
    # eyes behind glasses
    rgb(c, EYE)
    for sx in (-1, 1):
        c.circle(sx * 0.43, 0.05, 0.31, fill=1, stroke=0)
    rgb(c, (1, 1, 1))
    for sx in (-1, 1):
        c.circle(sx * 0.35, 0.13, 0.07, fill=1, stroke=0)
    rgb(c, INK, False)
    c.setFillColorRGB(*INK)
    for sx in (-1, 1):
        c.circle(sx * 0.43, 0.05, 0.31, fill=0, stroke=1)
    c.line(-0.12, 0.05, 0.12, 0.05)
    # smile
    c.arc(-0.28, -0.55, 0.28, -0.15, 200, 140)
    if cap:  # mortarboard
        rgb(c, INK)
        c.roundRect(-0.62, 0.72, 1.24, 0.3, 0.06, fill=1, stroke=0)
        p = c.beginPath()
        p.moveTo(-1.3, 0.98)
        p.lineTo(0, 1.38)
        p.lineTo(1.3, 0.98)
        p.lineTo(0, 0.7)
        p.close()
        c.drawPath(p, fill=1, stroke=0)
        rgb(c, GOLD, False)
        c.setLineWidth(0.06)
        c.line(1.1, 1.02, 1.1, 0.5)
        rgb(c, GOLD)
        c.circle(1.1, 0.46, 0.08, fill=1, stroke=0)
    if snow:  # frozen badge
        rgb(c, (0.18, 0.38, 0.65), False)
        c.setLineWidth(0.07)
        for a in (0, 60, 120):
            import math
            r = 0.26
            dx, dy = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
            c.line(0.95 - dx, 0.78 - dy, 0.95 + dx, 0.78 + dy)


def draw_robot(c, cx, cy, s, **kw):
    c.saveState()
    c.translate(cx, cy)
    c.scale(s, s)
    head(c, **kw)
    c.restoreState()


def laptop(c, cx, cy, s, accent):
    c.saveState()
    c.translate(cx, cy)
    c.scale(s, s)
    c.setLineWidth(0.06)
    rgb(c, INK, False)
    rgb(c, (0.98, 0.98, 1.0))
    c.roundRect(-0.8, -0.1, 1.6, 1.0, 0.1, fill=1, stroke=1)
    rgb(c, accent)
    c.circle(0, 0.4, 0.14, fill=1, stroke=0)
    rgb(c, INK)
    c.roundRect(-1.0, -0.28, 2.0, 0.2, 0.08, fill=1, stroke=0)
    c.restoreState()


def cover(c, x0, y0, x1, y1, col):
    (a, b), (d, e) = pt(x0, y0), pt(x1, y1)
    rgb(c, col)
    c.rect(a, e, d - a, b - e, fill=1, stroke=0)


def build():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(W, H))
    white = (1, 1, 1)
    # 1) teacher: cover chip icon in the "Teacher LRM" box
    cover(c, 590, 232, 692, 352, white)
    x, y = pt(628, 298)
    draw_robot(c, x, y, 9.5, accent=(0.45, 0.52, 0.65), cap=True)
    # 2) frozen student q0: cover chip icon, draw student at laptop
    cover(c, 1300, 940, 1690, 1140, white)
    x, y = pt(1490, 1015)
    _, ly = pt(0, 1098)
    laptop(c, x, ly, 10, (0.18, 0.38, 0.65))
    draw_robot(c, x, y, 13, accent=(0.18, 0.38, 0.65), snow=True)
    # 3) trained student p_theta: cover chip icon
    cover(c, 3418, 1672, 3512, 1768, white)
    x, y = pt(3465, 1722)
    draw_robot(c, x, y - 1, 9.4, accent=(0.69, 0.27, 0.22), bulb=True)
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


if __name__ == "__main__":
    base = PdfReader(str(BASE))
    page = base.pages[0]
    page.merge_page(build())
    w = PdfWriter()
    w.add_page(page)
    w.add_metadata({"/Title": "Method overview, numbered steps"})
    with open(OUT, "wb") as f:
        w.write(f)
    print("wrote", OUT)
