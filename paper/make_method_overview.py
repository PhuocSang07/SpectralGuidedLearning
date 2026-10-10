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
    rgb(c, (0.98, 0.76, 0.78))
    for sx in (-1, 1):
        c.circle(sx * 0.68, -0.32, 0.1, fill=1, stroke=0)
    rgb(c, INK, False)
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



def body(c, accent, book=False):
    """Torso below the head (head centre at 0,0)."""
    c.setLineWidth(0.09)
    rgb(c, INK, False)
    rgb(c, BODY)
    c.roundRect(-0.72, -1.55, 1.44, 0.8, 0.28, fill=1, stroke=1)
    rgb(c, accent)
    c.circle(0, -1.15, 0.13, fill=1, stroke=0)
    if book:
        rgb(c, (0.80, 0.25, 0.22))
        c.roundRect(-0.95, -1.5, 0.75, 0.55, 0.06, fill=1, stroke=1)
        rgb(c, (1, 1, 1))
        c.setLineWidth(0.05)
        c.line(-0.57, -1.45, -0.57, -0.98)


def bubble(c, x, y, r, tail):
    rgb(c, INK, False)
    c.setLineWidth(0.06)
    rgb(c, (1, 1, 1))
    c.circle(x, y, r, fill=1, stroke=1)
    tx, ty = tail
    for k, rr in ((0.35, 0.13), (0.65, 0.08)):
        c.circle(x + (tx - x) * k, y + (ty - y) * k, rr, fill=1, stroke=1)


def snowflake(c, x, y, r, col):
    import math
    rgb(c, col, False)
    c.setLineWidth(0.08)
    for a in (0, 60, 120):
        dx, dy = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
        c.line(x - dx, y - dy, x + dx, y + dy)


def student_scene(c, cx, cy, s, accent):
    """Student robot at a desk with books, plant, '?' and a frozen thought."""
    c.saveState()
    c.translate(cx, cy)
    c.scale(s, s)
    c.setLineWidth(0.07)
    rgb(c, INK, False)
    # desk
    rgb(c, (0.72, 0.52, 0.34))
    c.roundRect(-3.9, -1.75, 7.8, 0.4, 0.08, fill=1, stroke=1)
    c.rect(-3.5, -2.45, 0.28, 0.7, fill=1, stroke=0)
    c.rect(3.22, -2.45, 0.28, 0.7, fill=1, stroke=0)
    # books
    for i, col in enumerate(((0.35, 0.55, 0.45), (0.85, 0.45, 0.35), (0.35, 0.45, 0.7))):
        rgb(c, col)
        c.roundRect(-3.6 + 0.05 * (i % 2), -1.35 + 0.3 * i, 1.4, 0.3, 0.05, fill=1, stroke=1)
    # plant
    rgb(c, (0.40, 0.68, 0.40))
    for dx, dy, r in ((-0.22, 0.12, 0.28), (0.22, 0.12, 0.28), (0, 0.32, 0.3)):
        c.circle(3.0 + dx, -0.7 + dy, r, fill=1, stroke=1)
    rgb(c, (0.85, 0.5, 0.35))
    p = c.beginPath()
    p.moveTo(2.6, -1.35); p.lineTo(3.4, -1.35); p.lineTo(3.25, -0.92); p.lineTo(2.75, -0.92); p.close()
    c.drawPath(p, fill=1, stroke=1)
    # robot behind the laptop
    c.saveState()
    c.translate(0, 0.25)
    head(c, accent=accent)
    c.restoreState()
    rgb(c, BODY)
    c.roundRect(-0.8, -1.35, 1.6, 0.55, 0.2, fill=1, stroke=1)
    # laptop
    rgb(c, (0.97, 0.98, 1.0))
    c.roundRect(-1.25, -1.5, 2.5, 1.05, 0.12, fill=1, stroke=1)
    rgb(c, accent)
    c.circle(0, -0.97, 0.16, fill=1, stroke=0)
    # thought bubbles: "?" and frozen
    bubble(c, -2.5, 1.95, 0.6, (-1.2, 0.95))
    rgb(c, INK)
    c.setFont("Helvetica-Bold", 1.05 * 1)
    c.drawCentredString(-2.5, 1.62, "?")
    bubble(c, 2.5, 1.95, 0.6, (1.2, 0.95))
    snowflake(c, 2.5, 1.95, 0.36, accent)
    c.restoreState()


def teacher_icon(c, cx, cy, s, accent):
    c.saveState()
    c.translate(cx, cy)
    c.scale(s, s)
    body(c, accent, book=True)
    head(c, accent=accent, cap=True)
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
    cover(c, 585, 226, 695, 360, white)
    x, y = pt(636, 275)
    teacher_icon(c, x, y, 7.6, (0.45, 0.52, 0.65))
    # 2) frozen student q0: student at a desk in the large white box
    cover(c, 1290, 935, 1690, 1135, white)
    x, y = pt(1490, 1035)
    student_scene(c, x, y, 9.0, (0.18, 0.38, 0.65))
    # 3) trained student p_theta
    cover(c, 3418, 1672, 3512, 1768, white)
    x, y = pt(3465, 1712)
    c.saveState()
    c.translate(x, y)
    c.scale(7.4, 7.4)
    body(c, (0.69, 0.27, 0.22))
    head(c, accent=(0.69, 0.27, 0.22), bulb=True)
    c.restoreState()
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
