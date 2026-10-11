"""Write method_overview.drawio: the method overview figure as an editable draw.io diagram.

Coordinates match method_overview.tex (1107 x 447, origin top left). Formulas are
LaTeX rendered by draw.io's MathJax; icons (Microsoft Fluent Emoji 3D,
MIT licence) are embedded so the file is self-contained.

Export: draw.io -x -f pdf --crop -o ../method_overview.pdf method_overview.drawio
"""
import base64
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).parent
INK, MUTE, NAVY, RED = "#1F2430", "#5B6577", "#3E4C8A", "#C9503E"
cells = []


def _add(style, x, y, w, h, value=""):
    i = f"c{len(cells) + 2}"
    cells.append(
        f'<mxCell id="{i}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>'
    )


def rect(x0, y0, x1, y1, fill, stroke="none", sw=0.8, r=0, extra=""):
    style = f"rounded={1 if r else 0};absoluteArcSize=1;arcSize={2 * r};fillColor={fill};strokeColor={stroke};strokeWidth={sw};{extra}"
    _add(style, x0, y0, x1 - x0, y1 - y0)


def circle(cx, cy, rad, fill, stroke="none", sw=1):
    _add(f"ellipse;fillColor={fill};strokeColor={stroke};strokeWidth={sw};", cx - rad, cy - rad, 2 * rad, 2 * rad)


def text(anchor, x, y, size, html, color=INK, bold=False, w=420, h=20):
    align = {"west": "left", "center": "center"}[anchor]
    x0 = x if anchor == "west" else x - w / 2
    style = (f"text;html=1;align={align};verticalAlign=middle;spacing=0;whiteSpace=nowrap;"
             f"fontFamily=Helvetica;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};"
             "fillColor=none;strokeColor=none;")
    _add(style, x0, y - h / 2, w, h, html)


def icon(name, cx, cy, size):
    data = base64.b64encode((HERE / "icons" / f"{name}.png").read_bytes()).decode()
    _add(f"shape=image;imageAspect=1;aspect=fixed;image=data:image/png,{data};", cx - size / 2, cy - size / 2, size, size)


def line(points, color, width, dashed=False, arrow=False):
    (sx, sy), *mid, (tx, ty) = points
    i = f"c{len(cells) + 2}"
    pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in mid)
    arr = f'<Array as="points">{pts}</Array>' if mid else ""
    style = (f"endArrow={'block;endFill=1;endSize=5' if arrow else 'none'};startArrow=none;html=1;rounded=0;"
             f"strokeColor={color};strokeWidth={width};{'dashed=1;dashPattern=4 3;' if dashed else ''}")
    cells.append(
        f'<mxCell id="{i}" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{sx}" y="{sy}" as="sourcePoint"/><mxPoint x="{tx}" y="{ty}" as="targetPoint"/>'
        f'{arr}</mxGeometry></mxCell>'
    )


def badge(color, cx, cy, n):
    circle(cx, cy, 10, color)
    text("center", cx, cy, 11, n, "#FFFFFF", bold=True, w=20)


def m(tex):  # inline LaTeX for draw.io MathJax
    return rf"\({tex}\)"


# ------------------------------------------------------------------ panels
rect(0, 0, 1107, 447, "#FFFFFF")
for x0, x1, bg, st, hd, title in [
    (2, 256, "#F1F3F6", "#C9CFD8", "#44506A", "A&nbsp;&nbsp;Fixed teacher trace"),
    (287, 766, "#EAF1F9", "#B3C8E0", "#2B5F97", "B&nbsp;&nbsp;Offline answer-gain scoring"),
    (797, 1105, "#FAF1EC", "#E2C4B8", "#A8483A", "C&nbsp;&nbsp;Soft step allocation"),
]:
    rect(x0, 2, x1, 445, bg, st, 0.8, 7)
    rect(x0, 2, x1, 38, hd, "none", 0, 7)
    rect(x0, 24, x1, 38, hd)  # square off the header's bottom corners
    text("west", x0 + 14, 20, 14.2, title, "#FFFFFF", bold=True)
for x in (259, 769):
    line([(x, 201), (x + 27, 201)], RED, 1.4, dashed=True, arrow=True)

# ------------------------------------------------------------------ A
icon("q", 33, 73, 30)

text("west", 52, 73, 12, f"Question {m('x')}")
icon("teacher", 197, 75, 48); icon("cap", 196, 52, 26); icon("scroll", 219, 90, 20)

text("center", 197, 106, 11.2, "Teacher", bold=True, w=100)

line([(114, 73), (166, 73)], "#7C8597", 1, arrow=True)
line([(197, 113), (197, 142)], "#7C8597", 1, arrow=True)
badge("#44506A", 26, 115, "1")
text("west", 42, 115, 12, "Split into steps", bold=True)
text("west", 42, 129, 10.5, "at sentence boundaries", MUTE)
rect(17, 143, 241, 289, "#FFFFFF", "#C9CFD8", 0.8, 7)
text("west", 27, 157, 11.2, m("y=(s_1,s_2,s_3)"))
for y0, fill, i, label in [(171, "#F8E6E3", 1, "explores a wrong branch"),
                           (208, "#EEF0F3", 2, "restates the problem"),
                           (246, "#FBE7DC", 3, "derives the answer")]:
    rect(24, y0, 234, y0 + 31, fill, r=5)
    rect(30, y0 + 5, 52, y0 + 27, NAVY, r=3)
    text("center", 41, y0 + 16, 10.5, m(f"s_{i}"), "#FFFFFF", w=22)
    text("west", 60, y0 + 15, 11.2, label)
rect(17, 298, 241, 344, "#FFFFFF", "#A1A8B5", 0.8, 8)
icon("gold", 36, 321, 22)
text("west", 60, 321, 11.2, "Gold answer " + m("a^{*}") + r" (last <font face='Courier New'>\boxed{}</font>)")
for y, fill, label in [(356, "#E9B8B0", "answer less likely after the step"),
                       (374, "#CFD5DE", "no change"),
                       (392, "#F0A98A", "answer more likely after the step")]:
    rect(19, y, 30, y + 11, fill, r=3)
    text("west", 36, y + 5.5, 10.5, label, MUTE)
text("center", 129, 423, 11.2, "fixed: no step is cut or rewritten", MUTE)

# ------------------------------------------------------------------ B
badge("#2B5F97", 311, 62, "2")
text("west", 327, 62, 12, "Build one probe per prefix", bold=True)
for k, y0 in enumerate((82, 106, 130, 154)):
    text("west", 303, y0 + 10, 11.2, m(f"k={k}"))
    rect(337, y0, 356, y0 + 20, "#CDD5E1", r=3)
    text("center", 346.5, y0 + 10, 11, m("x"), w=19)
    for j in range(1, k + 1):
        xa = 358 + 26 * (j - 1)
        rect(xa, y0, xa + 24, y0 + 20, NAVY, r=3)
        text("center", xa + 12, y0 + 10, 11, m(f"s_{j}"), "#FFFFFF", w=24)
    xp, xa = 358 + 26 * k, 378 + 26.33 * k
    rect(xp, y0, xp + 18, y0 + 20, "#FBEFC9", r=3)
    text("center", xp + 9, y0 + 10, 11, m(r"\pi"), w=18)
    rect(xa, y0 + 1, xa + 22, y0 + 20, "#FFFFFF", INK, 0.8, 3)
    text("center", xa + 11, y0 + 10.5, 11, m("a^{*}"), w=22)
rect(497, 80, 751, 127, "#FFFAEA", "#E6CF86", 0.8, 7)
text("west", 506, 96, 11.2, "probe " + m(r"\pi="))
text("west", 506, 112, 9.8, r'<font face="Courier New">Therefore, the final answer is \boxed{</font>')
rect(497, 135, 751, 174, "#FFFFFF", "#B3C8E0", 0.8, 7)
text("west", 506, 147, 11.2, f"{m('a^{*}')} = gold answer + &ldquo;}}&rdquo;, read by teacher forcing,")
text("west", 506, 162, 11.2, "without sampling")

badge("#2B5F97", 311, 200, "3")
text("west", 327, 200, 12, "Score with the frozen student", bold=True)

icon("student", 352, 247, 62)
icon("snow", 385, 270, 26)
text("center", 352, 293, 12, f"frozen {m('q_0')}", bold=True, w=110)
text("center", 352, 307, 10.5, "before fine-tuning", MUTE, w=110)
text("center", 452, 226, 11.2, m(r"q_0(a^{*}\mid\cdot)"), w=60)
text("center", 501, 226, 11.2, m("b_k"), w=30)
for y, q, b, col in [(250.5, "0.02", "-3.91", INK), (270, "0.01", "-4.61", INK),
                     (289.5, "0.01", "-4.61", INK), (306.5, "0.10", "-2.30", RED)]:
    text("center", 453, y, 11.2, q, col, w=40)
    text("center", 501, y, 11.2, m(b), col, w=40)
text("west", 302, 331, 10.5, "one forward pass per probe; the shared prefix is", MUTE)
text("west", 302, 344, 10.5, "cached", MUTE)

rect(536, 191, 751, 367, "#FFFFFF", "#B3C8E0", 0.8, 7)
badge("#2B5F97", 556, 206, "4")
text("west", 572, 209, 12, f"Gain {m('=')} change in belief", bold=True)
line([(550, 329), (738, 329)], "#C7C9CC", 0.8)
g = [(560, 291.5), (616, 312), (672, 312), (728, 243.5)]
line(g[:3], "#B3B3B3", 1.9)
line(g[2:], RED, 2.6)
for x, y in g:
    circle(x, y, 3.6, INK)
rect(604, 250, 674, 270, RED, r=3)
text("center", 639, 260, 11, m("u_3=+2.30"), "#FFFFFF", bold=True, w=70)
text("center", 588, 276, 10.5, m("-0.69"), MUTE, w=40)
text("center", 644, 299, 10.5, m("0.00"), MUTE, w=40)
for x, lab in [(560, "before"), (616, m("+s_1")), (672, m("+s_2")), (728, m("+s_3"))]:
    text("center", x, 343, 10.5, lab, MUTE if lab == "before" else INK, w=50)
text("center", 414, 394, 12, m(r"b_k=\log q_0(a^{*}\mid x,s_{1:k},\pi)"), w=240)
text("center", 643, 394, 12, m("u_k=b_k-b_{k-1}"), w=160)
text("center", 526, 424, 10.5,
     "offline, once per trace: no sampling, no gradients, no step labels beyond the gold answer", MUTE, w=470)

# ------------------------------------------------------------------ C
badge("#A8483A", 821, 62, "5")
text("west", 837, 62, 12, "Turn gains into weights", bold=True)
text("west", 837, 77, 11.2, m(r"c=2,\ \tau=1,\ \lambda=0.5"), MUTE)
for x, s in [(960, 1), (1006.5, 2), (1053, 3)]:
    text("center", x, 103, 11.2, m(f"s_{s}"), MUTE, w=30)
rows = [
    (118.5, r"gain \(u_k\)", ["-0.69", "0.00", "+2.30"], ["#E3E7EE", "#F1F2F5", RED], [INK, INK, "#FFFFFF"], False),
    (145.5, r"(i) standardize \(z_k\)", ["-0.96", "-0.42", "+1.38"], ["#E3E7EE", "#F1F2F5", "#F6C6B3"], [INK] * 3, False),
    (172.5, r"(ii) clip, exp \(r_k\)", ["0.38", "0.66", "3.97"], ["#FFFFFF"] * 3, [INK] * 3, False),
    (199.5, r"(iii) token mass \(\hat w_k\)", ["0.23", "0.39", "2.38"], ["#FFFFFF"] * 3, [INK] * 3, False),
    (226.5, r"(iv) toward uniform \(w_k\)", ["0.61", "0.70", "1.69"], ["#FBD9C4", "#F4A582", RED], [INK, INK, "#FFFFFF"], True),
]
for y, lab, vals, fills, cols, bold in rows:
    text("west", 812, y, 11.2, lab)
    for x, v, f, c in zip((960, 1006.5, 1053), vals, fills, cols):
        rect(x - 21, y - 10.5, x + 21, y + 10.5, f, "#E2C4B8", 0.6, 4)
        text("center", x, y, 11, m(v) if v[0] in "+-" else v, c, bold=bold or c == "#FFFFFF", w=42)
text("center", 857, 261, 12, m(r"z_k=\dfrac{u_k-\mu_u}{\sigma_u}"), w=120, h=34)
text("center", 1010, 262, 12, m(r"w_k=(1-\lambda)+\lambda\,\hat w_k"), w=180)

badge("#A8483A", 821, 306, "6")
text("west", 837, 307, 12, "Train with weighted SFT", bold=True)
text("west", 812, 326, 10.5, "every token of " + m("s_k") + " is trained with weight " + m("w_k"), MUTE)
for x0, x1, f, lab, c in [(812, 884, "#FBD9C4", "s_1", INK), (884, 963, "#F4A582", "s_2", INK),
                          (963, 1090, RED, "s_3", "#FFFFFF")]:
    rect(x0, 336, x1, 360, f)
    text("center", (x0 + x1) / 2, 348, 11, m(lab), c, w=x1 - x0)
line([(812, 363), (812, 367), (1090, 367), (1090, 363)], "#A5A7AC", 0.7)
text("center", 951, 377, 10.5,
     "same total " + m(r"\sum_k \ell_k w_k=\sum_k \ell_k") + " &nbsp;and&nbsp; " + m(r"w_k\ge 1-\lambda"), MUTE, w=300)

icon("student", 834, 412, 42); icon("train", 852, 426, 22)
text("west", 868, 413, 11.2,
     "<b>Student " + m(r"p_\theta") + "</b>: " + m(r"\mathcal{L}=\dfrac{1}{Z}\sum_k w_k\,\mathrm{NLL}(s_k)"), h=34)

xml = (
    '<mxfile host="Electron"><diagram id="method-overview" name="method overview">'
    '<mxGraphModel dx="1107" dy="447" grid="0" gridSize="10" guides="1" tooltips="1" connect="0" arrows="1" '
    'fold="1" page="1" pageScale="1" pageWidth="1107" pageHeight="447" math="1" shadow="0">'
    '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>"
)
(HERE / "method_overview.drawio").write_text(xml, encoding="utf-8")
print(f"wrote method_overview.drawio with {len(cells)} cells")
