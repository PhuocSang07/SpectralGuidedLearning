"""Compose the Teacher / Student scenes of the method overview from Fluent Emoji icons (MIT).

Writes figures/src/img/{teacher,student,student_frozen}.svg, which method_overview.html embeds.
"""
import re
from pathlib import Path

IMG = Path(__file__).parent / "figures/src/img"
INK = "#30405c"


def icon(name, x, y, w, h=None, rot=0):
    body = re.search(r"<svg[^>]*>(.*)</svg>", (IMG / "icons" / f"{name}.svg").read_text(), re.S).group(1)
    h = h or w
    t = f' transform="rotate({rot} {x + w / 2} {y + h / 2})"' if rot else ""
    return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="0 0 32 32"{t}>{body}</svg>'


def cloud(cx, cy, rx, ry, tail):
    """Thought cloud (white, outlined) with two trailing dots toward `tail`."""
    tx, ty = tail
    dots = "".join(
        f'<circle cx="{cx + (tx - cx) * k:.1f}" cy="{cy + (ty - cy) * k:.1f}" r="{r}" fill="#fff" stroke="{INK}" stroke-width="2.5"/>'
        for k, r in ((0.62, 6), (0.82, 3.8))
    )
    return (f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#fff" stroke="{INK}" stroke-width="3"/>' + dots)


def svg(w, h, body):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{body}</svg>'


def teacher():
    b = []
    # chalkboard with ABC, apple and test tube
    b.append('<rect x="100" y="8" width="196" height="132" rx="9" fill="#8a5d3b" stroke="#5a3b25" stroke-width="3"/>')
    b.append('<rect x="109" y="17" width="178" height="114" rx="4" fill="#2f5d4f"/>')
    b.append('<path d="M118 123H278" stroke="#cfe0d8" stroke-width="3" opacity=".6"/>')
    b.append(icon("green-apple", 172, 22, 34))
    b.append(icon("test-tube", 232, 22, 34, rot=18))
    b.append('<text x="170" y="96" font-size="50" font-style="italic" fill="#f2f6f3" font-family="Liberation Serif, serif">ABC</text>')
    b.append('<path d="M130 140v60M266 140v60" stroke="#5a3b25" stroke-width="7" stroke-linecap="round"/>')
    # robot body (the emoji is a head only)
    b.append(f'<rect x="50" y="138" width="92" height="90" rx="24" fill="#e9edf6" stroke="{INK}" stroke-width="3.5"/>')
    b.append('<rect x="64" y="160" width="64" height="9" rx="4.5" fill="#8fd3e6"/><rect x="64" y="176" width="64" height="9" rx="4.5" fill="#8fd3e6"/>')
    # pointer arm
    b.append(f'<path d="M136 160Q170 150 188 112" fill="none" stroke="{INK}" stroke-width="12" stroke-linecap="round"/>')
    b.append('<path d="M136 160Q170 150 188 112" fill="none" stroke="#e9edf6" stroke-width="6" stroke-linecap="round"/>')
    b.append('<path d="M186 112L222 68" stroke="#d7b46b" stroke-width="5" stroke-linecap="round"/>')
    b.append(f'<circle cx="188" cy="110" r="9" fill="#f6f8fb" stroke="{INK}" stroke-width="3"/>')
    # head + mortarboard
    b.append(icon("robot", 36, 24, 120))
    b.append(icon("graduation-cap", 46, -4, 100, rot=-4))
    # book held on the left
    b.append(f'<path d="M52 174Q28 190 36 210" fill="none" stroke="{INK}" stroke-width="12" stroke-linecap="round"/><path d="M52 174Q28 190 36 210" fill="none" stroke="#e9edf6" stroke-width="6" stroke-linecap="round"/>')
    b.append(icon("open-book", 4, 184, 62))
    return svg(300, 240, "".join(b))


def student(frozen):
    b = []
    # desk
    b.append(f'<rect x="6" y="196" width="288" height="18" rx="5" fill="#c08a5a" stroke="#6b4b33" stroke-width="3.5"/>')
    b.append('<path d="M26 214v32M274 214v32" stroke="#6b4b33" stroke-width="8" stroke-linecap="round"/>')
    b.append(icon("books", 6, 148, 56))
    b.append(icon("potted-plant", 238, 148, 56))
    # robot head, body, laptop seen from behind (lid with a logo)
    b.append(icon("robot", 90, 40, 120))
    b.append(f'<rect x="112" y="148" width="76" height="50" rx="16" fill="#e9edf6" stroke="{INK}" stroke-width="3.5"/>')
    b.append(f'<rect x="92" y="146" width="116" height="52" rx="8" fill="#dfe4ee" stroke="{INK}" stroke-width="3.5"/>')
    b.append('<circle cx="150" cy="172" r="9" fill="#f6f8fb" stroke="#9aa5ba" stroke-width="2.5"/>')
    if frozen:
        b.append(icon("locked", 134, 154, 32))
    b.append(f'<rect x="84" y="194" width="132" height="6" rx="3" fill="{INK}"/>')
    # thought clouds
    b.append(cloud(52, 44, 40, 30, (96, 88)))
    b.append(icon("red-question-mark", 32, 18, 40))
    b.append(cloud(248, 44, 40, 30, (204, 92)))
    if frozen:
        b.append(icon("snowflake", 224, 18, 48))
    else:
        b.append(icon("light-bulb", 222, 16, 40) + icon("gear", 252, 36, 26))
    return svg(300, 250, "".join(b))


(IMG / "teacher.svg").write_text(teacher())
(IMG / "student.svg").write_text(student(False))
(IMG / "student_frozen.svg").write_text(student(True))
print("wrote scenes")
