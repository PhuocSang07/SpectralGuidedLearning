"""Figure 4: answer gain and weight of every step of one training trace, as a draw.io diagram.

Run from SpectralGuidedLearning/ once the answer-gain arm has scored the 1.5B student:

    python paper/make_example_steps.py

It reads the segmented traces and the per-step gains written by sgl.signals.answer_gain
(configs/sgl/methods/iwc-gain.yaml), picks the trace whose problem contains --match, recomputes
the weights with the allocation used in training (sgl.allocation.iwc) and writes
paper/figures/example_steps.drawio, which stays editable in draw.io. When draw.io desktop is on
PATH it also exports paper/figures/example_steps.pdf; otherwise export it from draw.io with
"File > Export as > PDF" (crop) or

    drawio -x -f pdf --crop -o paper/figures/example_steps.pdf paper/figures/example_steps.drawio

No GPU or model is needed. Step texts are abridged by hand for the default trace; any other
trace falls back to truncated step texts.
"""

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sgl.allocation.iwc import stable_iwc_step_weights  # noqa: E402
from sgl.segment.sentence import split_into_step_texts  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2, as in make_figures.py
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#b9b8b3"
PREFIX_FILL, CONT_FILL = "#ecebe7", "#dbe8f8"
FONT = "Times New Roman"
PREFIX_END = "<End_of_Prefix>"

# Inline math for the hand-abridged labels (draw.io HTML labels).
P, T, R, A = "<i>P</i>", "<i>T</i>", "<i>R</i>", "<i>A</i>"
PW, T1, T2 = "<i>P</i><sub>w</sub>", "<i>T</i><sub>1</sub>", "<i>T</i><sub>2</sub>"
RHO, RHO1, RHO2 = "<i>ρ</i>", "<i>ρ</i><sub>1</sub>", "<i>ρ</i><sub>2</sub>"
POWER_LAW = f"{PW} = ½{RHO}{A}<i>v</i><sup>3</sup>"
MARKER = f'<font color="{MUTED}" face="Courier New" style="font-size:9px">&lt;End_of_Prefix&gt;</font>'

# Abridged steps of the wind-power trace (P-ALIGN row 118), in the order the sentence
# segmenter cuts them; used only when the trace matches.
WIND_POWER_STEPS = [
    f"Alright, so I need to find {PW} in terms of {P} and {T2}/{T1}.",
    "Hmm, okay.",
    "Let me start by recalling some basics about wind power …",
    f"Wind power is … {POWER_LAW}, where {RHO} is the air density …",
    "But how does temperature come into play here?",
    "Temperature affects air density.",
    f"The ideal gas law tells us that {RHO} = {P}/({R}{T}) …",
    f"So if … {T1} to {T2}, … {RHO1} = {P}/({R}{T1}) to {RHO2} = {P}/({R}{T2}) …",
    f"Wait, … maybe {P} here is the pressure?",
    f"Or is {P} the power?",
    "That needs clarification.",
    f"But actually, … express {PW} in terms of {P} and {T2}/{T1}.",
    f"… {RHO} depends on temperature.{MARKER} Step 1: {POWER_LAW} …",
    f"So, the wind power depends on air density … Step 2: {RHO} = {P}/({R}{T}) …",
    f"So, if … {T1} to {T2} … {RHO2}/{RHO1} = {T1}/{T2} … Step 3: …",
    f"So, if {RHO} ∝ 1/{T} … Final Answer: {PW} = {P}·{T1}/{T2}",
]
WIND_POWER_FIRST = "Alright, so I need to find the wind power"

# Layout in draw.io units (px). LaTeX scales the exported figure to \textwidth (6.3 in), so
# about 760 px wide gives 0.6x and 11 px text prints at roughly 6.6 pt.
ROW, PITCH, TOP = 22, 26, 30
BRACKET_X, BRACKET_W = 0, 20
CARD_X, CARD_W = 26, 440
GAIN_X, GAIN_W = 486, 150
WEIGHT_X, WEIGHT_W = 652, 108


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/r1-qwen-1.5b-palign",
                        help="holds train-segmented.jsonl and signals/step_answer_gain.json")
    parser.add_argument("--match", default="Find the wind power", help="substring of the problem")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--interpolation", type=float, default=0.5, help="lambda")
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--output", default="paper/figures/example_steps.drawio")
    return parser


def load_trace(data_dir: Path, match: str) -> tuple[dict, list[float]]:
    gains = json.loads((data_dir / "signals/step_answer_gain.json").read_text())
    for line in open(data_dir / "train-segmented.jsonl"):
        record = json.loads(line)
        if match in record["prompt"]:
            return record, gains[str(record["id"])]
    raise SystemExit(f"no trace whose problem contains {match!r} in {data_dir}")


def abridge(text: str, width: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text if len(text) <= width else text[: width - 2].rstrip() + " …"
    return html.escape(text).replace(html.escape(PREFIX_END), MARKER)


def mix(color: str, share: float) -> str:
    """`color` blended with white: share 0 is white, 1 is the color itself."""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 + (c - 255) * share):02x}" for c in rgb)


def fmt(style: dict) -> str:
    return "".join(f"{key}={value};" for key, value in style.items())


class Diagram:
    """Collects mxGraph cells; every shape is a plain vertex so the file opens cleanly in draw.io."""

    def __init__(self) -> None:
        self.cells: list[str] = []

    def box(self, x: float, y: float, w: float, h: float, value: str = "", **style: object) -> None:
        base = {"html": 1, "whiteSpace": "wrap", "fontFamily": FONT, "fontColor": INK, "strokeColor": "none"}
        self.cells.append(
            f'<mxCell id="c{len(self.cells)}" value={quoteattr(value)} style={quoteattr(fmt(base | style))} '
            f'vertex="1" parent="1"><mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'as="geometry"/></mxCell>')

    def text(self, x: float, y: float, w: float, h: float, value: str, size: float = 11, **style: object) -> None:
        self.box(x, y, w, h, value, fillColor="none", fontSize=size, spacing=0, **style)

    def line(self, x1: float, y1: float, x2: float, y2: float, **style: object) -> None:
        base = {"endArrow": "none", "html": 1, "strokeColor": MUTED, "strokeWidth": 1}
        self.cells.append(
            f'<mxCell id="c{len(self.cells)}" style={quoteattr(fmt(base | style))} edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1:.1f}" y="{y1:.1f}" as="sourcePoint"/>'
            f'<mxPoint x="{x2:.1f}" y="{y2:.1f}" as="targetPoint"/></mxGeometry></mxCell>')

    def xml(self) -> str:
        cells = "\n".join(self.cells)
        return ('<mxfile host="make_example_steps.py"><diagram name="Figure 4" id="example-steps">'
                '<mxGraphModel grid="0" page="0" math="0" shadow="0"><root>'
                f'<mxCell id="0"/><mxCell id="1" parent="0"/>\n{cells}\n</root></mxGraphModel></diagram></mxfile>\n')


def draw(labels: list[str], lengths: list[int], gains: list[float], weights: list[float],
         prefix_step: int | None, interpolation: float) -> Diagram:
    d = Diagram()
    n = len(labels)

    def row_y(k: int) -> float:
        return TOP + k * PITCH

    bottom = row_y(n - 1) + ROW

    d.text(CARD_X, 6, 200, 16, "<b>Step</b> (abridged)", align="left")
    d.text(CARD_X + CARD_W - 60, 6, 56, 16, "tokens", size=10, align="right", fontColor=MUTED)
    d.text(GAIN_X, 6, GAIN_W, 16, "<b>Answer gain</b> <i>u<sub>k</sub></i> (nats)", align="center")
    d.text(WEIGHT_X, 6, WEIGHT_W, 16, f"<b>Weight</b> <i>w<sub>k</sub></i> (<i>λ</i> = {interpolation:g})",
           align="center")

    # Who wrote which steps. The teacher prefix ends inside `prefix_step`, so the split (and the
    # dashed line) closes that step's row.
    split = row_y(prefix_step) + ROW + (PITCH - ROW) / 2 if prefix_step is not None else bottom + 2
    d.box(BRACKET_X, TOP - 2, BRACKET_W, min(split, bottom + 2) - TOP + 1, "DeepSeek-R1 prefix", horizontal=0,
          fillColor=PREFIX_FILL, fontSize=10, fontColor=MUTED, rounded=1, arcSize=30)
    if split < bottom:
        d.box(BRACKET_X, split + 1, BRACKET_W, bottom - split + 1, "Qwen2.5-7B", horizontal=0, fillColor=CONT_FILL,
              fontSize=10, fontColor=MUTED, rounded=1, arcSize=30)

    # Gain axis around 0 and weight axis around the uniform weight 1, each padded for value labels.
    low, high = min(0.0, min(gains)), max(0.0, max(gains))
    pad = 0.32 * (high - low or 1.0)
    gain_scale = GAIN_W / (high - low + 2 * pad)
    gain_zero = GAIN_X + (pad - low) * gain_scale
    reach = max(abs(w - 1.0) for w in weights) or 0.1
    w_low, w_high = min(1.0, min(weights)), max(1.0, max(weights))
    w_pad = 0.35 * (w_high - w_low or 0.1)
    weight_scale = WEIGHT_W / (w_high - w_low + 2 * w_pad)
    weight_one = WEIGHT_X + (w_pad + 1.0 - w_low) * weight_scale
    for x in (gain_zero, weight_one):
        d.line(x, TOP - 4, x, bottom + 4, strokeColor=RULE)
    d.text(gain_zero - 10, bottom + 5, 20, 12, "0", size=9, fontColor=MUTED, align="center")
    d.text(weight_one - 30, bottom + 5, 60, 12, "1 (uniform)", size=9, fontColor=MUTED, align="center")

    for k in range(n):
        y, w, g = row_y(k), weights[k], gains[k]
        shade = mix(BLUE if w >= 1 else ORANGE, 0.06 + 0.38 * min(1.0, abs(w - 1.0) / reach))
        d.box(CARD_X, y, CARD_W, ROW, "", fillColor=shade, rounded=1, arcSize=18)
        d.text(CARD_X + 5, y, 24, ROW, f"<i>s</i><sub>{k + 1}</sub>", align="left", verticalAlign="middle")
        d.text(CARD_X + 30, y, CARD_W - 72, ROW, labels[k], align="left", verticalAlign="middle")
        d.text(CARD_X + CARD_W - 40, y, 34, ROW, str(lengths[k]), size=10, fontColor=MUTED, align="right",
               verticalAlign="middle")

        for value, origin, scale, label in ((g, gain_zero, gain_scale, f"{g:+.2f}".replace("-", "−")),
                                            (w, weight_one, weight_scale, f"{w:.2f}")):
            delta = value if origin == gain_zero else value - 1.0
            bar = abs(delta) * scale
            up = delta >= 0
            d.box(origin if up else origin - bar, y + 5, max(bar, 0.8), ROW - 10, fillColor=BLUE if up else ORANGE)
            d.text(origin + bar + 3 if up else origin - bar - 39, y, 36, ROW, label, size=9, fontColor=MUTED,
                   align="left" if up else "right", verticalAlign="middle")

    if split < bottom:
        d.line(BRACKET_X, split, WEIGHT_X + WEIGHT_W, split, dashed=1, dashPattern="5 3")

    # Legend: the card shade encodes the weight, like the bars.
    y = bottom + 22
    d.text(CARD_X, y, 90, 14, "card shade = <i>w<sub>k</sub></i>:", size=9, fontColor=MUTED, align="left")
    for i, (color, share, text) in enumerate([(ORANGE, 0.44, "down-weighted"), (BLUE, 0.06, "≈ uniform"),
                                              (BLUE, 0.44, "up-weighted")]):
        d.box(CARD_X + 92 + i * 100, y + 1, 14, 12, fillColor=mix(color, share), rounded=1, arcSize=25)
        d.text(CARD_X + 110 + i * 100, y, 80, 14, text, size=9, fontColor=MUTED, align="left")
    d.line(CARD_X + 395, y + 7, CARD_X + 420, y + 7, dashed=1, dashPattern="5 3")
    d.text(CARD_X + 425, y, 240, 14, "teacher prefix ends inside the step above", size=9, fontColor=MUTED,
           align="left")
    return d


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    record, gains = load_trace(Path(args.data_dir), args.match)
    texts = split_into_step_texts(record["response"])
    lengths = [step["token_end"] - step["token_start"] for step in record["steps"]]
    if not len(texts) == len(lengths) == len(gains):
        raise SystemExit(f"steps={len(lengths)}, step texts={len(texts)}, gains={len(gains)}")
    weights = stable_iwc_step_weights(gains, lengths, args.temperature, args.interpolation, args.clip)

    labels = (WIND_POWER_STEPS if len(texts) == len(WIND_POWER_STEPS) and WIND_POWER_FIRST in texts[0]
              else [abridge(text) for text in texts])
    prefix_step = next((k for k, text in enumerate(texts) if PREFIX_END in text), None)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(draw(labels, lengths, gains, weights, prefix_step, args.interpolation).xml())
    top = max(range(len(gains)), key=gains.__getitem__)
    lowest = sorted(range(len(weights)), key=weights.__getitem__)[:2]
    print(f"wrote {output}: {len(gains)} steps, largest gain s{top + 1} ({gains[top]:+.2f}, w={weights[top]:.2f}), "
          f"lowest weights {', '.join(f's{k + 1} ({weights[k]:.2f})' for k in lowest)}, token-weighted mean "
          f"{sum(w * n for w, n in zip(weights, lengths)) / sum(lengths):.3f}")

    drawio = shutil.which("drawio") or shutil.which("draw.io")
    if drawio:
        pdf = output.with_suffix(".pdf")
        subprocess.run([drawio, "-x", "-f", "pdf", "--crop", "-o", str(pdf), str(output)], check=True)
        print(f"exported {pdf}")


if __name__ == "__main__":
    main()
