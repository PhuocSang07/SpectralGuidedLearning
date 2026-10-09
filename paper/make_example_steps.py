"""Figure 4 (figures/example_steps.pdf): answer gain and weight of every step of one training trace.

Run from SpectralGuidedLearning/ once the answer-gain arm has scored the 1.5B student:

    python paper/make_example_steps.py

It reads the segmented traces and the per-step gains written by sgl.signals.answer_gain
(configs/sgl/methods/iwc-gain.yaml), picks the trace whose problem contains --match, and
recomputes the weights with the allocation used in training (sgl.allocation.iwc), so the
figure needs neither a GPU nor the model. Step texts are abridged by hand for the default
trace; any other trace falls back to truncated step texts.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sgl.allocation.iwc import stable_iwc_step_weights  # noqa: E402
from sgl.segment.sentence import split_into_step_texts  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2, as in make_figures.py
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PREFIX_END = "<End_of_Prefix>"

# Abridged steps of the wind-power trace (P-ALIGN row 118), in the order the sentence
# segmenter cuts them; checked against the trace before use.
WIND_POWER_STEPS = [
    r"Alright, so I need to find $P_w$ in terms of $P$ and $T_2/T_1$.",
    r"Hmm, okay.",
    r"Let me start by recalling some basics about wind power ...",
    r"Wind power is ... $P_w=\frac{1}{2}\rho A v^3$, where $\rho$ is the air density ...",
    r"But how does temperature come into play here?",
    r"Temperature affects air density.",
    r"The ideal gas law tells us that $\rho=P/(RT)$ ...",
    r"So if ... $T_1$ to $T_2$, ... $\rho_1=P/(RT_1)$ to $\rho_2=P/(RT_2)$ ...",
    r"Wait, ... maybe $P$ here is the pressure?",
    r"Or is $P$ the power?",
    r"That needs clarification.",
    r"But actually, ... express $P_w$ in terms of $P$ and $T_2/T_1$.",
    r"... $\rho$ depends on temperature. [prefix ends] Step 1: $P_w=\frac{1}{2}\rho A v^3$",
    r"So, $P_w$ depends on $\rho$ and $v^3$. ... Step 2: $\rho=P/(RT)$ ...",
    r"So, if ... $T_1$ to $T_2$ ... $\rho_2/\rho_1=T_1/T_2$ ... Step 3: ...",
    r"So, if $\rho\propto 1/T$ ... Final Answer: $P_w=P\cdot T_1/T_2$",
]
WIND_POWER_FIRST = "Alright, so I need to find the wind power"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/r1-qwen-1.5b-palign",
                        help="holds train-segmented.jsonl and signals/step_answer_gain.json")
    parser.add_argument("--match", default="Find the wind power", help="substring of the problem")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--interpolation", type=float, default=0.5, help="lambda")
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--output", default="paper/figures/example_steps.pdf")
    return parser


def abridge(text: str, width: int = 70) -> str:
    text = re.sub(r"\s+", " ", text.replace(PREFIX_END, " [prefix ends] ")).strip()
    text = text.replace("$", r"\$")  # raw LaTeX would not survive truncation as mathtext
    return text if len(text) <= width else text[: width - 4].rstrip() + " ..."


def load_trace(data_dir: Path, match: str) -> tuple[dict, list[float]]:
    gains = json.loads((data_dir / "signals/step_answer_gain.json").read_text())
    for line in open(data_dir / "train-segmented.jsonl"):
        record = json.loads(line)
        if match in record["prompt"]:
            return record, gains[str(record["id"])]
    raise SystemExit(f"no trace whose problem contains {match!r} in {data_dir}")


def style(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", colors=MUTED, labelsize=6.5, width=0.6, length=2.5)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


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
    n = len(texts)
    rows = list(range(n))  # step k is drawn at y = k, top to bottom

    plt.rcParams.update({"font.family": "serif", "font.size": 7, "mathtext.fontset": "stix",
                         "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]})
    fig = plt.figure(figsize=(6.3, 0.17 * n + 0.55))
    grid = fig.add_gridspec(1, 3, width_ratios=[3.4, 1.2, 1.2], wspace=0.1)
    text_ax, gain_ax, weight_ax = (fig.add_subplot(grid[0, i]) for i in range(3))

    for ax in (text_ax, gain_ax, weight_ax):
        ax.set_ylim(n - 0.5, -0.5)
    text_ax.axis("off")
    for k in rows:
        text_ax.text(0.0, k, f"$s_{{{k + 1}}}$", ha="left", va="center", color=INK, fontsize=7)
        text_ax.text(0.055, k, labels[k], ha="left", va="center", color=INK, fontsize=6.5)
        text_ax.text(1.0, k, f"{lengths[k]}", ha="right", va="center", color=MUTED, fontsize=6.5)
    text_ax.set_xlim(0, 1)
    text_ax.text(0.0, -1.0, "Step (abridged)", ha="left", va="bottom", color=INK, fontsize=7)
    text_ax.text(1.0, -1.0, "tokens", ha="right", va="bottom", color=MUTED, fontsize=6.5)

    gain_ax.barh(rows, gains, height=0.7, color=[BLUE if g >= 0 else ORANGE for g in gains], zorder=2)
    gain_ax.axvline(0, color=MUTED, linewidth=0.6, zorder=3)
    gain_ax.set_title(r"(a) Answer gain $u_k$ (nats)", fontsize=7, color=INK, loc="left")
    span = max(abs(g) for g in gains) or 1.0
    gain_ax.set_xlim(min(0.0, min(gains)) - 0.4 * span, max(0.0, max(gains)) + 0.4 * span)
    for k, g in zip(rows, gains):
        gain_ax.text(g + (0.04 if g >= 0 else -0.04) * span, k, f"{g:+.2f}", ha="left" if g >= 0 else "right",
                     va="center", color=MUTED, fontsize=5.5)
    style(gain_ax)

    # Bars start at the uniform weight 1, so up- and down-weighted steps point opposite ways.
    weight_ax.barh(rows, [w - 1.0 for w in weights], left=1.0, height=0.7,
                   color=[BLUE if w >= 1 else ORANGE for w in weights], zorder=2)
    weight_ax.axvline(1.0, color=MUTED, linewidth=0.6, zorder=3)
    weight_ax.set_title(rf"(b) Weight $w_k$ ($\lambda={args.interpolation:g}$)", fontsize=7, color=INK, loc="left")
    reach = max(abs(w - 1.0) for w in weights) or 0.1
    weight_ax.set_xlim(min(1.0, min(weights)) - 0.45 * reach, max(1.0, max(weights)) + 0.45 * reach)
    for k, w in zip(rows, weights):
        weight_ax.text(w + (0.06 if w >= 1 else -0.06) * reach, k, f"{w:.2f}", ha="left" if w >= 1 else "right",
                       va="center", color=MUTED, fontsize=5.5)
    style(weight_ax)

    if prefix_step is not None:
        # The teacher prefix ends inside this step; the dashed line closes its row.
        for ax in (text_ax, gain_ax, weight_ax):
            ax.axhline(prefix_step + 0.5, color=MUTED, linewidth=0.7, linestyle=(0, (4, 2)), zorder=1,
                       clip_on=False)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    top = max(rows, key=lambda k: gains[k])
    print(f"wrote {args.output}: {n} steps, largest gain s{top + 1} ({gains[top]:+.2f}, w={weights[top]:.2f}), "
          f"weights {min(weights):.2f}-{max(weights):.2f}, token-weighted mean "
          f"{sum(w * n for w, n in zip(weights, lengths)) / sum(lengths):.3f}")


if __name__ == "__main__":
    main()
