"""Figures for the manuscript. Run from SpectralGuidedLearning/ with the iwc venv active:

    python paper/make_figures.py

(a) within-trace entropy percentile of the step with the largest answer gain, excluding the
    step that writes \\boxed{} (it is the top step in about half of the traces by construction);
(b) paired difference to NLL per sampling seed, for the three weighting arms.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

DATA = Path("data/r1-qwen-1.5b-palign")
OUT = Path("paper/figures")
BENCHMARKS = ["aime25", "aime24", "amc12", "math500"]
ARMS = {
    "NLL": ["results/sft-nll-r1-qwen-1.5b-palign", "results_evalseed/sft-nll-e43", "results_evalseed/sft-nll-e44"],
    "Entropy": ["results/iwc-nogate-l05-r1-qwen-1.5b-palign", "results_evalseed/iwc-nogate-l05-e43",
                "results_evalseed/iwc-nogate-l05-e44"],
    "Shuffled": ["results/iwc-shuf-l05-s42-r1-qwen-1.5b-palign", "results_evalseed/iwc-shuf-l05-s42-e43",
                 "results_evalseed/iwc-shuf-l05-s42-e44"],
    "Answer gain": ["results/iwc-gain-l05-s42-r1-qwen-1.5b-palign", "results_evalseed/iwc-gain-l05-s42-e43",
                    "results_evalseed/iwc-gain-l05-s42-e44"],
}
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 of the dataviz reference palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def top_gain_entropy_percentiles() -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    gains = json.loads((DATA / "signals/step_answer_gain.json").read_text())
    entropies = pd.read_parquet(DATA / "spectral-strengths.parquet").set_index("id")["step_entropies"]
    percentiles = []
    for line in open(DATA / "train-segmented.jsonl"):
        record = json.loads(line)
        gain = np.array(gains[str(record["id"])])
        entropy = np.array(entropies.loc[record["id"]])
        if len(gain) != len(entropy) or len(gain) < 6:
            continue
        keep = np.array([
            "\\boxed" not in tokenizer.decode(record["input_ids"][s["token_start"]:s["token_end"]])
            for s in record["steps"]
        ])
        top = int(np.argmax(np.where(keep, gain, -np.inf)))
        percentiles.append(np.argsort(np.argsort(entropy))[top] / (len(entropy) - 1))
    return np.array(percentiles)


def per_seed_average(paths: list[str], metric: str) -> np.ndarray:
    values = []
    for path in paths:
        rows = {row["benchmark"]: row for row in json.load(open(Path(path) / "summary.json"))}
        values.append(100 * np.mean([rows[b][metric] for b in BENCHMARKS]))
    return np.array(values)


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.6, length=2.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8})
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.3, 2.2), gridspec_kw={"width_ratios": [1, 1.15]})

    percentiles = top_gain_entropy_percentiles()
    shares = np.histogram(percentiles, bins=10, range=(0, 1))[0] / len(percentiles) * 100
    left.bar(np.arange(10) * 10 + 5, shares, width=8.6, color=BLUE, zorder=2)
    left.axhline(10, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=3)
    left.text(75, 10.6, "uniform", color=MUTED, fontsize=7, ha="center", va="bottom")
    left.set_xlim(0, 100)
    left.set_xlabel("Entropy percentile within trace", color=INK, fontsize=8)
    left.set_ylabel("% of traces", color=INK, fontsize=8)
    left.set_title(f"(a) Top answer-gain step (n={len(percentiles)})", fontsize=8, color=INK, loc="left")
    style(left)

    arms = ["Entropy", "Shuffled", "Answer gain"]
    for metric, color, offset, label in (("pass@1", BLUE, -0.14, "pass@1"), ("pass@3", ORANGE, 0.14, "pass@3")):
        base = per_seed_average(ARMS["NLL"], metric)
        for i, arm in enumerate(arms):
            delta = per_seed_average(ARMS[arm], metric) - base
            x = np.full(3, i + offset) + np.array([-0.04, 0.0, 0.04])
            right.scatter(x, delta, s=16, color=color, edgecolor="white", linewidth=0.8, zorder=3,
                          label=label if i == 0 else None)
            right.plot([i + offset - 0.09, i + offset + 0.09], [delta.mean()] * 2, color=INK, linewidth=1.2,
                       zorder=4)
    right.axhline(0, color=MUTED, linewidth=0.8, zorder=2)
    right.set_xticks(range(len(arms)), arms)
    right.set_xlim(-0.5, len(arms) - 0.5)
    right.set_ylabel("Difference to NLL (points)", color=INK, fontsize=8)
    right.set_title("(b) Paired with NLL, per sampling seed", fontsize=8, color=INK, loc="left")
    right.legend(frameon=False, fontsize=7, loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    style(right)

    fig.tight_layout(w_pad=1.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure1.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure1.png", dpi=200, bbox_inches="tight")
    print(f"top-gain step: {np.mean(percentiles < 0.5):.1%} below the median entropy, "
          f"median percentile {np.median(percentiles):.2f}")


if __name__ == "__main__":
    main()
