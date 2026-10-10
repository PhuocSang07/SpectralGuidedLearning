"""Appendix I figure: answer gain against step predictability, per training trace (1.5B student).

    python paper/make_score_analysis.py

Reads paper/data/score_analysis.csv (one row per trace: within-trace Spearman correlation between
the answer gains and the frozen student's mean token log-probability of each step, and the share
of the ceil(0.1 n) highest-gain steps that are also among the ceil(0.1 n) most predictable) and
writes paper/figures/score_analysis.pdf. The grey reference draws two independent random rankings
of each trace's steps (50 per trace, fixed seed), the value both statistics take when the two
scores are unrelated.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE = "#2a78d6"  # categorical slot 1, as in make_figures.py
INK, MUTED, GRID, NULL = "#0b0b0b", "#52514e", "#e4e3df", "#8f8e89"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "pdf.fonttype": 42})


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.6, length=2.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def independent_draws(n_steps: np.ndarray, reps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Spearman rho and top-10% overlap of two independent random rankings of each trace's steps."""
    rng = np.random.default_rng(seed)
    rho, overlap = [], []
    for n in n_steps:
        k = int(np.ceil(0.1 * n))
        for _ in range(reps):
            a, b = rng.permutation(n), rng.permutation(n)
            rho.append(1 - 6 * np.sum((a - b) ** 2) / (n * (n * n - 1)))
            overlap.append(np.sum((a < k) & (b < k)) / k)
    return np.array(rho), np.array(overlap)


def panel(ax, observed: pd.Series, reference: np.ndarray, bins: np.ndarray, xlabel: str) -> None:
    ax.hist(observed, bins=bins, weights=np.full(len(observed), 1 / len(observed)), color=BLUE,
            edgecolor="white", linewidth=0.8, zorder=2)
    ax.hist(reference, bins=bins, weights=np.full(len(reference), 1 / len(reference)), histtype="step",
            color=NULL, linewidth=1.2, zorder=3)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel("share of traces", color=INK)
    style(ax)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="paper/data/score_analysis.csv")
    parser.add_argument("--output", default="paper/figures/score_analysis.pdf")
    args = parser.parse_args()

    d = pd.read_csv(args.csv)
    rho_ref, overlap_ref = independent_draws(d.n_steps.to_numpy(), reps=50, seed=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.15), gridspec_kw={"wspace": 0.28})

    panel(a, d.spearman, rho_ref, np.arange(-0.65, 0.90, 0.05),
          r"Spearman $\rho$ between answer gain and predictability")
    med, q1, q3 = d.spearman.median(), d.spearman.quantile(0.25), d.spearman.quantile(0.75)
    a.axvspan(q1, q3, color=BLUE, alpha=0.08, zorder=1, linewidth=0)
    a.axvline(med, color=INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=4)
    a.text(0.98, 0.95, f"median {med:+.2f}\nIQR {q1:+.2f} to {q3:+.2f}".replace("-", "−"),
           transform=a.transAxes, ha="right", va="top", fontsize=7, color=INK)
    a.set_title("(a) Rank correlation within a trace", fontsize=8, color=INK, loc="left")

    panel(b, d.top10_overlap, overlap_ref, np.arange(-0.05, 1.10, 0.1),
          "share of top-10% answer-gain steps also top-10% predictable")
    b.set_xticks(np.arange(0, 1.01, 0.2))
    b.text(0.98, 0.95, f"mean {d.top10_overlap.mean():.2f} (independent: {overlap_ref.mean():.2f})\n"
           f"median {d.top10_overlap.median():.2f}", transform=b.transAxes, ha="right", va="top", fontsize=7,
           color=INK)
    b.set_title("(b) Overlap of the top-10% steps", fontsize=8, color=INK, loc="left")

    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Line2D([0], [0], color=NULL, linewidth=1.2)]
    fig.legend(handles, [f"observed ({len(d)} traces)", "independent scores (simulated)"], loc="upper center",
               ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.06))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.output}: rho median {med:.3f} (IQR {q1:.3f}..{q3:.3f}); top-10% overlap mean "
          f"{d.top10_overlap.mean():.3f}, median {d.top10_overlap.median():.3f}, independent {overlap_ref.mean():.3f}")


if __name__ == "__main__":
    main()
