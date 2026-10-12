"""Answer gain against step predictability, per training trace (1.5B student).

    python paper/make_score_analysis.py

Reads paper/data/score_analysis.csv (one row per trace: number of steps, within-trace Spearman
correlation between the answer gains and the frozen student's mean token log-probability of each
step, and the share of the ceil(0.1 n) highest-gain steps that are also among the ceil(0.1 n) most
predictable) and writes paper/figures/score_analysis.pdf.

(a) plots each trace's correlation against its number of steps, over the 95% range that two
independent rankings of the same number of steps produce. (b) compares the top-10% overlap with
the same independent rankings. The reference is simulated with a fixed seed.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE = "#2a78d6"  # categorical slot 1, as in make_figures.py
NULL, NULL_LINE = "#c9c8c2", "#8f8e89"  # the independent-rankings reference stays recessive
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "pdf.fonttype": 42})


def style(ax, grid_axis: str) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.6, length=2.5)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def spearman_null(n: int, reps: int, rng) -> np.ndarray:
    """Spearman rho of two independent random rankings of n steps."""
    a = np.argsort(rng.random((reps, n)), axis=1)
    b = np.argsort(rng.random((reps, n)), axis=1)
    return 1 - 6 * np.sum((a - b) ** 2, axis=1) / (n * (n * n - 1))


def overlap_null(n: int, reps: int, rng) -> np.ndarray:
    """Share of the top ceil(0.1 n) of one random ranking that is also in the top of another."""
    k = int(np.ceil(0.1 * n))
    a = np.argsort(rng.random((reps, n)), axis=1) < k
    b = np.argsort(rng.random((reps, n)), axis=1) < k
    return np.sum(a & b, axis=1) / k


def draw_rho(ax, d: pd.DataFrame, rho_null: np.ndarray, outside: float, xlabel: str) -> None:
    """Histogram of the per-trace correlation, against independent rankings of the same traces."""
    bins = np.arange(-0.8, 0.85, 0.05)
    ax.hist(d.spearman, bins=bins, weights=np.full(len(d), 1 / len(d)), color=BLUE, edgecolor="white",
            linewidth=0.6, zorder=2)
    ax.hist(rho_null, bins=bins, weights=np.full(len(rho_null), 1 / len(rho_null)), histtype="step",
            color=NULL_LINE, linewidth=1.1, zorder=3)
    ax.set_xlim(-0.8, 0.85)
    ax.set_ylim(0, 0.16)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.04))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel("Share of traces", color=INK)
    # the ensemble claim is "no correlation": state the median and how often a trace leaves the chance range
    # top right is empty (few traces have rho > 0.3), so the note does not cover any bar
    ax.text(0.99, 0.96, f"median $\\rho$ = {d.spearman.median():+.2f}".replace("-", "−")
            + f"\n{outside:.0%} outside chance range\n(5% expected)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK, linespacing=1.2)
    style(ax, "y")


def legend_handles(n: int):
    return ([plt.Rectangle((0, 0), 1, 1, color=BLUE),
             plt.Rectangle((0, 0), 1, 1, facecolor=NULL, edgecolor=NULL_LINE, linewidth=0.8)],
            [f"Observed ({n} traces)", "Independent rankings (simulated)"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="paper/data/score_analysis.csv")
    parser.add_argument("--output", default="paper/figures/score_analysis.pdf")
    parser.add_argument("--output-a", default="paper/figures/score_analysis_a.pdf",
                        help="panel (a) alone, sized for one ACL column")
    args = parser.parse_args()

    d = pd.read_csv(args.csv)
    rng = np.random.default_rng(0)

    # 95% range of rho under independence, for every trace length in the data
    lengths = np.unique(d.n_steps)
    band = {n: np.quantile(spearman_null(n, 4000, rng), [0.025, 0.975]) for n in lengths}
    lo = d.n_steps.map(lambda n: band[n][0])
    hi = d.n_steps.map(lambda n: band[n][1])
    outside = float(np.mean((d.spearman < lo) | (d.spearman > hi)))

    ov_null = np.concatenate([overlap_null(n, 200, rng) for n in d.n_steps])
    cats = ["no overlap", "under half", "half or more"]

    def shares(x) -> np.ndarray:
        x = np.asarray(x)
        return np.array([np.mean(x == 0), np.mean((x > 0) & (x < 0.5)), np.mean(x >= 0.5)])

    obs_sh, null_sh = shares(d.top10_overlap), shares(ov_null)

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.2), gridspec_kw={"wspace": 0.38, "width_ratios": [1.15, 1]})

    # (a) distribution of the per-trace correlation, against independent rankings of the same traces
    rho_null = np.concatenate([spearman_null(n, 50, rng) for n in d.n_steps])
    draw_rho(a, d, rho_null, outside, r"Spearman $\rho$ between answer gain and predictability")
    a.set_title("(a) Rank correlation within each trace", fontsize=8, color=INK, loc="left")

    # (b) top-10% overlap, observed against independent rankings
    y = np.arange(len(cats))[::-1]
    h = 0.36
    a_bars = b.barh(y + h / 2, obs_sh, height=h, color=BLUE, zorder=2)
    n_bars = b.barh(y - h / 2, null_sh, height=h, color=NULL, edgecolor=NULL_LINE, linewidth=0.6, zorder=2)
    for bars, vals in ((a_bars, obs_sh), (n_bars, null_sh)):
        for bar, v in zip(bars, vals):
            b.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2, f"{v:.0%}", va="center",
                   fontsize=7, color=INK)
    b.set_yticks(y, cats)
    b.set_xlim(0, 0.68)
    b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    b.set_xlabel("Share of traces", color=INK)
    b.set_title("(b) Overlap of the top-10% steps", fontsize=8, color=INK, loc="left")
    b.tick_params(axis="y", length=0)
    style(b, "x")

    fig.legend(*legend_handles(len(d)), loc="upper center", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 1.07))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.output}: rho median {d.spearman.median():.3f}, outside band {outside:.3f}; "
          f"overlap shares observed {np.round(obs_sh, 3)}, independent {np.round(null_sh, 3)}; "
          f"mean overlap {d.top10_overlap.mean():.3f} vs {ov_null.mean():.3f}")

    # panel (a) alone, for a single-column figure
    fig_a, ax = plt.subplots(figsize=(3.2, 1.8))
    draw_rho(ax, d, rho_null, outside, r"Within-trace Spearman $\rho$, answer gain vs. predictability")
    handles, labels = legend_handles(len(d))
    ax.legend(handles, [labels[0], "Independent rankings"], loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, fontsize=7, handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)
    fig_a.savefig(args.output_a, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.output_a}")


if __name__ == "__main__":
    main()
