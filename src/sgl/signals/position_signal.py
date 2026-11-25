"""Position controls for the answer-gain allocation: replace a signal parquet's per-step signal.

Steps near the end of a trace tend to carry more answer gain, so a gain-weighted run could be
winning through position alone. Both modes keep everything else of the gain arm (the same
standardize / clip / exponentiate / token-mass / shrinkage construction in sgl.allocation.build):

    position   signal = relative step position k / (n - 1): weights rise with position only
    residual   signal = gain minus the corpus mean gain of its relative-position bin: the part
               of the gain that position does not explain

The relative position of step k (0-based) of an n-step trace is k / (n - 1), 0 for a one-step
trace. Bins split [0, 1] into `--bins` equal intervals, the last one closed.
"""

import argparse
from pathlib import Path

import pandas as pd

MODES = ("position", "residual")


def relative_positions(n_steps: int) -> list[float]:
    return [0.0] * n_steps if n_steps < 2 else [k / (n_steps - 1) for k in range(n_steps)]


def position_bin(position: float, bins: int) -> int:
    return min(int(position * bins), bins - 1)


def bin_means(signals: list[list[float]], bins: int) -> list[float]:
    """Mean signal per relative-position bin, pooled over every step of every trace."""
    totals, counts = [0.0] * bins, [0] * bins
    for signal in signals:
        for position, value in zip(relative_positions(len(signal)), signal):
            b = position_bin(position, bins)
            totals[b] += value
            counts[b] += 1
    return [total / count if count else 0.0 for total, count in zip(totals, counts)]


def derive(signals: list[list[float]], mode: str, bins: int = 10) -> list[list[float]]:
    if mode == "position":
        return [relative_positions(len(signal)) for signal in signals]
    if mode == "residual":
        means = bin_means(signals, bins)
        return [
            [value - means[position_bin(position, bins)]
             for position, value in zip(relative_positions(len(signal)), signal)]
            for signal in signals
        ]
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal", required=True, help="signal parquet whose step_entropies hold the gain")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--bins", type=int, default=10, help="residual: number of position bins")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    frame = pd.read_parquet(args.signal)
    signals = [list(map(float, values)) for values in frame["step_entropies"]]
    frame["step_entropies"] = derive(signals, args.mode, args.bins)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output)
    print(f"wrote {len(frame)} rows ({args.mode}) -> {args.output}")


if __name__ == "__main__":
    main()
