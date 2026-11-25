"""Diagnostics of a per-step answer-gain file (CPU only): where the gain sits and how the weights follow.

    python -m sgl.signals.gain_report --data-path data/r1-qwen-1.5b-palign/train-segmented.jsonl \
        --gains data/r1-qwen-1.5b-palign/signals/step_answer_gain.json \
        --tokenizer deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        [--compare masked=.../step_answer_gain-masked.json probe-so=...] --output report.json

Reports, for the main gain file:
  position   pooled and mean per-trace Spearman correlation of gain with relative step position;
             share of the positive gain and token-weighted mean allocation weight per position decile
  answer     share of steps stating the trace answer, their share of the positive gain, and how
             often the highest-gain step states it
  topk       at --keep-fraction, how many traces keep their final step under hard top-gain selection
  spread     token-weighted standard deviation of the allocation weights of the gain and of the
             position-only signal at each --temperatures value, to pick a position control with the
             same concentration as the gain arm
and for every --compare file: mean per-trace Spearman correlation with the main gain and the overlap
of the top-3 steps, i.e. whether the step ranking survives a different probe / masking / target.
"""

import argparse
import json
import statistics
from pathlib import Path

from sgl.allocation.build import top_signal_steps
from sgl.allocation.iwc import stable_iwc_step_weights
from sgl.eval.graders.answer_scoring import extract_boxed
from sgl.signals.answer_gain import step_texts
from sgl.signals.answer_mask import boxed_pattern, standalone_pattern
from sgl.signals.position_signal import position_bin, relative_positions


def ranks(values: list[float]) -> list[float]:
    """Average ranks (ties share the mean rank)."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2
        i = j + 1
    return result


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return None if sxx == 0 or syy == 0 else sxy / (sxx * syy) ** 0.5


def states_answer(text: str, answer: str) -> bool:
    return bool(boxed_pattern(answer).search(text) or standalone_pattern(answer).search(text))


def weighted_std(weights: list[float], lengths: list[int]) -> float:
    total = sum(lengths)
    mean = sum(w * n for w, n in zip(weights, lengths)) / total
    return (sum(n * (w - mean) ** 2 for w, n in zip(weights, lengths)) / total) ** 0.5


def report(records, gains, lengths, texts, args, bins: int = 10) -> dict:
    pooled_gain, pooled_pos, per_trace = [], [], []
    positive_by_bin, weight_mass, token_mass = [0.0] * bins, [0.0] * bins, [0] * bins
    answer_steps = steps = 0
    answer_positive = positive_total = 0.0
    top_is_answer = top_counted = topk_keeps_last = 0
    top_positions = []
    for record in records:
        rid = str(record["id"])
        gain, length = gains[rid], lengths[rid]
        positions = relative_positions(len(gain))
        pooled_gain += gain
        pooled_pos += positions
        rho = spearman(gain, positions)
        if rho is not None:
            per_trace.append(rho)
        weights = stable_iwc_step_weights(gain, length, args.temperature, args.interpolation, args.clip)
        for position, value, weight, n in zip(positions, gain, weights, length):
            b = position_bin(position, bins)
            positive_by_bin[b] += max(value, 0.0)
            weight_mass[b] += weight * n
            token_mass[b] += n
        top = max(range(len(gain)), key=lambda i: (gain[i], -i))
        top_positions.append(positions[top])
        topk_keeps_last += (len(gain) - 1) in top_signal_steps(gain, length, args.keep_fraction)
        answer = extract_boxed(record["response"])
        if answer and texts is not None:
            stated = [states_answer(text, answer.strip()) for text in texts[rid]]
            steps += len(stated)
            answer_steps += sum(stated)
            answer_positive += sum(max(v, 0.0) for v, s in zip(gain, stated) if s)
            positive_total += sum(max(v, 0.0) for v in gain)
            top_is_answer += stated[top]
            top_counted += 1
    positive_sum = sum(positive_by_bin) or 1.0
    result = {
        "traces": len(records),
        "position": {
            "spearman_pooled": spearman(pooled_gain, pooled_pos),
            "spearman_per_trace_mean": statistics.fmean(per_trace) if per_trace else None,
            "positive_gain_share_by_decile": [round(v / positive_sum, 4) for v in positive_by_bin],
            "mean_weight_by_decile": [round(w / n, 4) if n else None for w, n in zip(weight_mass, token_mass)],
            "top_step_position_median": statistics.median(top_positions),
        },
        "topk": {"keep_fraction": args.keep_fraction, "traces_keeping_final_step": topk_keeps_last / len(records)},
    }
    if top_counted:
        result["answer"] = {
            "steps_stating_answer": answer_steps / steps,
            "positive_gain_share_of_answer_steps": answer_positive / (positive_total or 1.0),
            "top_step_states_answer": top_is_answer / top_counted,
        }
    spread = {}
    for tau in args.temperatures:
        for name, signal in (("gain", None), ("position", "position")):
            values = [
                weighted_std(
                    stable_iwc_step_weights(
                        gains[str(r["id"])] if signal is None else relative_positions(len(gains[str(r["id"])])),
                        lengths[str(r["id"])], tau, args.interpolation, args.clip,
                    ),
                    lengths[str(r["id"])],
                )
                for r in records
            ]
            spread[f"{name}@tau={tau}"] = round(statistics.fmean(values), 4)
    result["weight_spread"] = spread
    return result


def compare(main: dict, other: dict) -> dict:
    rhos, overlaps = [], []
    for rid, gain in main.items():
        alt = other.get(rid)
        if alt is None or len(alt) != len(gain):
            continue
        rho = spearman(gain, alt)
        if rho is not None:
            rhos.append(rho)
        k = min(3, len(gain))
        top = set(sorted(range(len(gain)), key=lambda i: -gain[i])[:k])
        top_alt = set(sorted(range(len(alt)), key=lambda i: -alt[i])[:k])
        overlaps.append(len(top & top_alt) / k)
    return {
        "traces": len(rhos),
        "spearman_mean": statistics.fmean(rhos) if rhos else None,
        "spearman_median": statistics.median(rhos) if rhos else None,
        "top3_overlap_mean": statistics.fmean(overlaps) if overlaps else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--gains", required=True)
    parser.add_argument("--tokenizer", help="tokenizer of the stored ids; enables the answer-step statistics")
    parser.add_argument("--compare", nargs="*", default=[], metavar="NAME=PATH")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--interpolation", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--keep-fraction", type=float, default=0.59)
    parser.add_argument("--temperatures", type=float, nargs="*", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    gains = json.loads(Path(args.gains).read_text())
    records = [json.loads(line) for line in open(args.data_path)]
    records = [r for r in records if str(r["id"]) in gains and len(gains[str(r["id"])]) == len(r["steps"])]
    lengths = {str(r["id"]): [s["token_end"] - s["token_start"] for s in r["steps"]] for r in records}
    texts = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        texts = {str(r["id"]): step_texts(tokenizer, r) for r in records}
    result = report(records, gains, lengths, texts, args)
    for item in args.compare:
        name, path = item.split("=", 1)
        result.setdefault("compare", {})[name] = compare(gains, json.loads(Path(path).read_text()))
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
