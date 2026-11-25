"""Reference final answers for the records of a train-segmented.jsonl, from the source dataset.

P-ALIGN's training targets are stitched traces (teacher prefix + student continuation) whose last
\\boxed{} is the continuation's answer, which need not match the dataset's reference answer. This
writes {id: reference answer} for the records whose problem is found in s1K-1.1, for
sgl.signals.answer_gain --answer-source reference, and reports how often the two answers agree.

The reference is the last \\boxed{} of s1K's `solution` field, or the whole field when it is short
(s1K stores many bare answers such as "128"); longer solutions without a boxed answer are skipped,
since a multi-sentence target is not a final answer a probe can score.

    python -m sgl.data.reference_answers --data-path data/.../train-segmented.jsonl \
        --output data/.../signals/reference-answers.json
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

from sgl.data.s1k import last_boxed, load_rows
from sgl.eval.graders.answer_scoring import extract_boxed

KEY_CHARS = 200


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def reference_answer(solution: str, max_chars: int) -> str | None:
    boxed = last_boxed(solution or "")
    if boxed:
        return boxed
    solution = (solution or "").strip()
    return solution if solution and len(solution) <= max_chars else None


def match_references(records: list[dict], rows, max_chars: int) -> tuple[dict[str, str], dict]:
    """Map each record to its source problem by the problem's leading text inside the prompt."""
    keys = {}
    for row in rows:
        key = normalize(row["question"])[:KEY_CHARS]
        if key:
            keys[key] = reference_answer(row.get("solution"), max_chars)
    answers = {}
    stats = {"records": len(records), "unmatched": 0, "no_reference": 0, "agree_exact": 0}
    for record in records:
        prompt = normalize(record["prompt"])
        found = [key for key in keys if key in prompt]
        if not found:
            stats["unmatched"] += 1
            continue
        reference = keys[max(found, key=len)]
        if reference is None:
            stats["no_reference"] += 1
            continue
        answers[str(record["id"])] = reference
        trace = extract_boxed(record["response"])
        stats["agree_exact"] += bool(trace) and trace.strip() == reference.strip()
    stats["with_reference"] = len(answers)
    return answers, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl")
    parser.add_argument("--dataset", default="simplescaling/s1K-1.1", help="HF id, snapshot dir or local jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-answer-chars", type=int, default=50)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = [json.loads(line) for line in open(args.data_path)]
    answers, stats = match_references(records, load_rows(args.dataset, args.split), args.max_answer_chars)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(answers, ensure_ascii=False))
    print(json.dumps(stats))
    print(f"wrote {len(answers)} reference answers -> {args.output}")


if __name__ == "__main__":
    main()
