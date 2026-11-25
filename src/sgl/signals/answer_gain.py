"""Per-step answer-information gain: how much each reasoning step raises the frozen student's
log-probability of the target final answer.

    gain_k = log p(answer | prompt, steps_1..k) - log p(answer | prompt, steps_1..k-1)

p(answer | context) is read by appending a fixed probe ("... the final answer is \\boxed{") and
scoring the answer tokens + "}" with vLLM prompt_logprobs. All probes of a sample share the
response prefix, so vLLM's prefix cache makes this ~one forward pass per sample. By default the
answer is the last \\boxed{...} of the training response (the "trace answer"); P-ALIGN's student
continuation can end on an answer that differs from the dataset's reference, so
--answer-source reference scores the {id: answer} map of sgl.data.reference_answers instead.
Output: {id: [gain per step]} JSON, aligned with train-segmented.jsonl steps.

Controls for robustness analyses, all off by default (the default run is unchanged):
  --probe        another probe string, to test that step rankings do not hinge on its wording
  --mask-answer  replace the answer wherever the response states it (sgl.signals.answer_mask)
                 before scoring, so a step cannot gain merely by already containing the answer
"""

import argparse
import bisect
import json
from pathlib import Path

from sgl.eval.graders.answer_scoring import extract_boxed
from sgl.signals.answer_mask import MASK, mask_steps

PROBE = "\n\nTherefore, the final answer is \\boxed{"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--probe", default=PROBE, help="text appended before the scored answer")
    parser.add_argument(
        "--answer-source", choices=["trace", "reference"], default="trace",
        help="trace = last \\boxed{} of the response; reference = --reference-answers, falling back "
        "to the trace answer for samples without a reference",
    )
    parser.add_argument("--reference-answers", help="{id: answer} JSON from sgl.data.reference_answers")
    parser.add_argument("--mask-answer", action="store_true", help="mask the answer in every scored prefix")
    parser.add_argument(
        "--mask-min-chars", type=int, default=3,
        help="answers at least this long are masked everywhere; shorter ones only in the tail steps",
    )
    parser.add_argument("--mask-tail-steps", type=int, default=2)
    return parser


def step_texts(tokenizer, record: dict) -> list[str]:
    """The response split at the stored step boundaries (lossless: the pieces join to the response).

    Steps are stored as token spans; a token belongs to the step holding its first character, so
    each boundary is the first character of the step's first token. The first step also takes any
    text before it and the last one any text after it, so no character is lost.
    """
    response = record["response"]
    start, end = record["response_token_span"]
    encoding = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    if len(encoding["input_ids"]) != end - start:
        raise ValueError(f"record {record['id']}: response re-tokenizes to a different length")
    char_starts = [offset[0] for offset in encoding["offset_mapping"]]
    bounds = [char_starts[step["token_start"] - start] for step in record["steps"]]
    bounds = [0] + bounds[1:] + [len(response)]
    return [response[a:b] for a, b in zip(bounds, bounds[1:])]


def masked_response(tokenizer, record: dict, answers: list[str], min_chars: int, tail_steps: int):
    """Token ids of the masked response, the token cut after each step, and the mask count.

    The masked response is tokenized once as a whole (never per step: BPE is context dependent) and
    each step's end is mapped back from character to token space.
    """
    texts, n_masked = mask_steps(step_texts(tokenizer, record), answers, min_chars, tail_steps, MASK)
    encoding = tokenizer("".join(texts), add_special_tokens=False, return_offsets_mapping=True)
    starts = [offset[0] for offset in encoding["offset_mapping"]]
    cuts, position = [], 0
    for text in texts:
        position += len(text)
        cuts.append(bisect.bisect_left(starts, position))
    return encoding["input_ids"], cuts, n_masked


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    references = {}
    if args.answer_source == "reference":
        if not args.reference_answers:
            parser.error("--answer-source reference needs --reference-answers")
        references = json.loads(Path(args.reference_answers).read_text())

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = [json.loads(line) for line in open(args.data_path)][: args.limit]
    probe_ids = tokenizer(args.probe, add_special_tokens=False)["input_ids"]

    prompts, index = [], []  # index[i] = (record id, step k (0 = before any step), n answer tokens)
    skipped, from_reference, n_masked_total, n_masked_records = 0, 0, 0, 0
    for record in records:
        trace_answer = extract_boxed(record["response"])
        reference = references.get(str(record["id"]))
        answer = reference or trace_answer
        from_reference += bool(reference)
        if not answer:
            skipped += 1
            continue
        answer_ids = tokenizer(answer + "}", add_special_tokens=False)["input_ids"]
        start = record["response_token_span"][0]
        if args.mask_answer:
            # mask both the scored answer and, when it differs, the answer the trace itself states
            response_ids, step_cuts, n_masked = masked_response(
                tokenizer, record, [answer, trace_answer or ""], args.mask_min_chars, args.mask_tail_steps
            )
            n_masked_total += n_masked
            n_masked_records += n_masked > 0
            context_ids = record["input_ids"][:start] + response_ids
            cuts = [start] + [start + cut for cut in step_cuts]
        else:
            context_ids = record["input_ids"]
            cuts = [start] + [step["token_end"] for step in record["steps"]]
        for k, cut in enumerate(cuts):
            ids = context_ids[:cut] + probe_ids + answer_ids
            if len(ids) >= args.max_model_len:
                continue
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            index.append((record["id"], k, len(answer_ids)))
    print(f"{len(prompts):,} probes over {len(records) - skipped} samples (skipped {skipped} without \\boxed)")
    if args.answer_source == "reference":
        print(f"reference answer scored for {from_reference} samples, trace answer for the rest")
    if args.mask_answer:
        print(f"masked {n_masked_total:,} answer occurrences in {n_masked_records} samples")

    llm = LLM(
        model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, enable_prefix_caching=True, enforce_eager=True,
    )
    outputs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0))

    logp = {}  # (id, k) -> log p(answer | prefix up to step k)
    for (rid, k, n_answer), out, prompt in zip(index, outputs, prompts):
        ids = prompt["prompt_token_ids"]
        total = 0.0
        for position in range(len(ids) - n_answer, len(ids)):
            total += out.prompt_logprobs[position][ids[position]].logprob
        logp[(rid, k)] = total

    gains = {}
    for record in records:
        rid, n_steps = record["id"], len(record["steps"])
        if (rid, 0) not in logp:
            continue
        values = [logp.get((rid, k)) for k in range(n_steps + 1)]
        # a step whose probe was skipped (too long) inherits the last known value: zero gain
        for k in range(1, len(values)):
            if values[k] is None:
                values[k] = values[k - 1]
        gains[rid] = [values[k] - values[k - 1] for k in range(1, n_steps + 1)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(gains))
    print(f"wrote gains for {len(gains)} samples -> {args.output}")


if __name__ == "__main__":
    main()
