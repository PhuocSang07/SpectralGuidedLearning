"""Controls of the answer-gain signal: answer masking, reference answers, position signals, hard top-k."""

import json
from pathlib import Path

import pytest

from sgl.allocation.build import build_example, top_signal_steps
from sgl.data.reference_answers import match_references, reference_answer
from sgl.signals.answer_mask import MASK, mask_steps
from sgl.signals.gain_report import compare, ranks, spearman
from sgl.signals.position_signal import bin_means, derive, relative_positions

TOKENIZER = Path(__file__).parent / "regression" / "assets" / "tokenizer"


# --- answer masking ---------------------------------------------------------------------------

def test_long_answer_is_masked_in_every_step():
    steps = ["We guess 128 early. ", "Check: 2^7 = 128. ", "So \\boxed{128}."]
    masked, count = mask_steps(steps, "128")
    assert "128" not in "".join(masked)
    assert masked[2] == f"So \\boxed{{{MASK}}}."
    assert count == 3


def test_short_answer_is_masked_only_in_the_tail_and_in_boxed():
    steps = ["Take 2 apples. ", "Add 3 more, \\boxed{5} is wrong. ", "Then 5 left. ", "Answer is 5."]
    masked, count = mask_steps(steps, "5", tail_steps=2)
    assert masked[0] == steps[0]                          # untouched
    assert masked[1] == f"Add 3 more, \\boxed{{{MASK}}} is wrong. "  # boxed anywhere
    assert masked[2:] == [f"Then {MASK} left. ", f"Answer is {MASK}."]
    assert count == 3


def test_answer_inside_a_longer_number_is_not_masked():
    masked, count = mask_steps(["117 and 1170 and x17 differ. ", "Final: 17"], "17", tail_steps=2)
    assert masked == ["117 and 1170 and x17 differ. ", f"Final: {MASK}"]
    assert count == 1


def test_two_answers_are_both_masked_longest_first():
    masked, _ = mask_steps(["Got 246, not 117. ", "So 117."], ["117", "246"])
    assert masked == [f"Got {MASK}, not {MASK}. ", f"So {MASK}."]


def test_empty_answer_leaves_steps_untouched():
    steps = ["a. ", "b."]
    assert mask_steps(steps, ["", None]) == (steps, 0)


def test_masked_response_keeps_step_boundaries_in_token_space():
    transformers = pytest.importorskip("transformers")
    from sgl.signals.answer_gain import masked_response, step_texts

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(TOKENIZER))
    # every step after the first opens with a space-prefixed word, so each boundary is a token start
    # (a token belongs to the step holding its first character)
    texts = ["First we find 128 apples.", " Then we check it twice.", " So the answer is \\boxed{128}."]
    response = "".join(texts)
    prompt_ids = tokenizer("Q: apples?", add_special_tokens=False)["input_ids"]
    encoding = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    starts = [o[0] for o in encoding["offset_mapping"]]
    cuts, position = [], 0
    for text in texts:  # token span of each step: tokens whose first character lies in the step
        cuts.append(sum(1 for s in starts if s < position + len(text)))
        position += len(text)
    begin = len(prompt_ids)
    steps = [{"token_start": begin + a, "token_end": begin + b} for a, b in zip([0] + cuts[:-1], cuts)]
    record = {
        "id": 0, "response": response, "input_ids": prompt_ids + encoding["input_ids"],
        "response_token_span": [begin, begin + len(encoding["input_ids"])], "steps": steps,
    }
    assert "".join(step_texts(tokenizer, record)) == response
    assert step_texts(tokenizer, record) == texts

    masked_ids, step_cuts, n_masked = masked_response(tokenizer, record, ["128"], 3, 2)
    decoded = tokenizer.decode(masked_ids)
    assert n_masked == 2 and "128" not in decoded and MASK in decoded
    assert step_cuts == sorted(step_cuts) and step_cuts[-1] == len(masked_ids)
    # the middle step has no answer: its decoded tokens are unchanged
    assert tokenizer.decode(masked_ids[step_cuts[0]:step_cuts[1]]) == texts[1]


# --- reference answers ------------------------------------------------------------------------

def test_reference_answer_prefers_boxed_then_short_solution():
    assert reference_answer("so \\boxed{42}", 50) == "42"
    assert reference_answer("128", 50) == "128"
    assert reference_answer("a long derivation " * 10, 50) is None


def test_match_references_by_problem_text_inside_the_prompt():
    rows = [
        {"question": "Find  x such that x+1=3.", "solution": "2"},
        {"question": "Compute 3*4.", "solution": "\\boxed{12}"},
        {"question": "Explain why.", "solution": "because " * 20},
    ]
    records = [
        {"id": 0, "prompt": "<user>Please reason.Find x such that\nx+1=3.<assistant>", "response": "\\boxed{2}"},
        {"id": 1, "prompt": "<user>Compute 3*4.<assistant>", "response": "\\boxed{13}"},
        {"id": 2, "prompt": "<user>Explain why.<assistant>", "response": "x"},
        {"id": 3, "prompt": "<user>Unknown.<assistant>", "response": "\\boxed{1}"},
    ]
    answers, stats = match_references(records, rows, 50)
    assert answers == {"0": "2", "1": "12"}
    assert stats == {"records": 4, "unmatched": 1, "no_reference": 1, "agree_exact": 1, "with_reference": 2}


# --- position signals -------------------------------------------------------------------------

def test_relative_positions():
    assert relative_positions(1) == [0.0]
    assert relative_positions(5) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_position_mode_ignores_the_gain():
    assert derive([[5.0, -1.0, 2.0]], "position") == [[0.0, 0.5, 1.0]]


def test_residual_removes_the_positional_mean():
    signals = [[1.0, 3.0], [3.0, 5.0]]  # position 0 mean 2, position 1 mean 4
    assert bin_means(signals, 2) == [2.0, 4.0]
    assert derive(signals, "residual", bins=2) == [[-1.0, -1.0], [1.0, 1.0]]


def test_position_signal_module_rewrites_the_parquet(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from sgl.signals.position_signal import main

    source = tmp_path / "gain.parquet"
    pd.DataFrame([{"id": 0, "n_steps": 3, "step_strengths": [1.0] * 3, "step_entropies": [9.0, 0.0, 1.0]}]
                 ).to_parquet(source)
    main(["--signal", str(source), "--mode", "position", "--output", str(tmp_path / "pos.parquet")])
    out = pd.read_parquet(tmp_path / "pos.parquet")
    assert list(out.step_entropies[0]) == [0.0, 0.5, 1.0]
    assert list(out.step_strengths[0]) == [1.0] * 3


# --- hard top-k selection on the gain ---------------------------------------------------------

def test_top_signal_steps_reaches_the_token_budget_highest_first():
    signal, lengths = [0.1, 5.0, -1.0, 3.0], [10, 10, 10, 10]
    assert top_signal_steps(signal, lengths, 0.5) == [1, 3]
    assert top_signal_steps(signal, lengths, 0.51) == [0, 1, 3]
    assert top_signal_steps(signal, lengths, 1.0) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        top_signal_steps(signal, lengths, 0.0)


def test_gain_topk_variant_drops_low_gain_steps_at_unit_weight():
    record = {
        "id": 7, "input_ids": list(range(12)), "response_token_span": [2, 10],
        "steps": [{"token_start": 2, "token_end": 4}, {"token_start": 4, "token_end": 8},
                  {"token_start": 8, "token_end": 10}],
    }
    example, stats = build_example(
        record, [1.0, 1.0, 1.0], [0.0, -2.0, 3.0], 1.0, "gain-topk", 2.0, 0.5, 2.0, 1e-8, keep_fraction=0.25,
    )
    assert example["loss_mask"] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # last step + stop tokens
    assert example["loss_weights"][8:] == [1.0] * 4
    assert stats["step_drop"] == pytest.approx(2 / 3)


# --- rank statistics --------------------------------------------------------------------------

def test_ranks_and_spearman():
    assert ranks([3.0, 1.0, 1.0]) == [2.0, 0.5, 0.5]
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2], [1, 2]) is None


def test_compare_reports_rank_agreement():
    main = {"0": [0.1, 0.5, 0.2, 0.9], "1": [1.0, 2.0, 3.0]}
    same = json.loads(json.dumps(main))
    result = compare(main, same)
    assert result["spearman_mean"] == pytest.approx(1.0)
    assert result["top3_overlap_mean"] == pytest.approx(1.0)
