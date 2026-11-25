"""Remove the scored answer from a response before reading answer gains (leak-free gain).

When a step states the final answer, log p(answer | prefix) jumps because the answer string is
already in the prefix: that step's gain measures copying, not reasoning. Masking every place the
answer is stated keeps the probe from copying it.

Short answers ("2", "17") also occur as ordinary numbers inside derivations, so masking every
occurrence would erase unrelated arithmetic. Long answers (>= `min_chars`) are masked wherever they
occur as a standalone token; short ones only in the final `tail_steps` steps, where the answer is
stated (98% of the P-ALIGN traces state it in their last two steps), plus inside any \\boxed{...}.
"""

import re

MASK = "[MASKED]"
_EDGE = r"0-9A-Za-z"


def standalone_pattern(answer: str) -> re.Pattern:
    # not glued to a letter/digit on either side, so "17" does not match inside "117" or "x17"
    return re.compile(rf"(?<![{_EDGE}]){re.escape(answer)}(?![{_EDGE}])")


def boxed_pattern(answer: str) -> re.Pattern:
    return re.compile(r"\\boxed\{\s*" + re.escape(answer) + r"\s*\}")


def mask_steps(
    steps: list[str], answers: str | list[str], min_chars: int = 3, tail_steps: int = 2, mask: str = MASK
) -> tuple[list[str], int]:
    """Return the masked step texts and the number of replacements made.

    `answers` may hold several strings (the trace's own answer and a different reference answer):
    each is masked under the same rule, longest first so a short answer cannot split a longer one.
    """
    if isinstance(answers, str):
        answers = [answers]
    answers = sorted({answer.strip() for answer in answers if answer and answer.strip()}, key=len, reverse=True)
    first_tail = len(steps) - tail_steps
    masked, count = list(steps), 0
    for answer in answers:
        boxed, standalone = boxed_pattern(answer), standalone_pattern(answer)
        for index, text in enumerate(masked):
            text, n_boxed = boxed.subn(lambda _: "\\boxed{" + mask + "}", text)
            n_plain = 0
            if len(answer) >= min_chars or index >= first_tail:
                text, n_plain = standalone.subn(mask, text)
            masked[index] = text
            count += n_boxed + n_plain
    return masked, count
