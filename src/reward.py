"""Reference implementation of the RL reward function designed in
docs/system_design.md section 3.2.

The brief explicitly grades "the creativity and logic behind how you
formulate the RL pipeline and reward function" for Part A, which is a
no-code design section. This module exists so that design isn't only a
paragraph of prose: the exact reward formula is executable and unit-tested
(tests/test_reward.py), against concrete example completions a generative
policy might emit. It is NOT wired into a training loop -- RL training
itself remains intentionally undesigned-in-code, per docs/system_design.md
section 0 (SFT alone already solves this deterministic task; RL is an
optional alignment layer for a hypothetical generative decoder variant).
"""
import re

HEX_PATTERN = re.compile(r"^0x[0-9a-f]+$")

FORMAT_REWARD = 0.2
VALIDITY_REWARD = 0.1
CORRECTNESS_REWARD = 0.7
MAX_PARTIAL_CREDIT = 0.3
MALFORMED_PENALTY = -0.5


def compute_reward(raw_output: str, ground_truth_decimal: int) -> float:
    """Score a policy's raw text completion against the ground-truth decimal
    value. See docs/system_design.md section 3.2 for the full design
    rationale behind the tiering and the specific constants used here.
    """
    match = HEX_PATTERN.match(raw_output.strip().lower())
    if match is None:
        return MALFORMED_PENALTY

    try:
        predicted_value = int(match.group(0), 16)
    except ValueError:
        # Unreachable given HEX_PATTERN already constrains to valid hex
        # digits, kept as a defensive branch matching the design doc.
        return FORMAT_REWARD - 0.3

    if predicted_value == ground_truth_decimal:
        correctness_reward = CORRECTNESS_REWARD
    else:
        max_val = max(predicted_value, ground_truth_decimal, 1)
        closeness = 1 - abs(predicted_value - ground_truth_decimal) / max_val
        correctness_reward = MAX_PARTIAL_CREDIT * max(0.0, closeness)

    return FORMAT_REWARD + VALIDITY_REWARD + correctness_reward


DEMO_EXAMPLES = [
    ("garbage output", None, 420, "malformed"),
    ("0xzz", None, 420, "malformed (invalid hex digits)"),
    ("0x1a3", 419, 420, "valid hex, off by one (near miss)"),
    ("0xfff", 4095, 420, "valid hex, far off"),
    ("0x1a4", 420, 420, "exact match"),
    ("  0X1A4  ", 420, 420, "exact match, whitespace/case variation"),
]


if __name__ == "__main__":
    print(f"{'completion':<16} {'condition':<40} {'reward':>8}")
    print("-" * 66)
    for raw, _predicted, ground_truth, label in DEMO_EXAMPLES:
        r = compute_reward(raw, ground_truth)
        print(f"{raw.strip():<16} {label:<40} {r:>8.3f}")
