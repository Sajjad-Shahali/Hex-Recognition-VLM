"""Unit tests for the RL reward function (src/reward.py), validating the
design properties claimed in docs/system_design.md section 3.2: tiering,
the correctness ceiling, and malformed-output penalization."""
from reward import compute_reward, MALFORMED_PENALTY


def test_exact_match_gets_max_reward():
    assert compute_reward("0x1a4", 420) == 1.0


def test_exact_match_is_case_and_whitespace_insensitive():
    assert compute_reward("  0X1A4  ", 420) == 1.0


def test_malformed_output_is_hard_penalized():
    assert compute_reward("the answer is 420", 420) == MALFORMED_PENALTY
    assert compute_reward("", 420) == MALFORMED_PENALTY
    assert compute_reward("0xzz", 420) == MALFORMED_PENALTY


def test_valid_but_wrong_still_beats_malformed():
    wrong_reward = compute_reward("0xfff", 420)
    assert wrong_reward > MALFORMED_PENALTY


def test_correctness_ceiling_exceeds_format_plus_validity():
    # A prediction that is maximally "close" without being exact must never
    # score at or above an actual exact match -- otherwise the policy could
    # learn to farm partial credit instead of aiming for the exact answer.
    near_miss_reward = compute_reward("0x1a3", 420)  # ground truth 420, prediction off by 1
    exact_reward = compute_reward("0x1a4", 420)
    assert near_miss_reward < exact_reward
    assert exact_reward - near_miss_reward >= 0.7 - 0.3  # ceiling gap holds


def test_numeric_closeness_ranks_near_miss_above_far_miss():
    near_miss = compute_reward("0x1a3", 420)   # 419 vs 420
    far_miss = compute_reward("0xfff", 420)    # 4095 vs 420
    assert near_miss > far_miss


def test_reward_range_bounds():
    ground_truth = 420
    samples = ["garbage", "0xfff", "0x1a3", "0x1a4", "0xzz", ""]
    for s in samples:
        r = compute_reward(s, ground_truth)
        assert -0.5 <= r <= 1.0
