---
title: "GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization"
authors: Shih-Yang Liu, et al.
year: 2026
venue: ArXiv
citations: 89
url: https://consensus.app/papers/details/1da7b49eb319534ca21d3928e3b033bf/?utm_source=claude_code
found_via: Consensus search — "reward shaping format correctness sub-reward correctness sub-reward GRPO PPO language model"
backs: docs/system_design.md §3.1, §3.3 (multi-objective reward decomposition, GRPO)
---

## Why this paper

Searched to ground the design decision to decompose the reward into
separate format/validity/correctness tiers rather than one monolithic
signal, and to justify GRPO over vanilla PPO for the designed RL pipeline.

## Abstract (as returned by Consensus)

As language models become increasingly capable, users expect them to
provide not only accurate responses but also behaviors aligned with
diverse human preferences across a variety of scenarios. To achieve this,
Reinforcement learning (RL) pipelines have begun incorporating multiple
rewards, each capturing a distinct preference, to guide models toward
these desired behaviors. However, recent work has defaulted to apply Group
Relative Policy Optimization (GRPO) under multi-reward setting without
examining its suitability. In this paper, we demonstrate that directly
applying GRPO to normalize distinct rollout reward combinations causes them
to collapse into identical advantage values, reducing the resolution of
the training signal and resulting in suboptimal convergence and, in some
cases, early training failure. We then introduce Group reward-Decoupled
Normalization Policy Optimization (GDPO), a new policy optimization method
to resolve these issues by decoupling the normalization of individual
rewards, more faithfully preserving their relative differences and
enabling more accurate multi-reward optimization, along with substantially
improved training stability.

## Claim used

Combining multiple reward objectives (format, validity, correctness) into
one signal without care can collapse into identical advantage values and
hurt training — motivates the design doc's explicit tiering scheme
(additive, not collapsed into one undifferentiated number) and its use of
GRPO's group-relative baseline as a critic-free approach for the designed
pipeline in §3.3.
