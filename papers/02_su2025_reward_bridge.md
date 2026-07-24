---
title: "Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains"
authors: Yi Su, et al.
year: 2025
venue: ArXiv
citations: 147
url: https://consensus.app/papers/details/34b00f100680508cabf95f91ebdce298/?utm_source=claude_code
found_via: Consensus search — "reinforcement learning verifiable rewards sparse binary reward exact match language model"
backs: docs/system_design.md §3.1 (why sparse binary reward alone is weak)
---

## Why this paper

Searched to ground the claim that binary correctness signals work for
domains with structured, checkable answers (like this task's hex→decimal
conversion), but need shaping to avoid flat-gradient problems near the
start of training.

## Abstract (as returned by Consensus)

Reinforcement learning with verifiable rewards (RLVR) has demonstrated
significant success in enhancing mathematical reasoning and coding
performance of large language models (LLMs), especially when structured
reference answers are accessible for verification. However, its extension
to broader, less structured domains remains unexplored. In this work, we
investigate the effectiveness and scalability of RLVR across diverse
real-world domains including medicine, chemistry, psychology, economics,
and education, where structured reference answers are typically
unavailable. We reveal that binary verification judgments on broad-domain
tasks exhibit high consistency across various LLMs provided expert-written
reference answers exist. Motivated by this finding, we utilize a generative
scoring technique that yields soft, model-based reward signals to overcome
limitations posed by binary verifications, especially in free-form,
unstructured answer scenarios. We further demonstrate the feasibility of
training cross-domain generative reward models using relatively small (7B)
LLMs without the need for extensive domain-specific annotation. Through
comprehensive experiments, our RLVR framework establishes clear performance
gains, significantly outperforming state-of-the-art open-source aligned
models such as Qwen2.5-72B and DeepSeek-R1-Distill-Qwen-32B across domains
in free-form settings.

## Claim used

RLVR (reinforcement learning with verifiable rewards) works well for
domains with structured, checkable answers like math and code — this
task's hex→decimal conversion is exactly that kind of domain, and the
paper's own move away from purely binary verification (toward soft,
shaped signals) mirrors the design doc's argument for the 3-tier reward
in §3.2.
