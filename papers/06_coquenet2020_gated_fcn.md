---
title: Recurrence-free unconstrained handwritten text recognition using gated fully convolutional network
authors: Denis Coquenet, et al.
year: 2020
venue: 2020 17th International Conference on Frontiers in Handwriting Recognition (ICFHR)
citations: 42
url: https://consensus.app/papers/details/327752f0011e5214b90c2f88b3ca80e0/?utm_source=claude_code
found_via: Consensus search — "fully convolutional network without recurrent layer CTC text recognition efficiency latency" (same query as paper 4)
backs: docs/system_design.md §2.1 (corroborates FCN ablation variant)
---

## Why this paper

Second recurrence-free architecture paper returned by the same search that
found paper 4 (Yousef et al. 2018). Added late — `src/model.py`'s
`HexFCN` docstring cited this paper from the start, but it wasn't added to
`docs/system_design.md`'s formal reference list until an independent
review caught the gap (see `docs/PROJECT_REPORT.md` §7).

## Abstract (as returned by Consensus)

Unconstrained handwritten text recognition is a major step in most
document analysis tasks. This is generally processed by deep recurrent
neural networks and more specifically with the use of Long Short-Term
Memory cells. The main drawbacks of these components are the large number
of parameters involved and their sequential execution during training and
prediction. One alternative solution to using LSTM cells is to compensate
the long time memory loss with an heavy use of convolutional layers whose
operations can be executed in parallel and which imply fewer parameters.
In this paper we present a Gated Fully Convolutional Network architecture
that is a recurrence-free alternative to the well-known CNN+LSTM
architectures. Our model is trained with the CTC loss and shows
competitive results on both the RIMES and IAM datasets.

## Claim used

A second, independent confirmation that recurrence-free, CTC-trained fully
convolutional architectures are a viable, competitive design family (not
just one paper's idiosyncratic result) — corroborates paper 4's framing
behind `HexFCN`.
