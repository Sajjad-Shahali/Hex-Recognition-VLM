---
title: Accurate, Data-Efficient, Unconstrained Text Recognition with Convolutional Neural Networks
authors: Mohamed Yousef, et al.
year: 2018
venue: Pattern Recognition
citations: 134
url: https://consensus.app/papers/details/142fe9ef8e8a513eb60eeec8455c239e/?utm_source=claude_code
found_via: Consensus search — "fully convolutional network without recurrent layer CTC text recognition efficiency latency"
backs: docs/system_design.md §2.1 (FCN ablation variant)
---

## Why this paper

Searched specifically to ground the fully-convolutional (no-recurrence)
architecture used as one of the two alternatives in the CRNN-vs-FCN-vs-
ConvAttn ablation.

## Abstract (as returned by Consensus)

Unconstrained text recognition is an important computer vision task,
featuring a wide variety of different sub-tasks, each with its own set of
challenges. One of the biggest promises of deep neural networks has been
the convergence and automation of feature extractors from input raw
signals, allowing for the highest possible performance with minimum
required domain knowledge. To this end, we propose a data-efficient,
end-to-end neural network model for generic, unconstrained text
recognition. In our proposed architecture we strive for simplicity and
efficiency without sacrificing recognition accuracy. Our proposed
architecture is a fully convolutional network without any recurrent
connections trained with the CTC loss function. Thus it operates on
arbitrary input sizes and produces strings of arbitrary length in a very
efficient and parallelizable manner. We show the generality and
superiority of our proposed text recognition architecture by achieving
state of the art results on seven public benchmark datasets, covering a
wide spectrum of text recognition tasks... Our proposed architecture has
won the ICFHR2018 Competition on Automated Text Recognition on a READ
Dataset.

## Claim used

A fully convolutional network trained with CTC and no recurrent
connections is a real, competitive architecture family for text
recognition, not a toy simplification — directly grounds `HexFCN` in
`src/model.py` (3-layer dilated Conv1d stack replacing the BiGRU).
