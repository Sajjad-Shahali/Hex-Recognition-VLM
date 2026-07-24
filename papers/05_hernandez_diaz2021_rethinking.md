---
title: Rethinking Text Line Recognition Models
authors: Daniel Hernandez Diaz, et al.
year: 2021
venue: ArXiv
citations: 65
url: https://consensus.app/papers/details/1b1aa822e50a556e85292eb42ae57381/?utm_source=claude_code
found_via: Consensus search — "fixed-length multi-head classification versus CTC sequence recognition short text"
backs: docs/system_design.md §2.1 (ConvAttn ablation variant)
---

## Why this paper

Searched to ground the self-attention-encoder alternative in the
architecture ablation — this paper's comparison of encoder families
(BiLSTM vs. self-attention vs. GRCL) directly motivated building
`HexConvAttn`.

## Abstract (as returned by Consensus)

In this paper, we study the problem of text line recognition. Unlike most
approaches targeting specific domains such as scene-text or handwritten
documents, we investigate the general problem of developing a universal
architecture that can extract text from any image, regardless of source or
input modality. We consider two decoder families (Connectionist Temporal
Classification and Transformer) and three encoder modules (Bidirectional
LSTMs, Self-Attention, and GRCLs), and conduct extensive experiments to
compare their accuracy and performance on widely used public datasets of
scene and handwritten text. We find that a combination that so far has
received little attention in the literature, namely a Self-Attention
encoder coupled with the CTC decoder, when compounded with an external
language model and trained on both public and internal data, outperforms
all the others in accuracy and computational complexity. Unlike the more
common Transformer-based models, this architecture can handle inputs of
arbitrary length, a requirement for universal line recognition.

## Claim used

Self-attention encoders are competitive with — and in some settings
outperform — recurrent (BiLSTM) encoders for line recognition, especially
when paired with a CTC decoder — grounds `HexConvAttn`'s design (2-layer
Transformer encoder + CTC) as a legitimate alternative to the BiGRU
baseline, not just an arbitrary architecture to include for variety.
