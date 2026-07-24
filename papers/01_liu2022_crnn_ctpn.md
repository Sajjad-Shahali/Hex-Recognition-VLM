---
title: Sequence Recognition of Scene Text Based on CRNN and CTPN Models
authors: Yiyi Liu
year: 2022
venue: Proceedings of the 2022 6th International Conference on Electronic Information Technology and Computer Engineering
citations: 5
url: https://consensus.app/papers/details/59d90546ceab5195961229aeb90b6161/?utm_source=claude_code
found_via: Consensus search — "CRNN CTC small vocabulary short sequence text recognition lightweight model"
backs: docs/system_design.md §2 (CRNN+CTC architecture choice)
---

## Why this paper

Original architecture-selection query for the deployed CRNN (CNN+BiGRU+CTC).
Used to justify choosing CRNN over a fixed-length classifier or a full
transformer-OCR model — and its documented short-text weakness directly
motivated the augmentation knobs in `generate_dataset.py`.

## Abstract (as returned by Consensus)

Image-based sequence recognition has lately emerged as a prominent study
subject in the science of computer vision, while text detection and
identification in natural situations has emerged as an active research
field. Based on scene text data, this paper addresses the theory of deep
learning-based CRNN and CTPN models and the process of processing text.
Using CRNN, text recognition can be turned into a time-dependent sequence
learning issue, which is commonly employed for indeterminate-length text
sequences. Contextual relationships between text images are learned using
BLSTM and CTC, thus effectively improving text recognition accuracy and
making the model more robust. It also excels in text recognition tests for
wordless and lexical-based scenes, as it is not constrained by any
predefined language. It produces a more efficient, but smaller, model that
is more suited to real-world settings. CRNN recognition accuracy is lower
for short texts with large morphological changes, such as artistic words,
or texts with large changes in natural scenes. Because of the Anchor
setting, CTPN can only detect horizontally distributed text, but a small
improvement can detect vertical text by adding horizontal Anchor. As a
result of the limitations of the framework, the irregularly inclined text
can be detected very broadly.

## Claim used

"Turns text recognition into a sequence-labeling task without needing
per-character alignment at train time... not constrained by a fixed output
length or a predefined lexicon" — grounds CRNN+CTC as the right choice for
variable-length (1-3 digit) hex recognition. The documented short-text
accuracy degradation is why `generate_dataset.py`'s augmentation knobs
(font/size/position/rotation/noise jitter) exist.
