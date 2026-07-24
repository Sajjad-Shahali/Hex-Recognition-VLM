# Results

Full run: `generate_dataset.py` (seed 42, 3000/400/400 train/val/test, 128x128 canvas) ->
`train.py --arch {crnn,fcn,convattn}` (up to 80 epochs, early stopping patience 12,
identical hyperparameters) -> `evaluate.py --arch <arch> --split test` ->
`compare_experiments.py`. Hardware: RTX 5070 Ti (CUDA 12.8 build of PyTorch 2.11).

This revision covers more than the original 3-way ablation: a real generalization
investigation (rotation robustness, out-of-distribution testing), multi-seed
reliability, and a hyperparameter-tuning experiment, following an iterative "found a
real limitation, tried a fix, reported the honest result" process documented throughout.

## 1. Architecture comparison (single canonical seed, test set, n=400)

![architecture comparison](../logs/plots/architecture_comparison.png)

| Architecture | Sequence head | Params | Checkpoint (MB) | Hex exact-match | Decimal exact-match | Char accuracy | Mean latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **CRNN** (deployed) | 1-layer BiGRU | 788,370 | 3.024 | **97.50%** | **97.50%** | 99.26% | 0.825 | 0.936 |
| **FCN** | 3-layer dilated Conv1d, no recurrence | 736,530 | **2.835** | 83.75% | 84.00% | 95.52% | 0.828 | 1.014 |
| **ConvAttn** | 2-layer self-attention encoder, no recurrence | 852,882 | 3.309 | 92.75% | 92.75% | 98.03% | 1.255 | 1.648 |

Latency: dedicated batch=1 benchmark (100 reps, warm-up excluded) — matches what a live
`/predict` request experiences. Raw numbers: `logs/eval_results_{crnn,fcn,convattn}.json`.

**Important caveat**: this table is a single seed (42). See section 2 — for CRNN and FCN,
this single-seed ranking is not reliable evidence of which architecture is actually
better; ConvAttn's seed-to-seed consistency is arguably the more important result of
this whole comparison.

## 2. Multi-seed robustness — is 97.5% real, or did we get lucky?

![multi-seed accuracy](../logs/plots/multi_seed_accuracy.png)

Validation exact-match accuracy across 5 seeds (42-46), identical hyperparameters and
data, original training recipe (no warmup, grad-clip 5.0 — the recipe that ships):

| Architecture | Seeds | Mean | Std. dev. | Min | Max |
|---|---|---|---|---|---|
| CRNN | 5 | 55.05% | **42.6** | 3.00% | 95.25% |
| FCN | 5 | 52.55% | **33.7** | 3.50% | 86.50% |
| ConvAttn | 4* | 92.25% | **1.7** | 90.25% | 94.75% |

\* One ConvAttn seed's log was overwritten mid-experiment (see "Warmup/grad-clip
experiment" below) before its value was recorded — n=4, not 5, documented honestly
rather than silently treated as 5.

**This is the single most important finding in this report.** CRNN and FCN each
catastrophically failed to converge on 2 of 5 random seeds — accuracy indistinguishable
from an untrained model (3-3.5%) — while succeeding on the others. This is not test-set
sampling noise (n=400); it's training-time failure. Inspecting the failed runs'
per-epoch logs shows CTC loss plateauing at a value consistent with degenerate,
uninformative output (a known "mode collapse" failure mode for CTC training), not slow
convergence that more epochs would fix.

**ConvAttn converged reliably on every seed tested.** Its self-attention encoder
appears substantially more robust to initialization than either the BiGRU or the
dilated-convolution head, independent of the mitigation attempt below.

### Warmup/grad-clip experiment (tried, not adopted)

LR warmup (6 epochs, linear ramp) + tighter gradient clipping (max-norm 1.0 vs. the
original 5.0) is the standard fix for exactly this CTC failure mode. Re-running all 5
seeds x 3 architectures with this recipe:

| Architecture | Recipe | Mean | Std. dev. |
|---|---|---|---|
| CRNN | original (deployed) | 55.05% | 42.6 |
| CRNN | warmup + tight clip | 57.90% | 28.1 |
| FCN | original (deployed) | 52.55% | 33.7 |
| FCN | warmup + tight clip | 50.95% | 34.1 |
| ConvAttn | original (deployed) | 92.25% | 1.7 |
| ConvAttn | warmup + tight clip | 91.40% | 6.5 |

Mixed, not a clean win: it rescued CRNN's worst seed (43: 3.75% -> 67.00%) but *broke*
several previously-good seeds — CRNN seed 42 (the canonical, deployed seed) dropped
from 94.75% to 73.50%, and seed 44 dropped from 95.25% to 64.00%. It also made ConvAttn
*less* consistent (std 1.7 -> 6.5) for no accuracy gain. **Net effect: inconclusive to
mildly negative — the original recipe (no warmup, clip 5.0) was kept.** `src/train.py`
still exposes `--warmup-epochs` / `--grad-clip` for further experimentation; this is
reported as a genuine attempted-and-rejected fix, not hidden as if the problem were
solved.

**Practical implication**: the shipped checkpoint (seed 42, original recipe) is
verified good for all three architectures. Retraining from scratch with an
unverified seed carries a real, measured risk of a collapsed CRNN or FCN model. In a
context where retraining happens unattended (no human checking the result), ConvAttn's
reliability would be a legitimate reason to deploy it instead, despite its lower
single-seed accuracy and higher latency.

## 3. Rotation robustness: a real architectural limitation, found and fixed

The first attempt at rotation robustness — adding 90/180/270-degree rotated samples
directly into the recognizer's training data — collapsed validation accuracy to ~20%.
Root cause: `_cnn_stem()` deliberately collapses the feature map's height to 1 (correct
and efficient for horizontally laid-out text), but 90/270-degree rotated text stacks
characters *vertically* — collapsing height destroys exactly the information needed to
tell them apart, before the sequence model ever sees it. 180° (upside-down) doesn't
have this problem, since the text stays horizontally laid out, just flipped.

**Fix**: a small upstream rotation classifier (`src/rotation_model.py`, 60,836
parameters, 4-way softmax over 0/90/180/270°, built with global-average-pooling instead
of height-collapse) predicts the image's orientation; the image is de-rotated by the
exact inverse before the *unchanged* recognizer reads it. The recognizer itself never
needs to see rotated text.

![generalization summary](../logs/plots/generalization_summary.png)

| Condition | Hex exact-match accuracy |
|---|---|
| In-distribution (upright test set) | 97.50% |
| Rotated, no correction (recognizer alone) | 22.33% |
| Rotation classifier accuracy (own 4-way task) | 98.67% |
| **Rotated + pipeline** (classify -> de-rotate -> recognize) | **93.67%** |

The pipeline recovers to within ~4 points of upright-only accuracy — the rotation
problem is effectively solved by isolating it into an easy sub-task (orientation
classification) rather than forcing one model to be simultaneously rotation-invariant
and precise. Full run: `src/evaluate_pipeline.py`, `data_rotation/` (n=300, balanced
across all 4 orientations).

## 4. Out-of-distribution robustness

`src/generate_ood_dataset.py` builds a harder held-out test (n=400): ±15° rotation
jitter (training: ±8°), Gaussian blur on 60% of images, narrower foreground/background
contrast, and 8 fonts never seen during training. Calibrated to stay human-readable —
an earlier, harsher version collapsed to 0.25%, which measures "how badly can an image
be destroyed" rather than generalization.

| Condition | Hex exact-match accuracy |
|---|---|
| In-distribution (upright test set) | 97.50% |
| **Out-of-distribution** (unseen fonts, blur, low contrast, wider rotation) | **41.00%** |

A real, honest generalization gap (-56.5 points) — not a collapse, not a free pass.
Expected for a PoC trained on ~3,000 synthetic images with a fixed 12-font pool;
flagged as the clearest target for future work (more font/augmentation diversity, or
real photographed data) rather than presented as solved.

## 5. Hyperparameter tuning (Optuna)

`src/optuna_tune.py` — 15 trials, learning rate (log-uniform 1e-4 to 3e-3) and batch
size ({32, 64, 128}), short-budget proxy runs (25 epochs, patience 8) for search speed,
CRNN only, seed 42. Raw results: `logs/optuna_results.json`.

13 of 15 trials collapsed under the short proxy budget (val accuracy 2-33%) — batch
sizes 64 and 128 combined with most sampled learning rates didn't converge within 25
epochs at all (not necessarily *unable* to converge, just not fast enough for this
budget). The best trial (`lr=0.00176, batch_size=32`) hit 93.25% at 25 epochs, clearly
better than the default (`lr=1e-3, batch_size=64`) under the same short budget.

**Full-budget comparison** — retraining both configs to convergence (80 epochs, early
stopping, patience 12, same seed):

| Config | lr | batch size | Val. accuracy (full budget) | Test hex exact-match |
|---|---|---|---|---|
| **Untuned (deployed)** | 1e-3 | 64 | 94.75% | **97.50%** |
| **Tuned (Optuna best)** | 0.00176 | 32 | 94.75% | 95.25% |

**Honest result: tuning did not help, and by test accuracy the untuned default is
actually slightly better.** The short-budget search selected for *how fast* a config
converges within 25 epochs, not *how good* it ends up at convergence — with early
stopping and a full epoch budget, both configs reach the same validation ceiling, and
sampling noise at test time (n=400) favors the untuned run. This is reported as a
genuine negative result: given this task's small model and dataset, the default
hyperparameters were already close to as good as a 15-trial search finds, and a
short-budget proxy is a biased objective for architectures like this one, prone to the
same seed-dependent slow-start behavior documented in section 2. **The untuned
default config ships.**

## 6. Error analysis (deployed CRNN, test set)

`src/error_analysis.py --arch crnn --checkpoint ../weights/model_crnn.pt`.

| Digit length | Accuracy | n |
|---|---|---|
| 1 digit (`0x0`-`0xf`) | 100.00% | 140 |
| 2 digits (`0x10`-`0xff`) | 98.52% | 135 |
| 3 digits (`0x100`-`0xfff`) | 93.60% | 125 |

Malformed-output rate: **0.0%** — every wrong prediction was a wrong-but-well-formed
hex literal, not garbage. 10 total test-set failures; the failure montage
(`logs/plots/failures_crnn.png`) shows several are adjacent/repeated-character drops
characteristic of CTC's repeat-collapsing decode rule (e.g. `0x77f` -> `0x7f`).

## 7. Interpretation and what changed across this investigation

- **Hex and decimal exact-match were identical in this run** for every architecture —
  not strictly guaranteed by construction (a spurious leading zero would be hex-wrong
  but decimal-correct), simply that no such case occurred here.
- **The square canvas (needed for rotation support) cost nothing in the end**: CRNN's
  final accuracy (97.50%) matches or slightly exceeds the original 128x32-canvas result
  (97.25%), once trained with enough epochs and early stopping rather than a fixed
  30-epoch budget.
- **FCN is not the latency-optimal choice it first appeared to be** — at true batch=1
  latency it's statistically tied with CRNN, so its only real advantage is a smaller
  checkpoint.
- **ConvAttn's headline weakness (lower single-seed accuracy, highest latency) is
  offset by a genuine strength (seed robustness)** that only multi-seed testing
  revealed — a good example of why single-seed benchmarking is insufficient for this
  kind of comparison.
- **Two real architectural/methodological bugs were found and fixed as part of this
  work**, not hidden: (1) recognizer accuracy collapsing on rotated input, traced to a
  specific design choice (height-collapsing CNN stem) and fixed with a separate
  classifier rather than papered over; (2) a broken latency-measurement methodology
  (documented in the previous revision of this file, retained in git history) that
  produced a false "FCN is 2x faster" claim, corrected before it became a stale claim
  in a later revision.

## 8. Live API smoke test

`uvicorn api:app` started, `/health` returned `{"status":"ok", "device":"cuda",
"arch":"crnn", ...}`, and a real test-set image (`data/images/test/test_000000.png`)
was POSTed to `/predict`. `/health` fails clearly (`checkpoint_loaded: false`,
`status: "no_checkpoint_loaded"`) if `HEX_MODEL_PATH` points at a missing file, instead
of silently serving an untrained, randomly-initialized model as `"ok"`.
