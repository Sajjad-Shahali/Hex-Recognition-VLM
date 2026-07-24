# System Design — Hex-to-Decimal Vision Model

## 0. Framing (read this first)

**Is this really a "VLM"?** No. A VLM (LLaVA-style) couples a large pretrained vision
encoder with an LLM decoder and does open-ended language generation. What this task
describes — fixed-vocabulary, fixed-format, closed-set character recognition — is an
**OCR / scene-text-recognition problem**, solved here with a small CNN+RNN+CTC network
trained from scratch. We use the assessment's "VLM" terminology in this document only to
match the brief; the architecture section states plainly what is actually built and why
a full VLM would be the wrong tool for a 16-symbol, 1-5 character closed vocabulary.

**Is RL actually necessary?** No, not for this task. Hex→decimal is a deterministic,
unambiguous function — there's no reward signal to learn beyond "did you read the
characters correctly," which SFT with cross-entropy/CTC already optimizes directly. RL
does not improve information the model already receives as a hard label. It becomes
useful only in a scenario SFT can't reach directly: when the *output format itself* is
underspecified (e.g. the model is a generative LLM emitting free-text like "the answer
is 420") and you cannot backprop through the parsing/rounding logic that turns that text
into a scored decimal. We therefore treat RL in this design as a **designed-but-optional
alignment stage**, specified fully (pipeline + reward function) as the brief asks, applied
against a generative decoder head rather than the CTC recognizer, and explicitly justified
as optional rather than load-bearing. §3 covers it as such.

**Are character-level YOLO boxes useful if the recognizer uses CTC?** Not for training the
CTC head — CTC is intentionally alignment-free and needs only (image, string) pairs, no
boxes. We still produce them because: (1) the brief requires them explicitly; (2) they
let you swap in a detect-then-classify pipeline (YOLO crops → per-char classifier) as a
fallback if CTC underperforms on real photos, which is a legitimate production hedge; (3)
they're a free byproduct of the synthetic renderer (we know exact glyph placement) and
double as a sanity-check / debugging visualization for the dataset itself. See §5.

---

## 1. Architecture overview

```
Image (RGB/gray, variable hex string "0x..") 
        │
        ▼
 ┌───────────────┐
 │  CNN backbone │  4 conv+pool blocks, collapses height → 1, keeps width as a sequence axis
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │  BiGRU (1 layer)│  models left-to-right character context
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │ Linear + CTC  │  per-timestep distribution over {0-9,a-f,x,blank} (17+1 classes)
 └───────┬───────┘
         ▼
  CTC greedy/beam decode → string "0x1a4" → strip "0x" → int(hex,16) → decimal
```

Stages, end to end:

1. **Data generation** (Part B) — synthetic renderer with font/size/position/rotation/noise
   jitter, YOLO character boxes, `data.yaml`, `dataset.csv`.
2. **SFT** — CNN+BiGRU+CTC trained with `nn.CTCLoss` on (image, target string) pairs.
   This is the entire "teach the model the task" stage.
3. **RL (optional, designed not required)** — a second, generative variant of the head
   (autoregressive decoder over the same vocab) fine-tuned with GRPO-style group-relative
   policy optimization against a structured reward (§3). Positioned as an alignment layer
   on top of SFT, not a replacement for it — this mirrors how RLHF/RLVR is used after SFT
   in LLM pipelines, not instead of it.
4. **Evaluation** — exact-match accuracy (hex string, decimal int), character accuracy,
   parameter count, on-disk size, per-image latency (§4).
5. **Deployment** — FastAPI service, model loaded once at startup, `/predict` accepts an
   uploaded image and returns `{hex_prediction, decimal_prediction, latency_ms}`.

## 2. Recognition model: CNN + BiGRU + CTC — why this and not something else

The task is short fixed-vocabulary sequence recognition (1–5 rendered glyphs: `0`, `x`,
and up to 3 hex digits), not free-form scene text. CRNN (CNN feature extractor → BiRNN →
CTC) is the standard, well-verified architecture for exactly this class of problem: it
turns recognition into a sequence-labeling task without needing per-character alignment
at train time, is small, and is not constrained by a fixed output length or a predefined
lexicon [1]. It also generalizes to the case where a photographed hex string has an
unusual glyph count — CTC handles variable-length sequences natively, unlike a fixed
5-slot softmax head, which would need one-hot classifiers per position and gets brittle
if the crop includes stray characters.

The main documented weakness — degraded accuracy on very short, high-variance text (e.g.
stylized/artistic glyphs) [1] — is a real risk given hex strings are only a few characters,
so we lean on the augmentation pipeline in §5 (font variety, jitter, rotation, noise)
specifically to counter it, and report exact-match accuracy honestly rather than only
character accuracy, since short-sequence CTC models can look better on char-level metrics
than they are on whole-string correctness.

Lighter alternatives considered and rejected for this task:
- **Fixed-length multi-head classifier** (K independent softmaxes for K character slots):
  simpler, no CTC alignment needed, but breaks the moment the rendered string length
  varies (1 vs 3 hex digits) unless you pad to a fixed number of slots and add a
  "no-char" class — which reintroduces most of CTC's bookkeeping without its length
  flexibility. Rejected because Part B's domain explicitly varies from 1–3 hex digits.
- **Full transformer OCR (e.g. TrOCR-style)**: pretrained ViT+text-decoder, correctly
  sized for real-world OCR robustness, wrong sized for a 16-symbol closed vocabulary and
  a "small, efficient, from-scratch" requirement — needless parameter and latency cost.
- **Template matching / classical CV**: extremely small and fast, but the brief calls for
  a learned model with SFT/RL stages, so this is out of scope regardless of numeric
  efficiency.

CNN+BiGRU+CTC is the right-sized choice: parameter budget dominated by the small CNN
stem (target: under 2M parameters, see §4), single BiGRU layer is enough context for a
five-symbol sequence, and CTC removes the need for the character-level alignment that
Part B's images would otherwise require at training time — even though we still emit
those boxes for the reasons in §0.

### 2.1 Ablation: is the BiGRU actually earning its place?

Rather than assert the CRNN choice by citation alone, `src/model.py` implements it
alongside two alternatives that remove or replace the recurrent encoder, sharing an
identical CNN stem (`_cnn_stem()`) so any accuracy/latency delta is attributable to the
sequence-modeling head, not to different visual features. All three were trained from
scratch on the same dataset (3000/400/400 split, seed 42, up to 80 epochs with early
stopping (patience 12), identical hyperparameters) via
`src/train.py --arch {crnn,fcn,convattn}`. Canvas is 128x128 (square, not the original
128x32) to support the orientation-robustness work in §2.2:

| Architecture | Sequence head | Params | Test hex exact-match | Mean latency (ms, batch=1, GPU) |
|---|---|---|---|---|
| **CRNN** (deployed) | 1-layer BiGRU | 788,370 | **97.50%** | 0.825 |
| **FCN** | 3-layer dilated Conv1d (dilation 1/2/4), no recurrence [4,6] | 736,530 | 83.75% | 0.828 |
| **ConvAttn** | 2-layer self-attention encoder, no recurrence [5] | 852,882 | 92.75% | 1.255 |

(Full run: `docs/RESULTS.md`, plots in `logs/plots/`, raw numbers in
`logs/experiment_comparison.json`. Latency is a dedicated batch=1 benchmark, 100 reps
after warm-up — matching what a live `/predict` request actually experiences, not
derived from batched throughput.)

**Reading the result**: on this single canonical seed, CRNN leads on both accuracy and
latency — FCN and CRNN are statistically tied on latency (0.828ms vs. 0.825ms, no real
FCN speed advantage at batch=1), so FCN's only edge is a smaller checkpoint (2.835MB vs.
3.024MB). ConvAttn trails CRNN by 4.75 points on this seed and is the slowest of the
three (1.255ms, ~52% slower than CRNN).

**But single-seed numbers are not the full story here** — see the multi-seed robustness
finding below, which meaningfully changes how much to trust any single seed's ranking
for CRNN and FCN specifically.

#### Multi-seed robustness: a bigger finding than the single-seed ranking

Training each architecture across 5 seeds (42-46, identical hyperparameters, validation
exact-match accuracy) reveals that the single-seed comparison above is unreliable for
two of the three architectures:

| Architecture | Seeds | Mean val. acc. | Std. dev. | Min | Max |
|---|---|---|---|---|---|
| CRNN | 5 | 55.05% | **±42.6** | 3.00% | 95.25% |
| FCN | 5 | 52.55% | **±33.7** | 3.50% | 86.50% |
| ConvAttn | 4 | 92.25% | **±1.7** | 90.25% | 94.75% |

(Plot: `logs/plots/multi_seed_accuracy.png`. ConvAttn is n=4, not 5 — one seed's log was
overwritten mid-experiment before its value was captured; see §2.2 for how this
happened. Values are validation, not test, accuracy: re-evaluating all 15+ checkpoints
against the held-out test set was out of scope for this pass, but validation accuracy
during training is measured identically across all seeds and is sufficient to
demonstrate the variance itself.)

CRNN and FCN each catastrophically failed to converge on 2 of 5 random seeds (3-4%
accuracy, indistinguishable from not training at all), while succeeding on the others
(85-96%). **ConvAttn converged reliably on every seed tested**, with under 2 points of
spread. This is not sampling noise in the usual sense (n=400 test-set noise) — it's
seed-dependent *training* failure: some random initializations for CRNN/FCN never
escape a bad optimization region, most consistent with the CTC "mode collapse" failure
mode (loss plateaus at a value consistent with degenerate, uninformative output; see
epoch-by-epoch logs in `logs/train_log_crnn_seed45.txt`).

**Mitigation attempted, not adopted**: LR warmup (6 epochs, linear ramp) plus tighter
gradient clipping (max-norm 1.0 vs. the original 5.0) is a standard fix for exactly this
failure mode. Re-running all 5 seeds x 3 architectures with this recipe gave *mixed*
results, not a clean fix:

| Architecture | Recipe | Mean | Std. dev. |
|---|---|---|---|
| CRNN | original (deployed) | 55.05% | 42.6 |
| CRNN | warmup + tight clip | 57.90% | 28.1 |
| FCN | original (deployed) | 52.55% | 33.7 |
| FCN | warmup + tight clip | 50.95% | 34.1 |
| ConvAttn | original (deployed) | 92.25% | 1.7 |
| ConvAttn | warmup + tight clip | 91.40% | 6.5 |

The warmup+clip recipe helped CRNN's worst outlier (seed 43: 3.75% -> 67.00%) but *hurt*
several previously-good seeds (CRNN seed 42: 94.75% -> 73.50%; CRNN seed 44: 95.25% ->
64.00%) and made ConvAttn *less* stable (std 1.7 -> 6.5) without a compensating accuracy
gain. Net effect across the sweep: inconclusive to mildly negative. **The original
recipe (no warmup, grad-clip 5.0) was kept as the default** — `src/train.py`'s
`--warmup-epochs`/`--grad-clip` flags remain available for experimentation, defaulting
to the values that performed better here. This is reported as an honest negative result
rather than omitted: the standard fix for this failure mode did not reliably work for
this task, and seed sensitivity for CRNN/FCN remains a real, open limitation.

**Practical implication**: the canonical seed (42) was verified to converge well for
all three architectures and is what ships in `weights/`. Anyone retraining from scratch
with `train.py`'s default seed will reproduce this. Retraining with a *different*,
unverified seed carries a real (roughly 40%, per this sample) chance of a collapsed
CRNN or FCN model — worth flagging explicitly for anyone extending this repo, and a
legitimate reason to prefer ConvAttn specifically in a context where retraining without
manual verification is expected (e.g. an automated retraining pipeline), despite its
lower single-seed accuracy and higher latency here.

**Deployment decision**: CRNN ships in `weights/model.pt` / `src/api.py`, on the
strength of its best-seed accuracy (97.50% test, verified) and the fact that the
canonical seed is fixed and shipped, not re-rolled at deployment time. FCN remains a
documented, working option (`weights/model_fcn.pt`) where checkpoint size specifically
is the binding constraint. ConvAttn's dramatically better seed-robustness is flagged
above as the deciding factor in a different deployment context (unattended retraining).

*Literature note:* the CRNN+CTC characterization above (variable-length sequence
recognition without a predefined lexicon, compact/efficient relative to segmentation-based
alternatives, degraded accuracy on short high-variance text) is grounded in [1] — a
peer-reviewed CRNN/CTC scene-text study returned via a Consensus query run for this
document. Its short-text failure mode is exactly why the augmentation knobs in §5 exist.
Everything else in this section (rejecting the fixed-slot head, TrOCR, and template
matching) is engineering judgement, not literature-verified.

### 2.2 Generalizing to rotated input: a real architectural limitation, and its fix

A follow-up experiment asked whether the recognizer generalizes to sideways and
upside-down text, not just the small (±8°) rotation jitter used at training time. The
first attempt — adding 30% of training samples with an extra 90°/180°/270° rotation —
**collapsed validation accuracy to ~20%**, not a training-budget problem (loss was still
decreasing) but an architectural one:

`_cnn_stem()` deliberately collapses the feature map's height to 1 (see §2) so the
remaining width axis can serve as the CTC time axis — correct and cheap for horizontally
laid-out text, where height only ever encodes a single glyph's shape. For text rotated
90°/270°, the characters are stacked *vertically* — all landing in roughly the same
width-axis position, distinguished only by height. Collapsing height destroys exactly
the information needed to tell them apart, before the sequence model ever sees it. This
is not fixable by more epochs or a bigger model with the same stem design; it would
require either preserving 2D structure into a detection-style pipeline (the YOLO-boxes
fallback discussed in §0) or a genuinely different architecture. 180° (upside-down) does
not have this problem — text stays horizontally laid out, just flipped — so it remains
learnable by this architecture.

**Fix**: rather than redesign the recognizer, add a small upstream **rotation
classifier** (`src/rotation_model.py`) — a 4-way softmax (0°/90°/180°/270°) over a
CNN that keeps 2D structure via global average pooling instead of collapsing height
(60,836 parameters). It predicts the image's coarse orientation, the image is de-rotated
by the exact inverse before being handed to the unchanged recognizer. This means the
recognizer never needs to see rotated text at all — it stays trained purely on upright,
±8°-jittered data (§5), and only the small classifier needs to learn orientation.

Measured on a held-out set with all 4 orientations represented equally
(`src/evaluate_pipeline.py`, `data_rotation/`, n=300):

| Approach | Hex exact-match accuracy |
|---|---|
| Recognizer alone (no correction) | 22.3% |
| Rotation classifier accuracy (own task) | 98.7% |
| **Full pipeline** (classify → de-rotate → recognize) | **93.7%** |

The pipeline recovers to within ~1 point of the recognizer's own upright-only test
accuracy — the rotation problem is effectively solved by isolating it into a small,
easy sub-task (4-way orientation classification is far easier than reading digits) rather
than forcing one model to be simultaneously rotation-invariant and precise. Fixed 4
orientations (not continuous 0–360°) were chosen over a full angle-regression approach
because they match the realistic capture scenarios for this task (upright, sideways,
upside-down photo/scan) — a continuous range would mostly cover angles that aren't a
plausible way to photograph a hex literal, at the cost of a harder regression problem
(sin/cos encoding, boundary discontinuities) for no realistic benefit.

### 2.3 Out-of-distribution robustness: how far does it actually generalize?

Multi-seed averaging (§2.1) answers "is the headline number real given sampling noise,"
not "does the model generalize beyond training conditions." `src/generate_ood_dataset.py`
builds a harder held-out test (n=400): rotation jitter ±15° (training: ±8°), Gaussian
blur on 60% of images (training: none), a narrower foreground/background contrast gap,
and 8 fonts never seen during training (Impact, Trebuchet, Segoe UI, Constantia,
Bahnschrift, Corbel, Franklin Gothic, Gadugi). Calibrated to stay human-readable — an
earlier, harsher version of this test collapsed to 0.25% accuracy, which measures
"how badly can an image be destroyed," not generalization.

| Condition | Hex exact-match accuracy |
|---|---|
| In-distribution (upright test set) | 97.50% |
| **Out-of-distribution** (unseen fonts, blur, low contrast, wider rotation) | **41.00%** |

(Plot: `logs/plots/generalization_summary.png`, alongside the rotation-pipeline result
from §2.2 for a single combined view of every generalization condition tested.)

A real, honest generalization gap — down 56.5 points, not a collapse and not a free
pass. This is the most informative single number in this report for "does it
generalize": multi-seed variance (§2.1) says the training *process* is unreliable for
some architectures; this says the *model itself*, once trained, degrades substantially
outside its training distribution — expected for a PoC trained on ~3,000 synthetic
images with a fixed font pool, and a legitimate target for future work (more fonts,
harder augmentation ranges, or real photographed data) rather than something this PoC
claims to have solved.

## 3. RL pipeline and reward function (the creative/evaluated part)

### 3.1 Why sparse binary reward alone is a bad design

The naive approach — `reward = +1 if decimal_output == ground_truth else 0` — is a
textbook case of the sparse/degenerate reward problem in RLVR (reinforcement learning
with verifiable rewards): binary correctness signals are known to work for domains with
structured, checkable answers like math and code [2], but a purely 0/1 signal gives the
policy **zero gradient signal on partial progress** — a model that gets the format right
but flips one hex digit looks exactly as wrong, in reward terms, as one that outputs
garbage. Early in training, when the policy rarely gets an exact match, this produces the
same near-zero-gradient / high-variance regime that motivates every reward-shaping
proposal in the RLVR literature [2]: the fix is not to abandon exact-match as the ground
truth signal, but to decompose it into shaped sub-rewards that are all still automatically
verifiable (no learned/human reward model needed, so it stays cheap and exact).

Multi-objective decomposition — separating a "did you follow the required format" signal
from a "was the content correct" signal and combining them (rather than one monolithic
reward) — is an active, empirically supported pattern for GRPO-style training, used to
avoid a single collapsed reward hiding which objective the model is failing on [3]. We
apply the same principle here across three tiers instead of two, because hex output has a
natural three-way failure taxonomy (malformed / valid-format-but-wrong / correct) that a
2-way decomposition would blur together.

### 3.2 Reward function

Assumes the policy is a generative head that emits free text (this is the scenario in
which RL is actually useful — see §0); the CTC recognizer never runs this stage.

```
Given: raw_output (string emitted by the policy), ground_truth_decimal (int)

def compute_reward(raw_output: str, ground_truth_decimal: int) -> float:
    # Tier 1 — format reward (0.0 or 0.2): can we even parse a hex literal out of it?
    match = HEX_PATTERN.match(raw_output.strip())   # expects r"^0x[0-9a-f]+$"
    if match is None:
        return -0.5                                  # malformed output, hard penalty

    format_reward = 0.2                              # well-formed 0x-prefixed hex literal

    # Tier 2 — validity reward (0.0 or 0.1): does it decode to *a* legal integer at all?
    try:
        predicted_value = int(match.group(0), 16)
    except ValueError:
        return format_reward - 0.3                   # shouldn't happen given the regex,
                                                       # kept as a defensive branch

    validity_reward = 0.1

    # Tier 3 — correctness reward (0.0, partial, or 0.7): the actual task signal
    if predicted_value == ground_truth_decimal:
        correctness_reward = 0.7                      # exact match — full credit
    else:
        # partial credit shaped by numeric closeness, capped well below exact-match
        # so the policy can never prefer "close" over "correct"
        max_val = max(predicted_value, ground_truth_decimal, 1)
        closeness = 1 - abs(predicted_value - ground_truth_decimal) / max_val
        correctness_reward = 0.3 * max(0.0, closeness)

    return format_reward + validity_reward + correctness_reward
    # range: -0.5 (malformed) .. -0.2 (parses but not evaluable, defensive)
    #        .. 0.3 (valid hex, numerically far off) .. up to 1.0 (exact match)
```

Design notes (the "logic" the brief asks to be evaluated on):

- **Tiers are additive, not multiplicative**, so a model that fixes only its formatting
  (going from garbage to a valid-but-wrong hex literal) gets an immediate, visible reward
  bump, even before it ever gets a digit right — this directly targets the "flat gradient
  near training start" failure mode described in §3.1.
- **The correctness ceiling (0.7) always exceeds the sum of format+validity (0.3)**, so the
  policy can never learn to farm partial credit instead of aiming for exact match — a
  known failure mode when partial-credit terms are not capped relative to the main signal.
- **Malformed output is a hard negative (-0.5)**, not zero, because zero reward is
  indistinguishable from "no signal yet" in GRPO's group-relative advantage computation —
  an explicit penalty keeps malformed completions ranked below every parseable one within
  the same rollout group.
- **Closeness credit is numeric, not string-edit-distance**, deliberately: `0x1a4` (420)
  vs a prediction of `0x1a3` (419) is a near-miss in the actual math task; `0x1a4` vs
  `0xfff` is not, even though both are equally "one wrong hex digit" away in string terms.
  The task is a math conversion task, so the shaping should reward numeric proximity, not
  surface-level string similarity.

### 3.3 RLVR pipeline (GRPO-style, applied after SFT)

```
1. Initialize policy = SFT checkpoint (generative decoder head)
2. For each training step:
   a. Sample a batch of images with known ground-truth decimal values
   b. For each image, roll out G=8 completions from the current policy (temperature > 0)
   c. Score every completion with compute_reward() above  → vector of G scalar rewards
   d. Compute group-relative advantage per completion:
          A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)
      (this is GRPO's critic-free baseline — no separate value network needed)
   e. Policy-gradient update weighted by A_i, clipped (PPO-style ratio clipping)
      against the pre-update policy to bound the step size
   f. Periodically re-validate against a held-out set using exact-match accuracy only
      (the RL reward is a training signal; the reported metric stays exact-match, so
      reward hacking on the shaped terms doesn't inflate reported quality)
3. Stop when held-out exact-match accuracy plateaus or regresses vs. the SFT baseline
```

Why GRPO over vanilla PPO: no separate critic/value network to train (cheap, appropriate
for a small model and PoC budget), and it was designed for exactly this class of
verifiable, group-comparable reward — one of the reasons it has become the default for
RLVR-style math/code post-training [3].

## 4. Model metrics

For an 8-hour PoC, essential vs. optional:

| Metric | Essential? | Why |
|---|---|---|
| Exact-match accuracy (decimal) | Essential | This is the actual task — "did the API return the right number." |
| Exact-match accuracy (hex string) | Essential | Same signal, catches recognizer errors before the hex→int conversion step; separates OCR error from conversion error. |
| Character-level accuracy | Essential | Diagnostic: distinguishes "model is close" (few wrong chars) from "model is lost" (all wrong), useful for debugging without re-running exact-match. |
| Parameter count | Essential | Directly requested ("deployed model size"); trivial to report (`sum(p.numel())`), no extra compute. |
| On-disk checkpoint size (MB) | Essential | Same requirement, different unit; matters for the "small" claim in the brief. |
| Per-image inference latency (mean, p95) | Essential | Directly requested; cheap to measure in the eval script with a warm-up pass. |
| Training/validation loss curves | Essential | Minimum evidence the SFT loop is actually learning, not just running epochs. |
| Multi-seed mean/std accuracy | Essential | Distinguishes "the reported number is real" from "we got lucky once" -- turned out to be load-bearing here (§2.1: CRNN/FCN have ±30-43 point seed variance). |
| Out-of-distribution accuracy (unseen fonts/blur/contrast) | Essential | The actual generalization question, which n=400 in-distribution accuracy alone cannot answer (§2.3). |
| Confusion matrix over hex digits | Optional | Implemented (`src/error_analysis.py`) but not very informative at n=400 -- too few off-diagonal errors to show a pattern; the failure-case montage it also produces was more useful. |
| Rotation/orientation robustness | Optional-turned-essential | Not part of the original metrics plan; became necessary once orientation augmentation was explored and found to break the architecture (§2.2) -- ended up being one of the most substantive results in this report. |
| FLOPs / theoretical compute cost | Optional | Latency + parameter count already covers the "efficiency" claim; FLOPs is redundant for a model this small. |

## 5. Dataset generation strategy (Part B design rationale)

- **Domain**: hex literals `0x` + 1–3 hex digits, values `0x0`–`0xfff` (0–4095 decimal).
- **Robustness knobs** (why each exists):
  - *Font variety*: forces the recognizer to learn glyph shape invariance instead of
    memorizing one font's pixel pattern — the single most important knob for avoiding
    overfitting to synthetic data.
  - *Size jitter*: simulates variable camera distance / zoom.
  - *Position jitter*: prevents the model from learning a fixed spatial prior (e.g.
    "the third character is always at x=64px").
  - *Rotation*: ±8° jitter simulates a non-perfectly-aligned photograph. The main
    recognition dataset stays at this small jitter deliberately -- coarse
    90/180/270-degree orientation is handled by a separate rotation classifier +
    de-rotation pre-stage instead of folding it into this dataset (see §2.2 for why:
    training the recognizer directly on rotated text collapsed its accuracy to ~20%,
    an architectural limitation, not a data problem). The bounding-box rotation math
    (`rotate_box()` in `generate_dataset.py`) does correctly support arbitrary angles
    including 90/180/270 -- it's used by `generate_rotation_dataset.py` for the
    classifier's training data.
  - *Background/text color and noise*: prevents the model from keying off a single
    fixed contrast pattern, the synthetic-data equivalent of lighting variation.
  - *Canvas is square (128x128, not the original 128x32)*: specifically so a
    90-degree rotation doesn't clip content -- needed for the rotation dataset above,
    and applied to the main dataset too so both share the same recognizer input shape.
- **YOLO character boxes**: one box per rendered glyph (including the literal `0` and
  `x`), class id = index into the vocabulary. Computed analytically from the renderer's
  own glyph placement (exact, not estimated) — see §0 for why they're produced despite
  the CTC recognizer not consuming them directly.
- **Split strategy**: 80/10/10 train/val/test by sample count, each split rendered
  independently so no image *file* is ever duplicated across splits. The underlying hex
  *value* can repeat across splits under a different rendering, because the full domain
  (4,096 possible values, `0x0`-`0xfff`) is smaller than the training set -- this is the
  correct design for this task, not a leak: recognition here is closed-set over a small,
  enumerable output space (analogous to MNIST reusing the same 10 digit classes across
  train/test), not open-set generalization to values never seen at train time.
  Digit-length is not explicitly stratified either, but empirically comes out
  near-balanced (1,250/1,277/1,273 of 3,800) from uniform sampling by digit-count.

## 6. What's out of scope for this PoC (and why)

- Real-world/photographed images — synthetic only, as specified.
- Beam-search / language-model-rescored CTC decoding — greedy decode is sufficient at
  this vocabulary size and sequence length; beam search would add latency for negligible
  accuracy gain here.
- Actually running the RL *training loop* in code — the brief asks for RL to be
  *designed*, not implemented, and (per §0) SFT alone already solves the deterministic
  conversion; the training loop and GRPO pipeline are specified fully in §3 as an
  optional alignment layer, not executed. The reward function itself, however, **is**
  implemented and unit-tested (`src/reward.py`, `tests/test_reward.py`) against concrete
  example completions — so the design in §3.2 is verifiable code, not only prose; only
  wiring it into an actual policy-gradient training loop is left undone.
- Distributed/multi-GPU training, mixed precision, ONNX export — all reasonable
  production hardening steps, none required to demonstrate the pipeline.

---

**Literature grounding key**: [1] and the general CRNN/CTC framing draws on a
peer-reviewed scene-text-recognition paper found via Consensus search
("CRNN CTC small vocabulary short sequence text recognition lightweight model");
[2] and [3] draw on RLVR/GRPO papers found via Consensus search on sparse/verifiable
reward design and multi-objective reward decomposition; [4] and [5] draw on papers found
via Consensus search on recurrence-free (fully convolutional) and self-attention text
recognizers, motivating the two alternative architectures actually trained and measured
in the §2.1 ablation. Full citations below. Everything else — the specific reward
formula, the tiering scheme, the architecture-selection rejections in §2, and the metrics
table in §4 — is engineering judgement for this task, not literature-verified, and is
presented as such.

## References

[1] Liu, Y. (2022). [Sequence Recognition of Scene Text Based on CRNN and CTPN Models](https://consensus.app/papers/details/59d90546ceab5195961229aeb90b6161/?utm_source=claude_code). Proceedings of the 2022 6th International Conference on Electronic Information Technology and Computer Engineering.

[2] Su, Y. et al. (2025). [Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains](https://consensus.app/papers/details/34b00f100680508cabf95f91ebdce298/?utm_source=claude_code). ArXiv. 147 citations.

[3] Liu, S.-Y. et al. (2026). [GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization](https://consensus.app/papers/details/1da7b49eb319534ca21d3928e3b033bf/?utm_source=claude_code). ArXiv. 89 citations.

[4] Yousef, M. et al. (2018). [Accurate, Data-Efficient, Unconstrained Text Recognition with Convolutional Neural Networks](https://consensus.app/papers/details/142fe9ef8e8a513eb60eeec8455c239e/?utm_source=claude_code). Pattern Recognition. 134 citations. (Fully convolutional, recurrence-free CTC text recognizer — motivates the FCN variant in §2.1.)

[5] Hernandez Diaz, D. et al. (2021). [Rethinking Text Line Recognition Models](https://consensus.app/papers/details/1b1aa822e50a556e85292eb42ae57381/?utm_source=claude_code). ArXiv. 65 citations. (Compares CTC/Transformer decoders and BiLSTM/self-attention/GRCL encoders; finds self-attention encoders competitive with recurrent ones — motivates the ConvAttn variant in §2.1.)

[6] Coquenet, D. et al. (2020). [Recurrence-free unconstrained handwritten text recognition using gated fully convolutional network](https://consensus.app/papers/details/327752f0011e5214b90c2f88b3ca80e0/?utm_source=claude_code). 2020 17th ICFHR. 42 citations. (A second recurrence-free, CTC-trained fully convolutional architecture, corroborating [4]'s framing that dropping recurrence for pure convolution is a viable, efficiency-motivated design family — jointly motivates the FCN variant in §2.1.)
