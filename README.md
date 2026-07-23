# Hex-to-Decimal Recognizer — Technical Assessment Submission

Full lifecycle PoC: synthetic dataset -> CNN+BiGRU+CTC model -> SFT training ->
evaluation -> FastAPI deployment. RL is designed (not coded) per the brief —
see `docs/system_design.md`. Includes a 3-way architecture ablation (BiGRU vs.
fully-convolutional vs. self-attention sequence heads) — see `docs/RESULTS.md`.

## Repo layout

```
docs/system_design.md    Part A design doc (architecture, ablation, RL pipeline/reward, metrics)
docs/PROJECT_REPORT.md   end-to-end project report (start here), plots + citations
docs/RESULTS.md          full ablation write-up with plots
docs/report/report.pdf   LaTeX-typeset version of the project report (report.tex source included)
src/
  common.py              shared vocab/constants/CTC decode helper
  generate_dataset.py    Part B synthetic dataset generator
  dataset.py             PyTorch Dataset for the generated data
  model.py               Part C 3 model architectures (crnn/fcn/convattn), shared CNN stem
  train.py               Part C SFT training loop, --arch selects architecture
  evaluate.py            Part C evaluation (accuracy, size, batch=1 latency), --arch/--checkpoint
  compare_experiments.py aggregates per-arch eval results into a comparison table + plot
  plot_results.py        renders per-arch training curves + cross-arch comparison plots
  error_analysis.py      per-digit-length accuracy, confusion matrix, failure-case montage
  reward.py               RL reward function (§3.2 of the design doc) as executable code
  api.py                 Part C FastAPI deployment (serves the CRNN, the deployed choice)
tests/
  test_reward.py         unit tests for reward.py (tiering, ceiling, malformed-output penalty)
data/                    generated dataset (images, YOLO labels, data.yaml, dataset.csv)
weights/
  model.pt               deployed checkpoint (CRNN, best val exact-match), loaded by api.py
  model_crnn.pt, model_fcn.pt, model_convattn.pt   best checkpoint per architecture
logs/
  train_log_<arch>.txt          per-arch training log (per-epoch loss/accuracy)
  eval_results_<arch>.json      per-arch test-set metrics (accuracy, size, batch=1 latency)
  error_analysis_<arch>.json    per-digit-length accuracy + malformed-output rate
  experiment_comparison.json    aggregated ablation numbers
  plots/                        training curves, test metrics, confusion matrix, failure montage
```

## Environment

Developed against Python 3.12 + PyTorch 2.11 with a CUDA 12.8 build (matched
to the dev machine's RTX 5070 Ti). `requirements.txt` pins `torch==2.11.0`.
Everything in `src/` auto-detects CUDA (`torch.cuda.is_available()`) and
falls back to CPU, so no code changes are needed either way — only the
install step differs:

```
python -m venv .venv
.venv\Scripts\activate            # Windows

# GPU (CUDA 12.8, matches the dev machine) -- two steps, torch has its own index:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install fastapi "uvicorn[standard]" python-multipart numpy pandas pillow PyYAML matplotlib pytest

# CPU-only -- same two steps, different torch index, no version pin needed:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi "uvicorn[standard]" python-multipart numpy pandas pillow PyYAML matplotlib pytest
```

(`requirements.txt` is kept as the authoritative version list / for `pip-compile`-style
tooling, but installing it directly with a single `pip install -r requirements.txt` pulls
torch from the default PyPI index, which does not host GPU wheels -- hence the two-step
install above for either target.)

Note: `src/` uses flat imports (`from common import ...`) rather than a package
layout, so run everything from inside `src/` (`cd src` first) -- `python -m
src.evaluate` or `uvicorn src.api:app` from the repo root will not resolve
those imports.

## Reproducing the pipeline

```
cd src

# 1. Generate the dataset (3000 train / 400 val / 400 test by default)
python generate_dataset.py

# 2. Train -- runs the 3-way ablation (SFT only, ~30s each on an RTX 5070 Ti / 30 epochs)
python train.py --arch crnn --epochs 30       # deployed architecture
python train.py --arch fcn --epochs 30        # fully-convolutional alternative
python train.py --arch convattn --epochs 30   # self-attention alternative

# 3. Evaluate each on the held-out test split
python evaluate.py --arch crnn --checkpoint ../weights/model_crnn.pt --split test
python evaluate.py --arch fcn --checkpoint ../weights/model_fcn.pt --split test
python evaluate.py --arch convattn --checkpoint ../weights/model_convattn.pt --split test

# 4. Aggregate into a comparison table + plot, and render per-arch training curves
python compare_experiments.py
python plot_results.py

# 5. (optional) error analysis on the deployed model: per-digit-length accuracy,
#    confusion matrix, failure-case montage
python error_analysis.py --arch crnn --checkpoint ../weights/model_crnn.pt --split test

# 6. Serve the deployed model (CRNN)
uvicorn api:app --host 0.0.0.0 --port 8000
```

Run the reward-function unit tests (the RL design in `docs/system_design.md` §3.2,
implemented as real code in `src/reward.py`):

```
cd tests
pytest -v
```

Only need the deployed model? Skip straight to `python train.py --arch crnn` and
`uvicorn api:app` -- the fcn/convattn runs are the ablation study, not required to
stand up the API.

Then test it:

```
curl -F "file=@../data/images/test/test_000000.png" http://localhost:8000/predict
curl http://localhost:8000/health
```

Reviewers can POST any image to `/predict` — see "Assumptions" below for how
non-matching-size images are handled.

## Results (deployed CRNN checkpoint; full 3-way ablation in `docs/RESULTS.md`)

| Metric | Value |
|---|---|
| Test hex exact-match accuracy | 97.25% |
| Test decimal exact-match accuracy | 97.25% |
| Mean character accuracy | 99.17% |
| Parameters | 542,226 |
| Checkpoint size | 2.08 MB |
| Mean inference latency (batch=1, GPU) | 0.71 ms |
| p95 inference latency (batch=1, GPU) | 0.82 ms |

Two alternative architectures (fully-convolutional, self-attention) were also
trained and evaluated on identical data for a controlled ablation. ConvAttn ties
CRNN's accuracy at higher latency and more parameters; FCN trades 6.5 accuracy
points for a smaller checkpoint (not for lower latency — see `docs/RESULTS.md`
for a measurement-methodology correction). Full comparison, convergence plots,
error analysis, and the deployment-choice reasoning: `docs/RESULTS.md`.

## Engineering assumptions (read before grading)

1. **"VLM" terminology**: the brief calls this a Vision-Language Model; architecturally
   it's a small CRNN (CNN+BiGRU+CTC) OCR model, the right-sized tool for a 16-symbol
   closed-vocabulary recognition task. See `docs/system_design.md` §0 for the full
   justification — flagged here so it doesn't read as a misunderstanding of VLM.
2. **RL is designed, not implemented in code.** Hex->decimal is a deterministic
   conversion; SFT with CTC already optimizes the actual task end to end. RL is specified
   as an optional alignment layer for a hypothetical generative-decoder variant (§3 of
   the design doc), matching the brief's request to design, not necessarily build, RL.
3. **Rendered text includes the literal `0x` prefix** (e.g. the image literally shows
   "0x1a4"), and the CTC vocabulary includes `x` as a class. This lets YOLO character
   boxes cover every rendered glyph (matching Part B's literal requirement) and keeps
   one uniform decode path from pixels to the full hex string.
4. **Fixed 128x32 canvas.** All generated images (and anything sent to `/predict`) are
   resized to exactly 128x32 grayscale. This keeps the model and preprocessing simple
   for a PoC; a production version would pad-to-aspect-ratio rather than stretch.
   Reviewer images of arbitrary size/aspect will still be accepted and resized, but
   extreme aspect ratios will distort the glyphs and hurt accuracy — a known, documented
   PoC limitation, not a crash risk.
5. **Value range**: hex values are 1-3 digits, `0x0`-`0xfff` (0-4095 decimal), sampled
   uniformly by digit-count then by value, per `generate_dataset.py`.
6. **Dataset size (3000/400/400)** and **epoch count (30)** were chosen to comfortably
   fit the "PoC, not production, limit epochs if needed" guidance — training converges
   to ~98% validation accuracy in under 30 seconds on the dev GPU; CPU-only users should
   expect it to still finish in a few minutes at this scale.
7. **CTC greedy decoding** (not beam search) — sufficient at this vocabulary size/sequence
   length; beam search would add latency for negligible accuracy gain here.
8. **Split leakage**: no image *file* is ever duplicated across splits (each split is
   rendered independently). The underlying hex *value* can repeat across splits with a
   different rendering, though -- the full output domain is only 4,096 possible values
   (`0x0`-`0xfff`), smaller than the 3,000-image training set, so this is expected and
   correct for this task: it's closed-set recognition over a small, enumerable output
   space (like MNIST's 10 digit classes), not open-set generalization to unseen values.
   An earlier version of this note incorrectly implied values never repeat across
   splits -- corrected after independent review flagged it.
9. **`dataset.csv` contains all three splits** (not just train) with `image_name` unique
   across splits by construction (`{split}_{index:06d}.png`), so a single CSV can be
   filtered by prefix or cross-referenced against `data/images/{split}/`.
10. **Model selection during training** keeps the checkpoint with the best validation
    exact-match accuracy (not final-epoch weights), written every epoch it improves.
11. **Three architectures, one deployed.** `train.py --arch {crnn,fcn,convattn}` trains
    any of three sequence-modeling heads on an identical CNN stem; `weights/model.pt`
    (loaded by `api.py`) is always the CRNN, chosen per the trade-off analysis in
    `docs/system_design.md` §2.1 / `docs/RESULTS.md`. The other two checkpoints are kept
    as a documented ablation, not dead code — `evaluate.py --arch <arch>` runs any of them.

## What's committed vs. what's regeneratable

- Committed: all `src/*.py`, `docs/`, `README.md`, `requirements.txt`,
  `weights/model.pt` + `weights/model_{crnn,fcn,convattn}.pt`,
  `logs/train_log_{crnn,fcn,convattn}.txt`, `logs/eval_results_{crnn,fcn,convattn}.json`,
  `logs/experiment_comparison.json`, `logs/plots/*.png`, `data/dataset.csv`, and a full
  copy of `data/images/` + `data/labels/` + `data.yaml` (small enough at this sample
  count — ~3800 tiny 128x32 PNGs — to keep in the repo for full reproducibility).
- Regeneratable at will: everything under `data/` and `weights/`, by re-running
  `generate_dataset.py` (with `--seed 42` for the exact same dataset) and `train.py`.

## Known limitations / what I'd sanity-check further given more time

- Synthetic-only training data; no real-world/photographed hex images tested.
- No stratification by digit-length in the train/val/test split (documented as an
  acceptable simplification in `docs/system_design.md` §5, given the small 1-3 digit space).
- The `/predict` endpoint's resize-to-fixed-canvas preprocessing will distort images
  with very different aspect ratios from the training data (wide vs. tall crops).
- RL training loop (`docs/system_design.md` §3.3) is specified but intentionally not
  implemented in code, per the brief's "no code needed" for Part A. The reward function
  itself (§3.2) *is* implemented and tested — `src/reward.py`, `tests/test_reward.py`.
- The character-level confusion matrix in `error_analysis.py` is diagonal-dominant and
  not very informative at n=400 (too few off-diagonal errors to show a pattern) — the
  failure-case montage it also produces is the more useful artifact at this sample size.
- `src/` uses flat imports and assumes `cd src` as the working directory; it isn't
  packaged as an installable module (`pip install -e .`), which would be the natural
  next step for a less PoC-shaped version of this repo.

## Independent review

This repo was reviewed by two independent reviewers after the initial build — a
manual read-through, and a second AI reviewer (OpenAI Codex, run in parallel with no
visibility into the first review) — then reconciled. Both found real, overlapping
issues: `api.py` hardcoded one architecture regardless of which checkpoint was loaded,
`/health` reported healthy status even with no checkpoint present, the latency
benchmark was measured incorrectly (producing a false "FCN is ~2x faster" claim), and
some documentation overstated or misdescribed what the code actually does. All are
fixed; the full list with before/after is in `docs/PROJECT_REPORT.md` §7 and
`docs/report/report.pdf` §9 — kept visible rather than silently corrected, since the
brief explicitly grades "the ability to spot and fix the inevitable bugs the AI creates."
