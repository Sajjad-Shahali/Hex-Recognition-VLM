# Hex-to-Decimal Recognizer — Technical Assessment Submission

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI-2E3440?logo=gunicorn&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-0080FF?logo=optuna&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)
![LaTeX](https://img.shields.io/badge/Report-LaTeX-008080?logo=latex&logoColor=white)

Full lifecycle PoC: synthetic dataset -> CNN+self-attention+CTC model -> SFT training ->
evaluation -> FastAPI deployment. RL is designed (not coded) per the brief —
see `report/report.tex` §8. Includes a 3-way architecture ablation (BiGRU vs.
fully-convolutional vs. self-attention sequence heads), a rotation-robustness
pipeline (a separate orientation classifier fixes a real architectural
limitation of the recognizer), an out-of-distribution robustness test,
multi-seed reliability testing, and a light hyperparameter search — see
`report/report.tex` §7 for the full generalization investigation.

## Repo layout

```
report/report.tex       full project report (design, ablation, generalization, RL, review)
report/report.pdf       typeset PDF of the same
papers/                 citation library backing the report (6 papers, 1 file each)
src/
  common.py              shared vocab/constants/CTC decode helper (128x32 recognizer canvas)
  generate_dataset.py    Part B synthetic dataset generator (upright, +-8deg jitter)
  generate_rotation_dataset.py   oriented dataset (0/90/180/270deg) for the rotation classifier
  generate_ood_dataset.py        harder held-out set: unseen fonts, blur, low contrast, wider jitter
  dataset.py / rotation_dataset.py   PyTorch Datasets
  model.py               Part C 3 recognizer architectures (crnn/fcn/convattn), shared CNN stem
  rotation_model.py      4-way orientation classifier (0/90/180/270deg), 60,836 params
  pipeline.py            two-stage inference: classify orientation -> de-rotate -> recognize
  train.py               Part C SFT training loop, --arch/--seed/--warmup-epochs/--grad-clip
  train_rotation_classifier.py   training loop for the orientation classifier
  evaluate.py            Part C evaluation (accuracy, size, batch=1 latency), --arch/--checkpoint
  evaluate_pipeline.py   end-to-end rotation-pipeline evaluation (recognizer-alone vs. pipeline)
  aggregate_seeds.py     multi-seed accuracy aggregation (mean/std across seeds)
  optuna_tune.py         light LR/batch-size search for CRNN (15 trials)
  compare_experiments.py aggregates per-arch eval results into a comparison table + plot
  plot_results.py        renders per-arch training curves + cross-arch comparison plots
  error_analysis.py      per-digit-length accuracy, confusion matrix, failure-case montage
  reward.py              RL reward function (§8.2 of the report) as executable code
  api.py                 Part C FastAPI deployment (arch read from the checkpoint, not hardcoded)
tests/
  test_reward.py         unit tests for reward.py (tiering, ceiling, malformed-output penalty)
data/                    main recognition dataset (upright only; images, YOLO labels, data.yaml, dataset.csv)
data_rotation/           oriented dataset for the rotation classifier (rotation_labels.csv)
data_ood/                out-of-distribution robustness test set (dataset.csv)
weights/
  model.pt               deployed checkpoint (ConvAttn, seed 45), loaded by api.py
  model_crnn.pt, model_fcn.pt, model_convattn.pt   best checkpoint per architecture
  model_<arch>_seed<seed>.pt   5-seed sweep checkpoints (seeds 42-46), backing the reliability claim
  rotation_classifier.pt orientation classifier checkpoint
logs/
  train_log_<arch>.txt          per-arch training log (per-epoch loss/accuracy)
  eval_results_<arch>.json      per-arch test-set metrics (accuracy, size, batch=1 latency)
  eval_results_ood_convattn.json  OOD test-set metrics for the deployed checkpoint
  pipeline_eval.json            rotation-pipeline end-to-end results
  error_analysis_<arch>.json    per-digit-length accuracy + malformed-output rate
  experiment_comparison.json    aggregated ablation numbers
  optuna_results.json           hyperparameter search results
  seed_results/                 per-seed test-set eval JSONs backing the mean+-std plots
  plots/                        training curves, test metrics, confusion matrix, failure montage,
                                 multi-seed accuracy, per-seed reliability lines
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
pip install fastapi "uvicorn[standard]" python-multipart numpy pandas pillow PyYAML matplotlib pytest optuna

# CPU-only -- same two steps, different torch index, no version pin needed:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi "uvicorn[standard]" python-multipart numpy pandas pillow PyYAML matplotlib pytest optuna
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

# 1. Generate the main (upright) dataset (3000 train / 400 val / 400 test)
python generate_dataset.py

# 2. Train -- runs the 3-way ablation (SFT only, up to 80 epochs, early stopping)
python train.py --arch crnn       # recurrent baseline
python train.py --arch fcn        # fully-convolutional alternative
python train.py --arch convattn   # self-attention alternative -- deployed architecture

# 3. Evaluate each on the held-out test split
python evaluate.py --arch crnn --checkpoint ../weights/model_crnn.pt --split test
python evaluate.py --arch fcn --checkpoint ../weights/model_fcn.pt --split test
python evaluate.py --arch convattn --checkpoint ../weights/model_convattn.pt --split test

# 4. Aggregate into a comparison table + plot, and render per-arch training curves
python compare_experiments.py
python plot_results.py

# 5. (optional) error analysis on the deployed model: per-digit-length accuracy,
#    confusion matrix, failure-case montage
python error_analysis.py --arch convattn --checkpoint ../weights/model_convattn.pt --split test

# 6. Rotation robustness: generate the oriented dataset, train the classifier,
#    evaluate the two-stage pipeline end to end (uses weights/model.pt, i.e. ConvAttn)
python generate_rotation_dataset.py
python train_rotation_classifier.py
python evaluate_pipeline.py

# 7. Out-of-distribution robustness test (unseen fonts, blur, low contrast)
python generate_ood_dataset.py
python evaluate.py --arch convattn --checkpoint ../weights/model.pt --data-dir ../data_ood --split test --out-json ../logs/eval_results_ood_convattn.json

# 8. Multi-seed reliability (repeat step 2 with --seed 43/44/45/46, then):
python aggregate_seeds.py --seeds 42 43 44 45 46

# 9. (optional) light hyperparameter search for CRNN
python optuna_tune.py --n-trials 15

# 10. Serve the deployed model (ConvAttn, with orientation correction available via pipeline.py)
uvicorn api:app --host 0.0.0.0 --port 8000
```

Run the reward-function unit tests (the RL design in `report/report.tex` §8.2,
implemented as real code in `src/reward.py`):

```
cd tests
pytest -v
```

Only need the deployed model? Skip straight to `python train.py --arch convattn` and
`uvicorn api:app` -- the crnn/fcn runs are the ablation study, not required to
stand up the API.

Then test it:

```
curl -F "file=@../data/images/test/test_000000.png" http://localhost:8000/predict
curl http://localhost:8000/health
```

Reviewers can POST any image to `/predict` — see "Assumptions" below for how
non-matching-size images are handled.

## Results (deployed ConvAttn checkpoint, seed 45; full 3-way ablation in `report/report.tex` §6)

| Metric | Value |
|---|---|
| Test hex exact-match accuracy | 96.50% |
| Test decimal exact-match accuracy | 96.50% |
| Mean character accuracy | 99.11% |
| Parameters | 606,738 |
| Checkpoint size | 2.37 MB |
| Mean inference latency (batch=1, GPU) | 1.00 ms |
| p95 inference latency (batch=1, GPU) | 1.12 ms |
| Rotated input, with de-rotation pipeline | 80.00% (vs. 6.00% with no correction) |
| Out-of-distribution (unseen fonts/blur/contrast) | 46.75% |

Two alternative architectures (recurrent, fully-convolutional) were also trained
and evaluated on identical data. Single-seed, CRNN can edge out ConvAttn by a couple
points on a lucky seed; **but across 5 random seeds (mean test accuracy, not just
validation), ConvAttn stays reliable every time (94.25% mean, std ~2 points) while
CRNN swings 35-95% and FCN swings 2.5-90% depending on the seed (std 23-43 points)**
— CRNN can hit the single highest peak of the three, but only ConvAttn is reliable on
average, which is why it's the deployed checkpoint: a submission that has to work
regardless of which seed it happened to train with should ship the reliable
architecture, not the lucky one. Full comparison, multi-seed data (including a
root-cause investigation that substantially improved CRNN's reliability), the
rotation-robustness pipeline, the OOD test, and the hyperparameter-tuning experiment:
`report/report.tex` §7.

## Engineering assumptions (read before grading)

1. **"VLM" terminology**: the brief calls this a Vision-Language Model; architecturally
   it's a small CNN+self-attention-encoder+CTC OCR model, the right-sized tool for a
   16-symbol closed-vocabulary recognition task. See `report/report.tex` §1 for the full
   justification — flagged here so it doesn't read as a misunderstanding of VLM.
2. **RL is designed, not implemented in code.** Hex->decimal is a deterministic
   conversion; SFT with CTC already optimizes the actual task end to end. RL is specified
   as an optional alignment layer for a hypothetical generative-decoder variant (§8 of
   the report), matching the brief's request to design, not necessarily build, RL.
3. **Rendered text includes the literal `0x` prefix** (e.g. the image literally shows
   "0x1a4"), and the CTC vocabulary includes `x` as a class. This lets YOLO character
   boxes cover every rendered glyph (matching Part B's literal requirement) and keeps
   one uniform decode path from pixels to the full hex string.
4. **Fixed 128x32 canvas for the recognizer** (a separate, square 128x128 canvas is used
   only by the rotation classifier's pre-stage — see assumption 12). All generated
   recognizer images (and anything sent to `/predict`) are resized to exactly 128x32
   grayscale. Reviewer images of arbitrary size/aspect will still be accepted and
   resized, but extreme aspect ratios will distort the glyphs and hurt accuracy -- a
   known, documented PoC limitation, not a crash risk.
5. **Value range**: hex values are 1-3 digits, `0x0`-`0xfff` (0-4095 decimal), sampled
   uniformly by digit-count then by value, per `generate_dataset.py`.
6. **Dataset size (3000/400/400)** was chosen to comfortably fit the "PoC, not
   production" guidance. **Epoch budget is up to 80 with early stopping** (patience 12
   epochs without validation improvement) rather than a fixed count, and early stopping
   means most runs finish well before 80 in practice. CPU-only users should expect
   proportionally longer runs but the same early-stopping behavior.
7. **CTC greedy decoding** (not beam search) — sufficient at this vocabulary size/sequence
   length; beam search would add latency for negligible accuracy gain here.
8. **Split leakage**: no image *file* is ever duplicated across splits (each split is
   rendered independently). The underlying hex *value* can repeat across splits with a
   different rendering, though -- the full output domain is only 4,096 possible values
   (`0x0`-`0xfff`), smaller than the 3,000-image training set, so this is expected and
   correct for this task: it's closed-set recognition over a small, enumerable output
   space (like MNIST's 10 digit classes), not open-set generalization to unseen values.
9. **`dataset.csv` contains all three splits** (not just train) with `image_name` unique
   across splits by construction (`{split}_{index:06d}.png`), so a single CSV can be
   filtered by prefix or cross-referenced against `data/images/{split}/`.
10. **Model selection during training** keeps the checkpoint with the best validation
    exact-match accuracy (not final-epoch weights), written every epoch it improves.
11. **Three architectures, one deployed.** `train.py --arch {crnn,fcn,convattn}` trains
    any of three sequence-modeling heads on an identical CNN stem; `weights/model.pt`
    (loaded by `api.py`) is ConvAttn (seed 45, the best-performing seed of the most
    reliable architecture across the 5-seed sweep) — `api.py` reads `arch` from the
    checkpoint itself, so swapping the deployed architecture is a file copy, not a code
    change. See the trade-off analysis in `report/report.tex` §5 / §7.1. The other
    checkpoints are kept as a documented ablation, not dead code — `evaluate.py --arch
    <arch>` runs any of them.
12. **Rotation robustness is a separate pre-stage, not baked into the recognizer.**
    `src/rotation_model.py` classifies orientation (0/90/180/270deg); `src/pipeline.py`
    de-rotates before handing the image to the unchanged recognizer. This follows from
    a real finding: training the recognizer directly on rotated text collapsed its
    accuracy to single digits (an architectural limitation of the height-collapsing CNN
    stem, `report/report.tex` §7.2), not a data or tuning problem. `api.py` itself still
    serves the recognizer alone (assumes upright input, matching the brief's task); the
    rotation-aware `pipeline.py` is available for anyone extending the API to accept
    arbitrarily rotated images.
13. **Seed sensitivity is real and disclosed, not hidden.** CRNN/FCN each fail to
    converge on roughly 40% of random seeds tested (accuracy indistinguishable from
    untrained); ConvAttn was reliable on every seed tested, which is why it's the
    deployed architecture. See `report/report.tex` §7.1 "Multi-seed robustness" for the
    full data, including a standard mitigation (LR warmup + tighter grad clipping) that
    was tried and found to be a net-neutral-to-negative fix, not adopted.
14. **Optuna tuning is scoped small on purpose** (15 trials, CRNN only, short per-trial
    epoch budget) — the brief explicitly says not to chase accuracy for this PoC; the
    search exists to demonstrate the technique and report an honest tuned-vs-untuned
    comparison (`report/report.tex` §7.4), not to squeeze out maximum accuracy. It
    predates the deploy switch to ConvAttn and is disclosed as such in the report.

## What's committed vs. what's regeneratable

- Committed: all `src/*.py`, `tests/`, `papers/`, `report/`, `README.md`, `requirements.txt`,
  `weights/model.pt` + `weights/model_{crnn,fcn,convattn}.pt` + per-seed sweep checkpoints
  + `weights/rotation_classifier.pt`, `logs/train_log_{crnn,fcn,convattn}.txt`,
  `logs/eval_results_{crnn,fcn,convattn}.json`, `logs/eval_results_ood_convattn.json`,
  `logs/pipeline_eval.json`, `logs/optuna_results.json`, `logs/experiment_comparison.json`,
  `logs/seed_results/`, `logs/plots/*.png`, `data/dataset.csv`, and a full copy of
  `data/images/` + `data/labels/` + `data.yaml` (small enough at this sample count —
  ~3800 tiny 128x32 PNGs — to keep in the repo for full reproducibility).
- Regeneratable at will: everything under `data/`, `data_rotation/`, `data_ood/`, and
  `weights/`, by re-running the relevant `generate_*.py` script (`--seed 42` for the
  exact same dataset) and `train.py`.

## Papers

`papers/` holds a short note per citation used in the report (title, abstract, and
exactly which claim it backs) — see `papers/README.md` for the index. Free arXiv PDFs
are included under `papers/pdfs/` for the 5 citations that have an open-access preprint;
one (Liu 2022, ACM Digital Library only) has no free PDF available, so only its note
is included.

## Known limitations / what I'd sanity-check further given more time

- Synthetic-only training data; no real-world/photographed hex images tested (the OOD
  test approximates this with unseen fonts/blur/contrast, but isn't a substitute).
- **46.75% out-of-distribution accuracy** (unseen fonts, blur, low contrast, wider
  rotation) vs. 96.50% in-distribution — a real, disclosed generalization gap, not
  something this PoC claims to have solved. The clearest target for future work.
- **CRNN and FCN fail to converge on ~40% of random training seeds** (see assumption
  13) — a genuine, only-partially-mitigated training-stability issue, and the reason
  ConvAttn (seed-stable on every seed tested) is the deployed architecture instead.
- No stratification by digit-length in the train/val/test split (documented as an
  acceptable simplification, given the small 1-3 digit space).
- The `/predict` endpoint's resize-to-fixed-canvas preprocessing will distort images
  with very different aspect ratios from the training data (wide vs. tall crops).
- RL training loop (`report/report.tex` §8.3) is specified but intentionally not
  implemented in code, per the brief's "no code needed" for Part A. The reward function
  itself (§8.2) *is* implemented and tested — `src/reward.py`, `tests/test_reward.py`.
- The character-level confusion matrix in `error_analysis.py` is diagonal-dominant and
  not very informative at n=400 (too few off-diagonal errors to show a pattern) — the
  failure-case montage it also produces is the more useful artifact at this sample size.
- `api.py` serves the recognizer alone, not the rotation-aware pipeline — matches the
  brief's task (reviewers post upright hex images) but means `/predict` will not
  correctly read a sideways/upside-down image without wiring in `pipeline.py`.
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
fixed; the full list with before/after is in `report/report.tex` §10 — kept visible
rather than silently corrected, since the brief explicitly grades "the ability to spot
and fix the inevitable bugs the AI creates."
