# Hexadecimal Image Recognition

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/ASGI-Uvicorn-2E3440?logo=gunicorn&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![Optuna](https://img.shields.io/badge/tuning-Optuna-0080FF?logo=optuna&logoColor=white)
![pytest](https://img.shields.io/badge/tests-7%20passed-brightgreen?logo=pytest&logoColor=white)
![Accuracy](https://img.shields.io/badge/test%20accuracy-96.50%25-success)
![Params](https://img.shields.io/badge/parameters-606K-blue)

A compact computer-vision system that recognizes a hexadecimal literal in an
image and returns its decimal value.

```text
image → CNN feature extractor → self-attention encoder → CTC transcription
      → hexadecimal string → deterministic decimal conversion
```

The project includes synthetic data generation, three trainable model
architectures, evaluation and robustness experiments, pretrained checkpoints,
and a FastAPI inference service.

## Quick start

### 1. Create an environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

GPU installation:

```powershell
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install fastapi "uvicorn[standard]" python-multipart numpy pandas pillow PyYAML matplotlib pytest optuna
```

CPU-only installation:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi "uvicorn[standard]" python-multipart numpy pandas pillow PyYAML matplotlib pytest optuna
```

### 2. Start the API

Scripts use flat imports, so run the service from `src/`:

```powershell
cd src
uvicorn api:app --host 0.0.0.0 --port 8000
```

The server loads `weights/model.pt` by default.

### 3. Send an image

```powershell
curl.exe -X POST `
  -F "file=@../data/images/test/test_000000.png" `
  http://localhost:8000/predict
```

Example response:

```json
{
  "hex_prediction": "0x92c",
  "decimal_prediction": 2348,
  "valid_hex_format": true,
  "latency_ms": 1.1
}
```

Check service readiness:

```powershell
curl.exe http://localhost:8000/health
```

```json
{
  "status": "ok",
  "device": "cuda",
  "arch": "convattn",
  "parameter_count": 606738,
  "checkpoint_loaded": true
}
```

## Model performance

### Deployed checkpoint

`weights/model.pt` is the ConvAttn seed-45 checkpoint.

| Metric | Result |
|---|---:|
| Test hexadecimal exact match | 96.50% |
| Test decimal exact match | 96.50% |
| Mean character accuracy | 99.11% |
| Parameters | 606,738 |
| Checkpoint size | 2.37 MB |
| Mean batch-1 GPU latency | 0.99 ms |
| p95 batch-1 GPU latency | 1.06 ms |

Latency was measured with 100 synchronized batch-1 forward passes after
warm-up on an NVIDIA RTX 5070 Ti.

### Architecture comparison

Each model uses the same CNN stem, dataset, training configuration, and seed
42. Only the sequence-modeling head changes.

| Model | Sequence head | Exact match | Parameters | Size | Mean latency |
|---|---|---:|---:|---:|---:|
| CRNN | Bidirectional GRU | 96.50% | 542,226 | 2.08 MB | 0.663 ms |
| FCN | Dilated Conv1d | 87.00% | 490,386 | 1.89 MB | 0.707 ms |
| ConvAttn | Transformer encoder | 94.50% | 606,738 | 2.37 MB | 1.015 ms |

### Reliability across seeds

| Model | Mean test accuracy | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| CRNN | 84.05% | 22.66 points | 38.75% | 96.50% |
| FCN | 36.05% | 42.11 points | 1.50% | 88.25% |
| ConvAttn | **94.25%** | **1.99 points** | **90.75%** | **96.50%** |

ConvAttn is deployed because it converged successfully for every tested seed.

### Robustness

| Condition | Exact match |
|---|---:|
| Standard test set | 96.50% |
| OOD fonts, blur, contrast, and angle jitter | 46.75% |
| Rotated images without correction | 6.00% |
| Rotated images with orientation correction | 80.00% |

The separate four-way orientation classifier achieves 99.00% accuracy.

## Architecture

All recognizers accept a normalized `1 × 32 × 128` grayscale image and emit
CTC log probabilities over 17 characters plus the blank token.

| Component | Configuration |
|---|---|
| Visual encoder | Shared four-block CNN |
| Sequence length | 32 timesteps |
| Vocabulary | `0123456789abcdefx` |
| Deployed sequence encoder | Two Transformer layers, four attention heads |
| Output | Greedy CTC hexadecimal transcription |
| Conversion | Deterministic `int(prediction, 16)` |

Three interchangeable sequence heads are implemented:

- `crnn`: one-layer bidirectional GRU;
- `fcn`: three dilated 1D convolution blocks; and
- `convattn`: two-layer self-attention encoder with positional encoding.

## Project structure

```text
.
├── data/                       # recognition images, YOLO labels, CSV, YAML
├── data_ood/                   # out-of-distribution evaluation set
├── data_rotation/              # orientation-classifier dataset
├── logs/                       # training logs, metrics, and plots
├── report/
│   ├── report.pdf              # technical report
│   ├── report.tex              # report source
│   └── figures/
├── src/
│   ├── api.py                  # FastAPI service
│   ├── common.py               # shared constants and CTC decoder
│   ├── dataset.py              # recognition dataset loader
│   ├── generate_dataset.py     # synthetic dataset generator
│   ├── model.py                # CRNN, FCN, and ConvAttn models
│   ├── train.py                # supervised CTC training
│   ├── evaluate.py             # accuracy, size, and latency evaluation
│   ├── pipeline.py             # orientation correction + recognition
│   ├── reward.py               # RL reward-function reference
│   └── ...
├── tests/                      # reward-function tests
├── weights/                    # deployed and experiment checkpoints
├── README.md
└── requirements.txt
```

## Reproducing the experiments

Run the following commands from `src/`.

### Generate the dataset

```powershell
python generate_dataset.py --seed 42
```

The generator creates 3,000 training, 400 validation, and 400 test images,
along with:

- one YOLO bounding box per rendered character;
- `data/data.yaml`; and
- `data/dataset.csv`.

### Train

```powershell
python train.py --arch crnn --epochs 80 --patience 12 --seed 42
python train.py --arch fcn --epochs 80 --patience 12 --seed 42
python train.py --arch convattn --epochs 80 --patience 12 --seed 42
```

Reproduce the deployed training run:

```powershell
python train.py --arch convattn --epochs 80 --patience 12 --seed 45
```

Training writes seed-specific checkpoints and TXT logs. It does not
automatically replace `weights/model.pt`.

### Evaluate

```powershell
python evaluate.py `
  --checkpoint ../weights/model.pt `
  --split test `
  --out-json ../logs/eval_results_convattn_deployed.json
```

Evaluate another checkpoint:

```powershell
python evaluate.py --checkpoint ../weights/model_crnn.pt --split test
python evaluate.py --checkpoint ../weights/model_fcn.pt --split test
python evaluate.py --checkpoint ../weights/model_convattn.pt --split test
```

### Rotation pipeline

```powershell
python generate_rotation_dataset.py --seed 42
python train_rotation_classifier.py
python evaluate_pipeline.py
```

### Out-of-distribution evaluation

```powershell
python generate_ood_dataset.py
python evaluate.py `
  --checkpoint ../weights/model.pt `
  --data-dir ../data_ood `
  --split test `
  --out-json ../logs/eval_results_ood_convattn.json
```

## Serving another checkpoint

Set `HEX_MODEL_PATH` before starting the API:

```powershell
$env:HEX_MODEL_PATH = "..\weights\model_crnn.pt"
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API reads the architecture and normalization configuration from checkpoint
metadata.

## Testing

Run from the repository root:

```powershell
pytest -q
```

## Dataset

Images contain literals from `0x0` to `0xfff`. Values are sampled uniformly
by hexadecimal digit count and rendered with variation in:

- font family and size;
- horizontal and vertical position;
- foreground and background intensity;
- small-angle rotation; and
- Gaussian pixel noise.

The recognition canvas is `128 × 32`. Large rotations are handled by a
separate classifier operating on a `128 × 128` canvas.

## RL design

The deployed recognizer uses supervised CTC training only. The report also
designs an optional RL stage for a hypothetical generative model that emits
the decimal answer directly.

The executable reward in `src/reward.py` combines:

| Reward component | Value |
|---|---:|
| Malformed output | -0.5 |
| Correct decimal format | +0.2 |
| Value within 0–4095 | +0.1 |
| Exact decimal answer | +0.7 |
| Wrong but numerically close answer | Up to +0.3 |

Exact correctness remains the final evaluation metric.

## Limitations

- Training and evaluation data are synthetic.
- OOD accuracy is substantially lower than in-distribution accuracy.
- `/predict` resizes uploads to `128 × 32`; extreme aspect ratios may distort text.
- The public API assumes upright input; rotation correction is currently a separate pipeline.
- Greedy CTC decoding can drop adjacent repeated characters.
- CRNN and FCN training is sensitive to random initialization.

## Documentation

See [report/report.pdf](report/report.pdf) for the complete system design,
experimental analysis, RL proposal, and deployment discussion.
