"""FastAPI deployment for the hex-to-decimal recognizer.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   -> basic liveness/model-loaded check
    POST /predict  -> multipart image upload -> {hex_prediction, decimal_prediction, ...}
"""
import io
import os
import time

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from common import IMG_HEIGHT, IMG_WIDTH, ctc_greedy_decode, hex_to_decimal
from model import count_parameters, get_model

MODEL_PATH = os.environ.get(
    "HEX_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "weights", "model.pt"),
)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB, generous for a 128x32-class image

app = FastAPI(title="Hex-to-Decimal Recognizer", version="1.0")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_arch = None
_n_params = None
_checkpoint_loaded = False


class PredictResponse(BaseModel):
    hex_prediction: str
    decimal_prediction: int | None
    valid_hex_format: bool
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    device: str
    arch: str | None
    parameter_count: int | None
    checkpoint_loaded: bool


@app.on_event("startup")
def load_model():
    # HEX_MODEL_PATH may point at any of weights/model_{crnn,fcn,convattn}.pt --
    # the architecture is read from the checkpoint's own "arch" field rather
    # than hardcoded, so this stays correct regardless of which checkpoint is
    # deployed (previously this always instantiated HexCRNN, which crashed on
    # load_state_dict for any other architecture's checkpoint).
    global _model, _arch, _n_params, _checkpoint_loaded

    if not os.path.isfile(MODEL_PATH):
        # Fail loudly rather than silently serving an untrained, randomly
        # initialized model under a "status": "ok" response.
        _model = None
        _arch = None
        _n_params = None
        _checkpoint_loaded = False
        return

    checkpoint = torch.load(MODEL_PATH, map_location=_device)
    _arch = checkpoint.get("arch", "crnn")
    _model = get_model(_arch, norm=checkpoint.get("norm", "batch")).to(_device)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()
    _n_params = count_parameters(_model)
    _checkpoint_loaded = True


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _model is not None else "no_checkpoint_loaded",
        device=str(_device),
        arch=_arch,
        parameter_count=_n_params,
        checkpoint_loaded=_checkpoint_loaded,
    )


def preprocess(image_bytes: bytes) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # Training images are a fixed 128x32 canvas; resizing preserves the
    # pipeline's simplicity for this PoC (documented assumption in README --
    # a production version would pad-to-aspect instead of stretching).
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype="float32")
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    tensor = (tensor / 255.0 - 0.5) / 0.5
    return tensor


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded (no checkpoint found at HEX_MODEL_PATH).")

    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image upload, got content-type={file.content_type!r}.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds 5 MB limit.")

    tensor = preprocess(image_bytes).to(_device)

    start = time.perf_counter()
    with torch.no_grad():
        log_probs = _model(tensor)
        pred_text = ctc_greedy_decode(log_probs.cpu())[0]
    latency_ms = (time.perf_counter() - start) * 1000

    try:
        decimal_value = hex_to_decimal(pred_text)
        valid = True
    except ValueError:
        decimal_value = None
        valid = False

    return PredictResponse(
        hex_prediction=pred_text,
        decimal_prediction=decimal_value,
        valid_hex_format=valid,
        latency_ms=round(latency_ms, 3),
    )
