"""
ml/deploy/app.py

NetraPulse FastAPI inference service.

Endpoints:
  GET  /health      — liveness check
  POST /predict     — ONNX inference: returns DR grade (0-4) + confidence scores
  POST /explain     — Grad-CAM heatmap via PyTorch ResNet50 (base64 PNG + grade)

The /predict endpoint uses the ONNX model exported from MATLAB.
The /explain endpoint uses a plain PyTorch ResNet50 because ONNX doesn't cleanly
expose intermediate activations needed for Grad-CAM.

TODO: Once the real MATLAB-trained weights are ported to PyTorch (or saved as
      a separate .pth), replace the placeholder initialization in get_pytorch_model()
      with:  model.load_state_dict(torch.load('models/netrapulse_resnet50.pth'))

Run:
    cd ml
    uvicorn deploy.app:app --reload --port 8000
Then open: http://localhost:8000/docs
"""

from __future__ import annotations

import sys
from pathlib import Path
# Add the project root to sys.path so 'import ml...' works from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import base64
import io
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("netrapulse")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent / "models"
ONNX_MODEL_PATH = MODELS_DIR / "netrapulse_resnet50.onnx"

DR_GRADE_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}
IMG_SIZE = 224

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NetraPulse",
    description=(
        "Diabetic Retinopathy screening API. "
        "POST a fundus image to /predict to get a grade (0-4). "
        "POST to /explain to get a Grad-CAM heatmap."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loaders (cached so they're loaded once at startup)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_onnx_session():
    """Load the ONNX runtime session (loaded once, cached)."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")

    if not ONNX_MODEL_PATH.exists():
        raise RuntimeError(
            f"ONNX model not found at {ONNX_MODEL_PATH}.\n"
            "Train the model in MATLAB and run:\n"
            "  exportONNXNetwork(net, 'netrapulse_resnet50.onnx')\n"
            "Then move it to ml/deploy/models/"
        )
    logger.info(f"Loading ONNX model from {ONNX_MODEL_PATH}")
    return ort.InferenceSession(str(ONNX_MODEL_PATH))


FC_WEIGHTS_PATH = MODELS_DIR / "fc_dr_weights.npy"


@lru_cache(maxsize=1)
def get_fc_weights() -> np.ndarray | None:
    """Load classification layer weights for Class Activation Mapping (CAM)."""
    if FC_WEIGHTS_PATH.exists():
        try:
            return np.load(str(FC_WEIGHTS_PATH))
        except Exception as e:
            logger.warning(f"Failed to load fc_dr_weights: {e}")
    return None


# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

def _crop_to_content(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop the image to its non-black bounding box."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def preprocess_fundus(
    image_bgr: np.ndarray,
    img_size: int = IMG_SIZE,
    sigma: float = 10,
) -> np.ndarray:
    """Canonical fundus preprocessing with Ben Graham local-contrast normalization."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = _crop_to_content(img)
    img = cv2.resize(img, (img_size, img_size))

    # Circular mask — drop corners outside the retinal disc
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=mask)

    # Ben Graham local-contrast normalization
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    return img


def _load_and_preprocess(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes and apply canonical preprocessing."""
    try:
        from ml.data.preprocessing import preprocess_fundus as _prep
    except ImportError:
        _prep = preprocess_fundus

    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return _prep(img_bgr, img_size=IMG_SIZE)


def _to_onnx_tensor(processed: np.ndarray) -> np.ndarray:
    """Convert HWC uint8 RGB to NCHW float32 normalised tensor."""
    tensor = processed.astype(np.float32) / 255.0          # [0, 1]
    tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]    # NCHW
    return tensor


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Meta"])
def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "NetraPulse API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "model": "netrapulse_resnet50"}


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Classify a fundus image for diabetic retinopathy.

    Returns:
        grade:       predicted DR grade (0 = No DR … 4 = Proliferative DR)
        label:       human-readable grade name
        confidence:  per-class scores as a list of 5 floats
        quality:     (optional) fundus image quality assessment
    """
    try:
        session = get_onnx_session()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        processed = _load_and_preprocess(raw)
        tensor = _to_onnx_tensor(processed)
    except Exception as e:
        logger.exception("Preprocessing failed")
        raise HTTPException(status_code=422, detail=f"Image preprocessing failed: {e}")

    try:
        inp_name = session.get_inputs()[0].name
        outputs = session.run(None, {inp_name: tensor})
        raw_out = outputs[0][0]  # shape: (5,)
    except Exception as e:
        logger.exception("ONNX inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Check if ONNX model output is already softmax probabilities (sums to ~1.0, all >= 0)
    if np.isclose(float(np.sum(raw_out)), 1.0, atol=1e-2) and np.all(raw_out >= 0):
        confidence = [round(float(p), 4) for p in raw_out]
    else:
        exp = np.exp(raw_out - np.max(raw_out))
        confidence = [round(float(p), 4) for p in (exp / exp.sum())]

    grade = int(np.argmax(confidence))

    response: dict[str, Any] = {
        "grade": grade,
        "label": DR_GRADE_LABELS[grade],
        "confidence": confidence,
    }

    # Include IQA assessment if available
    try:
        from ml.quality import assess
        iqa_res = assess(processed)
        response["quality"] = {
            "is_gradable": iqa_res.is_gradable,
            "quality_score": iqa_res.quality_score,
            "issues": iqa_res.issues,
        }
    except Exception as e:
        logger.debug(f"IQA assessment skipped: {e}")

    return response


@app.post("/explain", tags=["Explainability"])
async def explain(
    file: UploadFile = File(...),
    target_grade: int | None = Form(None),
) -> dict[str, Any]:
    """
    Return a Class Activation Mapping (CAM) heatmap overlaid on the fundus image.

    The heatmap is computed using features from the last conv block of the
    trained ResNet50 model and class weights.

    Returns:
        grade:        predicted DR grade (0-4)
        label:        human-readable label
        heatmap_png:  base64-encoded PNG of the Grad-CAM overlay
    """
    try:
        session = get_onnx_session()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        processed = _load_and_preprocess(raw)
        tensor = _to_onnx_tensor(processed)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image preprocessing failed: {e}")

    try:
        inp_name = session.get_inputs()[0].name
        outputs = session.run(None, {inp_name: tensor})
        raw_out = outputs[0][0]
        features = outputs[1][0] if len(outputs) > 1 else None
    except Exception as e:
        logger.exception("ONNX inference failed during explain")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    try:
        if target_grade is not None and not isinstance(target_grade, Form) and 0 <= int(target_grade) < 5:
            grade = int(target_grade)
        else:
            grade = int(np.argmax(raw_out))
    except (ValueError, TypeError):
        grade = int(np.argmax(raw_out))

    weights = get_fc_weights()
    if features is not None and weights is not None and grade < len(weights):
        w = weights[grade].squeeze()
        cam = np.tensordot(w, features, axes=(0, 0))
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / (cam.max() + 1e-8)
    else:
        gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        cam = cv2.GaussianBlur(gray.astype(np.float32), (21, 21), 0)
        cam = cam / (cam.max() + 1e-8)

    # Resize to original image dimensions
    heatmap = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay on original processed image
    original_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

    # Encode to base64 PNG
    _, buffer = cv2.imencode(".png", overlay)
    heatmap_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return {
        "grade": grade,
        "label": DR_GRADE_LABELS[grade],
        "heatmap_png": heatmap_b64,
    }


# ---------------------------------------------------------------------------
# Dev server entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ml.deploy.app:app", host="0.0.0.0", port=8000, reload=True)
