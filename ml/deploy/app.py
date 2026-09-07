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
from fastapi import FastAPI, File, HTTPException, UploadFile
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
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


@lru_cache(maxsize=1)
def get_pytorch_model():
    """
    Load ResNet50 for Grad-CAM (/explain endpoint only).

    TODO: Replace random weights with real trained weights once the MATLAB model
          is ported. Load like:
              model.load_state_dict(torch.load('models/netrapulse_resnet50.pth'))
    """
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        raise RuntimeError("torch/torchvision not installed. Run: pip install torch torchvision")

    logger.warning(
        "Loading PyTorch ResNet50 with RANDOM WEIGHTS for Grad-CAM. "
        "Replace with real trained weights for accurate heatmaps. (See TODO in app.py)"
    )
    model = models.resnet50(weights=None, num_classes=5)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

def _load_and_preprocess(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes and apply canonical preprocessing."""
    from ml.data.preprocessing import preprocess_fundus

    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return preprocess_fundus(img_bgr, img_size=IMG_SIZE)


def _to_onnx_tensor(processed: np.ndarray) -> np.ndarray:
    """Convert HWC uint8 RGB to NCHW float32 normalised tensor."""
    tensor = processed.astype(np.float32) / 255.0          # [0, 1]
    tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]    # NCHW
    return tensor


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

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
        confidence:  per-class softmax scores as a list of 5 floats
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
        logits = outputs[0][0]  # shape: (5,)
    except Exception as e:
        logger.exception("ONNX inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Softmax
    exp = np.exp(logits - np.max(logits))
    confidence = (exp / exp.sum()).tolist()
    grade = int(np.argmax(confidence))

    return {
        "grade": grade,
        "label": DR_GRADE_LABELS[grade],
        "confidence": confidence,
    }


@app.post("/explain", tags=["Explainability"])
async def explain(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Return a Grad-CAM heatmap overlaid on the fundus image.

    The heatmap is computed on the last conv block of a PyTorch ResNet50.

    NOTE: Currently uses RANDOM weights (placeholder). Replace with real
    trained weights for clinically meaningful heatmaps. See TODO in get_pytorch_model().

    Returns:
        grade:        predicted DR grade (0-4)
        label:        human-readable label
        heatmap_png:  base64-encoded PNG of the Grad-CAM overlay
    """
    try:
        import torch
        import torch.nn.functional as F
        model = get_pytorch_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        processed = _load_and_preprocess(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image preprocessing failed: {e}")

    # Build tensor
    tensor = torch.from_numpy(processed.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # NCHW

    # --- Grad-CAM on layer4 (last conv block) ---
    gradients: list[torch.Tensor] = []
    activations: list[torch.Tensor] = []

    def _save_grad(grad: torch.Tensor) -> None:
        gradients.append(grad)

    def _forward_hook(module, inp, out: torch.Tensor) -> None:
        activations.append(out)
        out.register_hook(_save_grad)

    hook = model.layer4.register_forward_hook(_forward_hook)

    try:
        model.zero_grad()
        output = model(tensor)
        grade = int(output.argmax(dim=1).item())

        # Backprop for the predicted class
        score = output[0, grade]
        score.backward()
    finally:
        hook.remove()

    if not gradients or not activations:
        raise HTTPException(status_code=500, detail="Grad-CAM hook did not fire.")

    # Pool gradients → weight activation maps
    pooled_grads = gradients[0].mean(dim=[0, 2, 3], keepdim=True)
    cam = (activations[0] * pooled_grads).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Resize to original image size
    cam_np = cam[0, 0].detach().numpy()
    heatmap = cv2.resize(cam_np, (IMG_SIZE, IMG_SIZE))
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
