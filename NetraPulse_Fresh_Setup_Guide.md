# NetraPulse — Fresh Setup Guide (Windows + Google Antigravity)

Target architecture: **MATLAB trains + explains → exports to ONNX → FastAPI serves it → Next.js + Orthanc/FHIR consume it.**
Everything below assumes you're starting the folder from zero. Commands are PowerShell.

---

## Phase 0 — Before you touch code

Decide these two things now, they're load-bearing for everything after:

1. MATLAB is your **training/R&D** environment only. It is never the thing that runs live during a demo.
2. Python (FastAPI) is the **only** thing Next.js, Orthanc, and judges' laptops ever talk to. This means nobody needs a MATLAB license to run your demo.

---

## Phase 1 — Install prerequisites

| Tool | Why | Where |
|---|---|---|
| Google Antigravity | your IDE | `antigravity.google/download` — **only this domain.** There's a documented trojanized installer going around on a typo-squatted lookalike domain, so don't use a search ad or mirror link. Pick the x64 installer unless you're on an ARM Windows laptop. |
| Git for Windows | version control | `git-scm.com` |
| Python 3.11 | the `/ml` backend | `python.org` — **tick "Add python.exe to PATH"** during install |
| Node.js LTS | the `/web` frontend | `nodejs.org`, then run `corepack enable` once installed to get `pnpm` |
| Docker Desktop | Orthanc + HAPI-FHIR | `docker.com` — needs WSL2 backend; the installer prompts you to enable it |
| MATLAB (you likely have this) | training + Grad-CAM | Open MATLAB → Add-On Explorer → search **"Deep Learning Toolbox Converter for ONNX Model Format"** → Install. You need this specific add-on for the ONNX export step later. |
| Kaggle account | dataset access | `kaggle.com` → Account → "Create New API Token" → downloads `kaggle.json` |

Verify installs:
```powershell
git --version
python --version
node --version
pnpm --version
docker --version
```

---

## Phase 2 — Project skeleton

```powershell
mkdir netrapulse
cd netrapulse
git init
mkdir matlab, data, web
mkdir ml
mkdir ml\data, ml\deploy, ml\explain, ml\grading, ml\quality, ml\reports, ml\robustness
```

Open the `netrapulse` folder in Antigravity: `File → Open Folder`. Copy your existing MATLAB scripts and Python package files into the matching folders above if you already have them from before.

---

## Phase 3 — Get real data (replaces synthetic generation)

**Kaggle CLI setup:**
```powershell
pip install kaggle
mkdir $env:USERPROFILE\.kaggle
copy C:\Users\<you>\Downloads\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
```

**APTOS 2019** (matches your existing 5-class 0–4 setup):
```powershell
kaggle competitions download -c aptos2019-blindness-detection -p data\aptos
Expand-Archive data\aptos\aptos2019-blindness-detection.zip -DestinationPath data\aptos
```
You'll need to accept the competition rules on the Kaggle website first (button on the competition page) or the download will 403.

**IDRiD** (real Indian-clinic data with pixel-level microaneurysm/hemorrhage/exudate masks — this is what will finally let you validate `extract_ma_patches.m` against ground truth instead of synthetic images):
- Go to `ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid`, sign in, download the zip, extract into `data\idrid`.

**Track it with DVC** (already in your `pyproject.toml`, this is the point to start using it):
```powershell
pip install dvc
dvc init
dvc add data\aptos data\idrid
echo "data/" >> .gitignore
git add .dvcignore *.dvc .gitignore
git commit -m "Track real APTOS + IDRiD data with DVC"
```

---

## Phase 4 — Python environment for `/ml`

```powershell
cd ml
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision timm fastapi "uvicorn[standard]" onnx onnxruntime mlflow scikit-learn opencv-python pillow numpy pandas python-multipart
```

Create `ml\data\preprocessing.py` — one canonical preprocessing function, used identically at train time and serve time (this is the fix for the "trained on one preprocessing, served on another" bug):

```python
import cv2
import numpy as np

def preprocess_fundus(image_bgr: np.ndarray, img_size: int = 512, sigma: float = 10) -> np.ndarray:
    """Circle-crop + local-contrast normalization for fundus photos."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = _crop_to_content(img)
    img = cv2.resize(img, (img_size, img_size))

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=mask)

    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    return img

def _crop_to_content(img: np.ndarray, tol: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]
```

---

## Phase 5 — MATLAB: train on real data, export to ONNX

1. In `preprocess_fundus.m` / `train_dr_classifier.m`, point the data loader at `data\aptos\train_images` and `data\idrid` instead of `generate_synthetic_fundus.m`'s output.
2. Train as before (`trainNetwork` / your existing script).
3. Export the trained network:
   ```matlab
   exportONNXNetwork(net, 'netrapulse_resnet50.onnx');
   ```
   If MATLAB prompts you to install the ONNX converter support package, that's the one from Phase 1 — install it and re-run.
4. Move the resulting file into `ml\deploy\models\netrapulse_resnet50.onnx`.

---

## Phase 6 — FastAPI inference service

`ml\deploy\app.py` (skeleton — Antigravity's agent can flesh this out):

```python
from fastapi import FastAPI, UploadFile
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from ml.data.preprocessing import preprocess_fundus
import cv2

app = FastAPI()
session = ort.InferenceSession("ml/deploy/models/netrapulse_resnet50.onnx")

@app.post("/predict")
async def predict(file: UploadFile):
    raw = await file.read()
    img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    processed = preprocess_fundus(img_bgr).astype(np.float32) / 255.0
    tensor = processed.transpose(2, 0, 1)[np.newaxis, ...]
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    grade = int(np.argmax(outputs[0]))
    return {"grade": grade, "confidence": outputs[0].tolist()}
```

Run it:
```powershell
cd ml
uvicorn deploy.app:app --reload --port 8000
```
Open `http://localhost:8000/docs` and upload a real APTOS image to sanity-check it returns a grade. Add the `/explain` (Grad-CAM) endpoint next, using a plain PyTorch copy of the ResNet50 for that one route, per the earlier architecture note.

---

## Phase 7 — Next.js frontend

```powershell
cd web
pnpm install
pnpm dev
```
Wire your upload page to `POST http://localhost:8000/predict` and render the grade + confidence with `recharts` (already a dependency).

---

## Phase 8 — Docker: Orthanc + HAPI-FHIR

```powershell
cd netrapulse
docker compose up -d
```
Check Orthanc at `http://localhost:8042` and HAPI-FHIR at `http://localhost:8080`. Write a small Python worker (`ml/deploy/orthanc_worker.py`) that polls Orthanc's REST API for new studies, sends the image to `/predict`, and POSTs a FHIR `DiagnosticReport` back to HAPI-FHIR with the grade. Build this after Phase 6/7 work, not before.

---

## Phase 9 — Working in Antigravity

Open **Manager view** (`Ctrl+E`) and split the remaining work across parallel agents instead of doing it serially:
- Agent 1: flesh out `ml/deploy/app.py` (`/predict` + `/explain`)
- Agent 2: Next.js upload page + results view
- Agent 3: `orthanc_worker.py`
- Agent 4: MLflow logging + quadratic weighted kappa scoring in `ml/grading`

Let the browser-capable agent actually click through the upload → result flow once wired, rather than trusting it compiles.

---

## Phase 10 — Sanity checklist before you call it "working"

- [ ] `data\aptos` and `data\idrid` populated, tracked in DVC
- [ ] MATLAB trains on real images, not `generate_synthetic_fundus.m`
- [ ] `netrapulse_resnet50.onnx` exists and loads in `onnxruntime`
- [ ] `POST /predict` returns a real grade for a real uploaded image
- [ ] Next.js shows that result end to end in the browser
- [ ] MLflow has at least one logged run with quadratic weighted kappa
- [ ] Orthanc → worker → FHIR round-trip produces a `DiagnosticReport`
