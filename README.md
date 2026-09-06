# NetraPulse

Diabetic retinopathy screening prototype.

## Architecture

```
MATLAB (ResNet50 training + Grad-CAM on APTOS/IDRiD)
  → ONNX export
    → FastAPI /ml/deploy  (the ONLY live backend)
      ├── Next.js /web frontend
      └── Orthanc DICOM + HAPI-FHIR (via docker-compose)
```

MATLAB is **training/R&D only** — it never runs live. FastAPI is the single endpoint
that the frontend, DICOM worker, and judges' laptops talk to.

---

## Quick Start

### 1. Python backend
```powershell
cd ml
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r deploy/requirements.txt
uvicorn deploy.app:app --reload --port 8000
```
→ Open http://localhost:8000/docs

### 2. Frontend
```powershell
cd web
pnpm install
pnpm dev
```
→ Open http://localhost:3000

### 3. DICOM / FHIR stack
```powershell
docker compose up -d
```
→ Orthanc: http://localhost:8042  
→ HAPI-FHIR: http://localhost:8080

---

## Folder Layout

| Path | Contents |
|---|---|
| `/matlab` | MATLAB scripts (training, Grad-CAM, IQA) |
| `/data/aptos` | APTOS 2019 fundus images (DVC-tracked) |
| `/data/idrid` | IDRiD fundus images (DVC-tracked) |
| `/ml/data` | Python preprocessing module |
| `/ml/deploy` | FastAPI app + ONNX model |
| `/ml/explain` | Grad-CAM utilities |
| `/ml/grading` | Evaluation + MLflow logging |
| `/ml/scripts` | Data download + ONNX smoke-test |
| `/web` | Next.js 15 frontend |

---

## Datasets

- **APTOS 2019**: `kaggle competitions download -c aptos2019-blindness-detection`
- **IDRiD**: Manual download from [ieee-dataport.org](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) → extract into `data/idrid`

Track with DVC after downloading:
```powershell
dvc add data\aptos data\idrid
git add *.dvc .gitignore
git commit -m "Track real data with DVC"
```
