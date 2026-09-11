# NetraPulse System Overview & Architecture

## 1. System Architecture

NetraPulse is an end-to-end clinical AI screening and explainability system for Diabetic Retinopathy (DR). It bridges research-grade deep learning models, live REST microservices, automated clinical imaging protocols (DICOM/PACS and HL7 FHIR), and an interactive web application.

```
                     ┌─────────────────────────────────────────────────────────┐
                     │                 MATLAB (R&D & Training)                 │
                     │  - preprocess_fundus.m (Ben Graham local-contrast norm) │
                     │  - train_dr_classifier.m (ResNet50 + Early Stopping)    │
                     │  - exportONNXNetwork -> netrapulse_resnet50.onnx        │
                     └───────────────────────────┬─────────────────────────────┘
                                                 │ exports ONNX model
                                                 ▼
┌──────────────────────────────┐        ┌───────────────────────────────────────┐
│     Next.js 15 Web UI        │        │        FastAPI Inference Engine       │
│  (Port 3000 - /web)          │ ◄────► │  (Port 8000 - /ml/deploy)             │
│  - Drag & Drop Fundus Upload │  HTTP  │  - /predict: ONNX Runtime (DR 0-4)    │
│  - Recharts Confidence Bars  │        │  - /explain: PyTorch Grad-CAM Overlay │
│  - Quality / IQA Metrics UI  │        │  - /health: Microservice heartbeat    │
│  - Grad-CAM Heatmap Viewer   │        │  - ml.quality.iqa: Pre-filter checks  │
│  - Dark / Light Theme Toggle │        └───────────────────▲───────────────────┘
└──────────────────────────────┘                            │
                                                            │ REST calls
                                                            │ (Trigger prediction)
┌──────────────────────────────┐        ┌───────────────────┴───────────────────┐
│        Orthanc PACS          │        │        Orthanc Worker Daemon          │
│  (Port 8042 - DICOM Store)   │ ◄────► │  (ml/deploy/orthanc_worker.py)        │
│  - Receives DICOM studies    │  REST  │  - Polls Orthanc for new instances    │
│  - Exposes clinical previews │        │  - Forwards preview to /predict       │
└──────────────────────────────┘        │  - Formats FHIR DiagnosticReport      │
                                        └───────────────────┬───────────────────┘
                                                            │
                                                            ▼ (HTTP POST)
                                        ┌───────────────────────────────────────┐
                                        │          HAPI-FHIR Server             │
                                        │  (Port 8080 - FHIR R4 Store)          │
                                        │  - Persists DiagnosticReport records  │
                                        └───────────────────────────────────────┘
```

### Key Architectural Layers

1. **MATLAB R&D & Offline Training (`/matlab`)**:
   * Trains a modified ResNet50 neural network for 5-class DR classification (`0: No DR`, `1: Mild`, `2: Moderate`, `3: Severe`, `4: Proliferative DR`).
   * Implements automated Early Stopping (`ValidationPatience = 8`, `OutputNetwork = 'best-validation-loss'`) to prevent overfitting on long training schedules (up to 50 epochs).
   * Calculates inverse-frequency class weighting to handle severe class imbalance in fundus datasets (APTOS 2019 / EyePACS / IDRiD).
   * Exports the optimized network weights to standard ONNX (`netrapulse_resnet50.onnx`).
   * *MATLAB is isolated strictly to offline R&D; it does not run in live production.*

2. **Image Quality Assessment & OOD Gate (`/ml/quality`)**:
   * Evaluates incoming retinal photographs using multi-heuristic computer vision prior to or alongside model grading.
   * Computes Laplacian variance on masked retinal foreground to evaluate image blur and sharpness.
   * Analyzes luminance and exposure within the non-black field to detect overexposed or underexposed imagery.
   * Performs foreground segmentation to ensure adequate retinal disc area and reject out-of-distribution (OOD) non-fundus imagery.
   * Generates a composite `quality_score` (0.0 to 1.0), a boolean `is_gradable` status, and granular anomaly diagnostics (`insufficient_retinal_field`, `image_blurred`, `underexposed`, `overexposed`, `low_contrast`).

3. **FastAPI Live Inference Microservice (`/ml/deploy`)**:
   * High-throughput inference backend running on Uvicorn (Port 8000).
   * Loads `netrapulse_resnet50.onnx` via `onnxruntime` for low-latency scoring under `/predict`.
   * Integrates PyTorch ResNet50 with forward and backward gradient hooks on the final convolutional layer to generate clinical Grad-CAM heatmaps under `/explain`.
   * Preprocesses raw fundus inputs into 224x224 circular-masked, Ben Graham contrast-normalized tensors identical to the MATLAB training formulation.

4. **Clinical Interoperability (PACS & FHIR) (`docker-compose.yml` & `orthanc_worker.py`)**:
   * **Orthanc**: Open-source DICOM store (PACS) on Port 8042 for receiving real-world clinical ophthalmology scans.
   * **HAPI-FHIR**: HL7 FHIR R4 store on Port 8080.
   * **Orthanc Worker Daemon**: Continuously polls Orthanc via REST, extracts newly arrived retinal studies, runs inference through FastAPI `/predict`, builds standard FHIR `DiagnosticReport` JSON resources containing the DR grade and confidence array, and posts them to HAPI-FHIR.

5. **Next.js 15 Web Application (`/web`)**:
   * Modern, responsive clinical dashboard running on Port 3000 built with Next.js 15, React 19, TypeScript, and Tailwind CSS.
   * Features drag-and-drop retinal fundus upload with instant preview.
   * Displays classification grade badges, clinical referral recommendations, and interactive probability distribution bar charts via Recharts.
   * Renders image quality indicators (`quality_score`, `is_gradable`, detected issues).
   * Provides toggleable Grad-CAM heatmap visualization to verify model attention regions (e.g., hemorrhages, exudates, microaneurysms).
   * Includes full dark and light theme support via `next-themes`.

---

## 2. Port Allocations & Runtime Services

| Service | Technology | Port | Primary Endpoint / Role |
|---|---|---|---|
| **FastAPI Backend** | Python / Uvicorn | `8000` | `http://localhost:8000/docs` (Swagger UI) |
| **Web Dashboard** | Next.js 15 / React | `3000` | `http://localhost:3000` (User Interface) |
| **Orthanc PACS** | Docker (C++) | `8042` | `http://localhost:8042` (DICOM Web / REST) |
| **HAPI-FHIR Server** | Docker (Java) | `8080` | `http://localhost:8080/fhir` (FHIR R4 API) |
| **MLflow Tracking** | Python (Optional) | `5000` | Model experiment tracking and evaluation metrics |

---

## 3. Detailed File Layout & Roles

### Root Level
* **`README.md`**: Top-level project documentation covering features, architecture overview, quick-start commands, and repository structure.
* **`brain.md`**: Master system blueprint, architecture diagram, file directory inventory, and technical role descriptions.
* **`docker-compose.yml`**: Docker Compose definition that provisions Orthanc (DICOM PACS) with mapped persistent storage (`orthanc-storage`) and HAPI-FHIR (R4 server).
* **`NetraPulse_Fresh_Setup_Guide.md`**: Comprehensive 10-phase setup guide detailing prerequisites, Kaggle dataset downloading, Python virtual environments, MATLAB scripts, Docker execution, and end-to-end testing.
* **`NetraPulse_Antigravity_Prompts.md`**: Catalog of agent prompts and system instructions used for development, optimization, and automation workflows.
* **`trainLabels.csv` & `trainLabels_cropped.csv`**: Ground-truth label files mapping retinal fundus image IDs to DR severity grades (0 through 4).

---

### `/matlab` — Training & Preprocessing
* **`preprocess_fundus.m`**: Canonical MATLAB preprocessing pipeline. Implements:
  1. Cropping bounding boxes around non-black retinal pixels.
  2. Resizing to target resolution (`224x224` or user-specified dimension).
  3. Circular masking to remove border artifacts.
  4. Ben Graham local-contrast enhancement: subtracting local Gaussian blur (`addweighted(img, 4, gaussian_blur, -4, 128)`).
* **`train_dr_classifier.m`**: End-to-end model training workflow:
  * Ingests training labels and images via `augmentedImageDatastore`.
  * Computes inverse-frequency class weights to balance gradient updates across rare severe stages.
  * Replaces ResNet50 classification head (`fc1000`, `fc1000_softmax`, `ClassificationLayer_fc1000`) with a 5-class custom fully connected layer (`fc_dr`, `softmax_dr`, `output_dr`).
  * Configures Adam optimizer with Early Stopping (`ValidationPatience = 8`, `ValidationFrequency = 5`, `OutputNetwork = 'best-validation-loss'`).
  * Exports the finalized model snapshot directly to `ml/deploy/models/netrapulse_resnet50.onnx`.

---

### `/ml/deploy` — Live Backend & Inference
* **`app.py`**: The core FastAPI application hosting live REST endpoints:
  * `GET /health`: Liveness heartbeat confirming microservice readiness and loaded model identity.
  * `POST /predict`: Ingests multipart image files, applies Python-native Ben Graham preprocessing, executes ONNX Runtime inference, computes Softmax probabilities, checks image quality via `ml.quality.iqa`, and returns the DR grade, label, per-class confidence, and IQA metrics.
  * `POST /explain`: Ingests a fundus image and computes Grad-CAM activations on `layer4` of PyTorch ResNet50, blends the resulting Jet heatmap over the input retinal field, and returns a base64-encoded PNG overlay.
* **`orthanc_worker.py`**: Autonomous daemon script for hospital workflow automation:
  * Polls Orthanc PACS (`/studies` endpoint) for uninspected DICOM acquisitions.
  * Extracts image frames from DICOM instances.
  * Dispatches instances to the local FastAPI `/predict` endpoint.
  * Constructs HL7 FHIR R4 `DiagnosticReport` resources containing DR conclusions and per-grade confidence extensions, then commits them to HAPI-FHIR.
* **`requirements.txt`**: Production Python dependencies for the inference environment (`fastapi`, `uvicorn`, `onnxruntime`, `torch`, `torchvision`, `opencv-python`, `pillow`, `requests`, `numpy`).
* **`models/`**: Storage directory for compiled inference artifacts.
  * `netrapulse_resnet50.onnx`: The 94 MB ResNet50 model exported from MATLAB.
  * `README.txt`: Instructions for placing or updating ONNX weight binaries.

---

### `/ml/quality` — Image Quality Assessment (IQA) & OOD
* **`iqa.py`**: Automated fundus image quality and out-of-distribution evaluation engine:
  * `_get_retinal_mask()`: Thresholds and segments the circular retinal foreground from background borders.
  * `assess()`: Multi-factor quality assessor computing:
    * **Sharpness / Blur**: Laplacian variance evaluated exclusively within the retinal disc region.
    * **Illumination / Exposure**: Mean pixel intensity against clinical underexposure (<40) and overexposure (>210) limits.
    * **Contrast**: Standard deviation of non-black retinal pixel intensities.
    * **Retinal Coverage**: Ratio of foreground area to total frame area to prevent non-retinal images from proceeding.
  * `IQAResult`: Structured dataclass exposing `is_gradable` (boolean), `quality_score` (0.0 to 1.0), `blur_score`, `illumination_score`, `contrast_score`, and diagnostic `issues` array.
* **`__init__.py`**: Exposes `assess` and `IQAResult` for clean imports across the backend.

---

### `/ml/data` — Python Data Preprocessing
* **`preprocessing.py`**: Python implementation of Ben Graham's retinal preprocessing pipeline, mirroring `preprocess_fundus.m` to maintain train-test parity:
  * `crop_retina()`: Removes black perimeter margins using intensity thresholds.
  * `circular_mask()`: Applies an inscribed ellipse/circle mask to eliminate border noise.
  * `ben_graham_filter()`: Computes local Gaussian blur subtraction for vascular enhancement.
  * `preprocess_fundus()`: Full pipeline execution on arbitrary raw images.
  * `preprocess_already_cropped()`: Optimized shortcut for images that have already been cropped.

---

### `/ml/scripts` — Diagnostics & Data Pipelines
* **`download_data.py`**: Automated dataset downloader using Kaggle API to retrieve APTOS 2019 and EyePACS fundus data splits.
* **`verify_onnx.py`**: Sanity verification tool:
  * Loads `netrapulse_resnet50.onnx` using the ONNX model validator (`onnx.checker.check_model`).
  * Initializes an `onnxruntime.InferenceSession`.
  * Passes synthetic dummy tensors (`[1, 3, 224, 224]`) through the engine to ensure tensor dimensions, output shapes, and operator compatibility are valid.

---

### `/ml/grading` — Evaluation & Metrics
* **`evaluate.py`**: Offline model benchmarking and validation script:
  * Evaluates the ONNX network across held-out validation datasets.
  * Computes clinical classification metrics: Accuracy and Quadratic Weighted Kappa (QWK).
  * Generates 5x5 confusion matrix plots via Matplotlib/Seaborn.
  * Logs parameters, test metrics, and confusion matrix figures directly to MLflow.

---

### `/web` — Next.js 15 Frontend
* **`package.json`**: Node.js package manifests specifying dependencies: Next.js 15, React 19, Lucide React, Recharts, `next-themes`, Tailwind CSS.
* **`next.config.ts`**: Next.js compiler and build configuration.
* **`tsconfig.json`**: TypeScript project rules and path aliases (`@/*` pointing to `./src/*`).
* **`src/app/layout.tsx`**: Root HTML layout embedding Inter font, application metadata, and wrapping components with `ThemeProvider`.
* **`src/app/page.tsx`**: Complete clinical screening user interface:
  * File upload zone with drag-and-drop and image preview.
  * Direct asynchronous calls to `http://localhost:8000/predict` and `http://localhost:8000/explain`.
  * Dynamic severity grade pill (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR) with contextual clinical recommendations.
  * Recharts interactive bar chart illustrating class probability distributions.
  * Fundus Quality indicator displaying `quality_score`, gradability status, and any detected imaging issues.
  * Grad-CAM visualizer toggle displaying heatmaps overlaid on the retinal structure.
* **`src/app/globals.css`**: Tailwind CSS design system with custom CSS variables for dark/light themes and Google-inspired card styling.
* **`src/components/theme-provider.tsx`**: Theme provider abstraction utilizing `next-themes` for seamless dark and light mode switching.
