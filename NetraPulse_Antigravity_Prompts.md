# NetraPulse — Antigravity Agent Prompts

Paste these into Antigravity roughly in order. Prompt 0 goes first, once, at the repo root — it gives every later agent the same shared context so you don't have to re-explain the architecture each time. Prompts marked **[parallel]** can run as separate agents in Manager view at the same time; the rest depend on something earlier finishing first.

---

### 0. Context-setting (paste once, at repo root)

```
This is NetraPulse, a diabetic retinopathy screening prototype. Target architecture:
MATLAB trains a ResNet50 classifier and generates Grad-CAM explanations on real fundus
images (APTOS 2019 + IDRiD datasets) → the trained network is exported to ONNX →
a FastAPI service (in /ml/deploy) loads the ONNX model and is the ONLY thing that
talks to a Next.js frontend (in /web) and a Docker-based interoperability layer
(Orthanc DICOM PACS + HAPI-FHIR, in docker-compose.yml). MATLAB never runs live
during a demo — it's training/R&D only.

Folder layout:
- /matlab: existing MATLAB scripts (preprocess_fundus.m, train_dr_classifier.m,
  extract_ma_patches.m, generate_explainability.m, NetraPulseApp.m)
- /ml: Python backend — /ml/data (preprocessing), /ml/deploy (FastAPI + ONNX model),
  /ml/explain (Grad-CAM), /ml/grading (metrics), /ml/quality, /ml/reports, /ml/robustness
- /web: Next.js 16 + React 19 frontend
- /data: real datasets (aptos/, idrid/), DVC-tracked, not committed to git

Remember this context for the tasks I give you next.
```

---

### 1. Data download + DVC tracking script **[parallel]**

```
In /ml, create a script scripts/download_data.py that:
1. Uses the kaggle CLI (assume kaggle.json is already configured) to download the
   aptos2019-blindness-detection competition into ../data/aptos and unzip it
2. Prints clear instructions for the user to manually download IDRiD from
   ieee-dataport.org (requires login, can't be scripted) into ../data/idrid
3. After both exist, runs `dvc add` on both folders and reports what was tracked
Make it safe to re-run (skip re-downloading if the zip already exists).
```

---

### 2. Canonical preprocessing module **[parallel]**

```
Create ml/data/preprocessing.py with a single function preprocess_fundus(image_bgr,
img_size=512, sigma=10) that: crops the image to its non-black content, resizes to
img_size, applies a circular mask to drop the corners, then does local-contrast
normalization by subtracting a Gaussian-blurred copy (Ben Graham's method: 
cv2.addWeighted(img, 4, blurred, -4, 128)). Add a small __main__ block that loads
one sample image from ../data/aptos/train_images, runs it through the function,
and saves a before/after comparison PNG to ml/data/preview.png so I can sanity-check
it visually. This exact function needs to be importable from both the training data
loader and the FastAPI inference service later, so keep it dependency-light (numpy + opencv only).
```

---

### 3. Point MATLAB scripts at real data

```
Open matlab/preprocess_fundus.m and matlab/train_dr_classifier.m. Currently they
likely load from generate_synthetic_fundus.m's output. Modify the data loading
section so it instead reads from ../data/aptos/train_images (using labels from
../data/aptos/train.csv, column "diagnosis", 5 classes 0-4) and ../data/idrid.
Keep the existing high-melanin color normalization and IQA logic intact — only
change the data source. Add a comment at the top of each file noting synthetic
generation is now only used as a fallback/unit-test fixture, not the primary data path.
```

---

### 4. ONNX export + load-back verification

```
Add an export section to the end of matlab/train_dr_classifier.m that calls
exportONNXNetwork(net, '../ml/deploy/models/netrapulse_resnet50.onnx') after
training completes. Then, in /ml, create scripts/verify_onnx.py that loads that
.onnx file with onnxruntime, runs a random dummy input of the correct shape through
it, and prints the output shape and a success message — just a smoke test that the
export worked and the file is loadable in Python, not a real prediction.
```

---

### 5. FastAPI /predict endpoint

```
In ml/deploy/app.py, create a FastAPI app with a POST /predict endpoint that:
accepts an uploaded image file, runs it through ml/data/preprocessing.py's
preprocess_fundus(), loads ml/deploy/models/netrapulse_resnet50.onnx with
onnxruntime, runs inference, and returns JSON with the predicted grade (0-4)
and the per-class confidence scores. Include a requirements.txt for this service
and a __main__ block or README note on how to run it with uvicorn. After it's
written, start it and test /docs in the browser with a real image from
../data/aptos/train_images to confirm it returns a sensible response.
```

---

### 6. FastAPI /explain endpoint (Grad-CAM)

```
Add a POST /explain endpoint to ml/deploy/app.py. Since Grad-CAM needs intermediate
activations that the ONNX graph doesn't expose cleanly, load a plain PyTorch/
torchvision ResNet50 with the same architecture as the MATLAB-trained model
specifically for this endpoint (weights can be a placeholder/randomly initialized
for now if the real trained weights aren't ported yet — note that clearly in a
TODO comment). Use a standard Grad-CAM implementation on the last conv block,
overlay the heatmap on the original image, and return it as a base64-encoded PNG
alongside the grade. Keep /predict on the ONNX path and only use PyTorch here.
```

---

### 7. Next.js upload + results UI **[parallel]**

```
In /web, build an upload page where a user can drag-and-drop or select a fundus
image, POST it to http://localhost:8000/predict, and display: the predicted DR
grade (0-4, with the standard labels No DR / Mild / Moderate / Severe / Proliferative),
a confidence bar chart using recharts, and — once available — the Grad-CAM heatmap
image from /explain overlaid or shown side-by-side with the original upload. Use
the existing Tailwind setup and lucide-react icons for a clean clinical look, not
a generic form. Handle loading and error states.
```

---

### 8. Orthanc → FHIR worker script **[parallel]**

```
In ml/deploy/orthanc_worker.py, write a script that polls Orthanc's REST API
(http://localhost:8042) for new studies, downloads the instance as an image,
sends it to the local FastAPI /predict endpoint, and posts the result as a FHIR
DiagnosticReport resource to HAPI-FHIR (http://localhost:8080/fhir) referencing
the study. Make the poll interval configurable and log each step to stdout so
I can watch it work during a demo. Assume docker-compose.yml already runs both
Orthanc and HAPI-FHIR on those ports.
```

---

### 9. MLflow logging + evaluation metric

```
In ml/grading, create an evaluate.py that loads a held-out split of
../data/aptos, runs each image through the /predict logic (import the function
directly, don't go over HTTP), computes accuracy AND quadratic weighted kappa
against the true "diagnosis" labels, and logs both metrics plus a confusion
matrix image to MLflow as a new run. Print the final kappa score at the end —
that's the number I care about most for this task.
```

---

### 10. End-to-end browser verification (use the browser-capable agent)

```
With the FastAPI service and Next.js dev server both running, open the Next.js
upload page in the browser, upload a real image from ../data/aptos/train_images,
and confirm the full flow works end to end: request goes out, a grade and
confidence chart render, and no console errors appear. Take a screenshot of the
final result and report back anything that's broken.
```
