"""
ml/grading/evaluate.py

Evaluates the NetraPulse model on a held-out APTOS split.
Computes accuracy and quadratic weighted kappa (the Kaggle competition metric).
Logs both metrics + confusion matrix to MLflow.

Usage:
    python ml/grading/evaluate.py [--data ../data/aptos] [--split 0.2] [--experiment netrapulse]

The quadratic weighted kappa is the number we care about most:
  - 0.0 = no better than random
  - 0.8+ = clinically useful
  - The APTOS Kaggle leaderboard top scores are around 0.93
"""

from __future__ import annotations

import sys
from pathlib import Path
# Add the project root to sys.path so 'import ml...' works from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import io
import logging
import os
from pathlib import Path

import cv2
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")

DR_GRADE_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


# ---------------------------------------------------------------------------
# Inference helpers (direct import — no HTTP)
# ---------------------------------------------------------------------------

def _load_onnx_session(model_path: Path):
    import onnxruntime as ort
    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}\n"
            "Run MATLAB training + export first."
        )
    logger.info(f"Loading ONNX model: {model_path}")
    return ort.InferenceSession(str(model_path))


def predict_single(session, image_bgr: np.ndarray) -> tuple[int, list[float]]:
    """Run inference on a single BGR image. Returns (grade, confidence_list)."""
    from ml.data.preprocessing import preprocess_already_cropped

    processed = preprocess_already_cropped(image_bgr, img_size=224)
    tensor = processed.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW

    inp_name = session.get_inputs()[0].name
    outputs = session.run(None, {inp_name: tensor})
    logits = outputs[0][0]

    exp = np.exp(logits - np.max(logits))
    confidence = (exp / exp.sum()).tolist()
    grade = int(np.argmax(confidence))
    return grade, confidence


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_split(data_dir: Path, labels_csv: Path, split: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """
    Load images + labels and return the held-out validation split.

    Supports two CSV formats:
      - trainLabels.csv:         columns 'image', 'level'
      - trainLabels_cropped.csv: columns '', 'Unnamed: 0', 'image', 'level'
    """
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels not found: {labels_csv}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {data_dir}")

    df = pd.read_csv(labels_csv)

    # Normalise column names
    if 'image' not in df.columns and 'id_code' in df.columns:
        df = df.rename(columns={'id_code': 'image', 'diagnosis': 'level'})
    df = df[['image', 'level']].copy()

    # Held-out split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_val = int(len(df) * split)
    val_df = df.iloc[:n_val].copy()

    # Build full paths — try .jpeg first, then .png
    def find_path(img_id: str) -> str | None:
        for ext in ('.jpeg', '.jpg', '.png'):
            p = data_dir / f"{img_id}{ext}"
            if p.exists():
                return str(p)
        return None

    val_df['image_path'] = val_df['image'].apply(find_path)
    val_df = val_df[val_df['image_path'].notna()].reset_index(drop=True)
    logger.info(f"Validation set: {len(val_df)} images from {data_dir.name}")
    return val_df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    data_dir: Path,
    labels_csv: Path,
    model_path: Path,
    split: float = 0.2,
    experiment_name: str = "netrapulse",
) -> dict[str, float]:
    session = _load_onnx_session(model_path)
    val_df = load_dataset_split(data_dir, labels_csv, split=split)

    y_true: list[int] = []
    y_pred: list[int] = []

    for idx, row in val_df.iterrows():
        img_bgr = cv2.imread(row['image_path'])
        if img_bgr is None:
            logger.warning(f"Could not read: {row['image_path']}")
            continue
        grade, _ = predict_single(session, img_bgr)
        y_true.append(int(row['level']))
        y_pred.append(grade)

        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1}/{len(val_df)} images ...")

    if not y_true:
        raise RuntimeError("No images could be processed. Check paths and ONNX model.")

    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    logger.info(f"\nResults on {len(y_true)} images:")
    logger.info(f"  Accuracy:                {accuracy:.4f}")
    logger.info(f"  Quadratic Weighted Kappa: {kappa:.4f}")

    # Confusion matrix figure
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=DR_GRADE_LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"NetraPulse — Confusion Matrix\nkappa={kappa:.3f}, acc={accuracy:.3f}")
    plt.tight_layout()

    cm_buf = io.BytesIO()
    fig.savefig(cm_buf, format="png", dpi=120)
    cm_buf.seek(0)
    plt.close(fig)

    # MLflow logging
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_param("split", split)
        mlflow.log_param("n_samples", len(y_true))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("quadratic_weighted_kappa", kappa)
        mlflow.log_image(cm_buf, "confusion_matrix.png")
        logger.info(f"MLflow run logged to experiment '{experiment_name}'.")

    print(f"\n{'='*50}")
    print(f"  Quadratic Weighted Kappa: {kappa:.4f}")
    print(f"  Accuracy:                 {accuracy:.4f}")
    print(f"{'='*50}\n")

    return {"accuracy": accuracy, "quadratic_weighted_kappa": kappa}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    # Default: use local resized_train_cropped + trainLabels.csv
    default_data    = repo_root / "resized_train_cropped"
    default_labels  = repo_root / "trainLabels.csv"
    default_model   = repo_root / "ml" / "deploy" / "models" / "netrapulse_resnet50.onnx"

    parser = argparse.ArgumentParser(description="Evaluate NetraPulse on held-out split.")
    parser.add_argument("--data",    type=Path, default=default_data,
                        help="Folder of images (default: resized_train_cropped)")
    parser.add_argument("--labels",  type=Path, default=default_labels,
                        help="Labels CSV (default: trainLabels.csv)")
    parser.add_argument("--model",   type=Path, default=default_model)
    parser.add_argument("--split",   type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--experiment", default="netrapulse")
    args = parser.parse_args()

    evaluate(args.data, args.labels, args.model, split=args.split, experiment_name=args.experiment)


if __name__ == "__main__":
    main()
