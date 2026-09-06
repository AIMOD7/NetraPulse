"""
ml/scripts/download_data.py

Downloads APTOS 2019 and provides instructions for IDRiD.
Safe to re-run — skips download if zip already exists.

Usage:
    python ml/scripts/download_data.py

Requirements:
    pip install kaggle dvc
    Place kaggle.json in %USERPROFILE%\.kaggle\kaggle.json
    Accept the APTOS competition rules at:
        https://www.kaggle.com/competitions/aptos2019-blindness-detection
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root, one level up from ml/)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
APTOS_DIR = DATA_DIR / "aptos"
IDRID_DIR = DATA_DIR / "idrid"
APTOS_ZIP = APTOS_DIR / "aptos2019-blindness-detection.zip"
APTOS_COMPETITION = "aptos2019-blindness-detection"


def _run(cmd: list[str], **kwargs) -> int:
    """Run a subprocess command and return its exit code."""
    print(f"[RUN] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    return result.returncode


def download_aptos() -> bool:
    """Download and unzip APTOS 2019. Returns True if successful."""
    APTOS_DIR.mkdir(parents=True, exist_ok=True)

    if APTOS_ZIP.exists():
        print(f"[SKIP] APTOS zip already exists: {APTOS_ZIP}")
    else:
        print("[INFO] Downloading APTOS 2019 via Kaggle CLI...")
        print("[INFO] Make sure you have accepted the competition rules at:")
        print("       https://www.kaggle.com/competitions/aptos2019-blindness-detection\n")
        rc = _run(
            [
                sys.executable, "-m", "kaggle",
                "competitions", "download",
                "-c", APTOS_COMPETITION,
                "-p", str(APTOS_DIR),
            ]
        )
        if rc != 0:
            print("\n[ERROR] Kaggle download failed. Common causes:")
            print("  1. kaggle.json not found — run:")
            print(f"       copy %USERPROFILE%\\Downloads\\kaggle.json %USERPROFILE%\\.kaggle\\kaggle.json")
            print("  2. Competition rules not accepted — visit the Kaggle competition page.")
            return False

    # Unzip
    train_images_dir = APTOS_DIR / "train_images"
    if train_images_dir.exists() and any(train_images_dir.iterdir()):
        print(f"[SKIP] APTOS already extracted at: {train_images_dir}")
    else:
        print(f"[INFO] Extracting {APTOS_ZIP} ...")
        with zipfile.ZipFile(APTOS_ZIP, "r") as zf:
            zf.extractall(APTOS_DIR)
        print(f"[OK] Extracted to: {APTOS_DIR}")

    n_images = len(list((APTOS_DIR / "train_images").glob("*.png")))
    print(f"[OK] APTOS — {n_images} training images found.")
    return True


def print_idrid_instructions() -> None:
    """Print manual download instructions for IDRiD."""
    print()
    print("=" * 70)
    print("  IDRiD — MANUAL DOWNLOAD REQUIRED")
    print("=" * 70)
    print()
    print("  IDRiD (Indian Diabetic Retinopathy Image Dataset) requires a")
    print("  free IEEE DataPort account. It cannot be scripted.")
    print()
    print("  Steps:")
    print("  1. Go to: https://ieee-dataport.org/open-access/")
    print("            indian-diabetic-retinopathy-image-dataset-idrid")
    print("  2. Sign in (or create a free account).")
    print("  3. Click Download and extract the zip.")
    print(f"  4. Place contents in: {IDRID_DIR}")
    print()
    print("  Expected structure after extraction:")
    print("    data/idrid/")
    print("      A. Segmentation/")
    print("      B. Disease Grading/")
    print("        1. Original Images/")
    print("          a. Training Set/   ← fundus images")
    print("          b. Testing Set/")
    print("        2. Groundtruths/")
    print("          a. a. IDRiD_Disease Grading_Training Labels.csv")
    print("=" * 70)
    print()


def track_with_dvc() -> None:
    """Run dvc add on both data folders if dvc is available."""
    aptos_ok = APTOS_DIR.exists() and any(APTOS_DIR.rglob("*.png"))
    idrid_ok = IDRID_DIR.exists() and any(IDRID_DIR.iterdir())

    if not aptos_ok:
        print("[SKIP] DVC tracking: APTOS not yet downloaded.")
        return
    if not idrid_ok:
        print("[SKIP] DVC tracking: IDRiD not yet present — add it first.")
        return

    print("[INFO] Running: dvc add data/aptos data/idrid ...")
    rc = _run(["dvc", "add", str(APTOS_DIR), str(IDRID_DIR)], cwd=str(REPO_ROOT))
    if rc == 0:
        print("[OK] DVC tracking complete. Run:")
        print("     git add *.dvc .gitignore && git commit -m 'Track data with DVC'")
    else:
        print("[WARN] dvc add failed — is DVC installed? (pip install dvc)")


def main() -> None:
    print("NetraPulse — Data Setup Script")
    print("=" * 70)

    aptos_ok = download_aptos()
    print_idrid_instructions()

    if aptos_ok:
        track_with_dvc()


if __name__ == "__main__":
    main()
