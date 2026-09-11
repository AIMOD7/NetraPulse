"""
ml/data/preprocessing.py

Canonical fundus image preprocessing used identically at:
  - Training time (loading from resized_train_cropped/)
  - FastAPI inference time (/predict endpoint, raw uploaded images)

Two entry points:
  preprocess_fundus(image_bgr)          -- full pipeline for raw images
  preprocess_already_cropped(image_bgr) -- fast path for pre-cropped images
                                           (resized_train_cropped are 1024x1024
                                            already cropped — skip _crop_to_content)

Dependencies: numpy, opencv-python (no torch, no heavy deps)
"""

import cv2
import numpy as np


def preprocess_fundus(
    image_bgr: np.ndarray,
    img_size: int = 512,
    sigma: float = 10,
) -> np.ndarray:
    """
    Preprocess a fundus image for DR classification.

    Steps:
      1. BGR → RGB
      2. Crop to non-black content (removes dark borders)
      3. Resize to img_size × img_size
      4. Apply circular mask (drop corners outside the retinal disc)
      5. Local-contrast normalization — Ben Graham's method:
         cv2.addWeighted(img, 4, GaussianBlur(img), -4, 128)

    Args:
        image_bgr:  Raw image as a NumPy array in BGR order (as loaded by cv2.imread).
        img_size:   Output spatial resolution (default 512 — matches ResNet50 input).
        sigma:      Gaussian blur sigma for local-contrast normalization (default 10).

    Returns:
        Preprocessed uint8 RGB image of shape (img_size, img_size, 3).
    """
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


def _crop_to_content(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Crop the image to its non-black bounding box.

    Args:
        img: RGB image as a NumPy array.
        tol: Pixel intensity threshold below which a pixel is considered black.

    Returns:
        Cropped image. Returns the original if no non-black pixels found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def preprocess_already_cropped(
    image_bgr: np.ndarray,
    img_size: int = 512,
    sigma: float = 10,
) -> np.ndarray:
    """
    Fast preprocessing path for images that are already cropped and square
    (e.g. resized_train_cropped/*.jpeg, which are 1024×1024).

    Skips _crop_to_content — goes straight to resize → circular mask → Ben Graham.
    This is what training uses so that the model is trained on the same
    transform that inference applies to already-clean images.

    Args:
        image_bgr:  Pre-cropped uint8 BGR image (e.g. 1024×1024).
        img_size:   Output resolution (default 512).
        sigma:      Gaussian blur sigma (default 10).

    Returns:
        Preprocessed uint8 RGB image of shape (img_size, img_size, 3).
    """
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=mask)

    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    return img


# ---------------------------------------------------------------------------
# Quick sanity-check — run with:  python ml/data/preprocessing.py
# Loads one APTOS sample, saves a before/after PNG to ml/data/preview.png
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    # Try to find a sample image — prefer resized_train_cropped (local data)
    local_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "resized_train_cropped"
    )
    aptos_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "aptos", "train_images"
    )
    local_dir = os.path.normpath(local_dir)
    aptos_dir = os.path.normpath(aptos_dir)

    if os.path.isdir(local_dir):
        sample_dir = local_dir
        use_fast_path = True
        print(f"[INFO] Using local resized_train_cropped: {sample_dir}")
    elif os.path.isdir(aptos_dir):
        sample_dir = aptos_dir
        use_fast_path = False
        print(f"[INFO] Using APTOS train_images: {sample_dir}")
    else:
        print(
            f"[ERROR] No image directory found.\n"
            f"        Tried: {local_dir}\n"
            f"        Tried: {aptos_dir}"
        )
        sys.exit(1)

    # Pick the first .jpeg / .png we find
    sample_path = None
    for fname in os.listdir(sample_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            sample_path = os.path.join(sample_dir, fname)
            break

    if sample_path is None:
        print("[ERROR] No image files found in:", sample_dir)
        sys.exit(1)

    print(f"[INFO] Using sample image: {sample_path}")

    original_bgr = cv2.imread(sample_path)
    if original_bgr is None:
        print("[ERROR] cv2.imread failed — is opencv-python installed?")
        sys.exit(1)

    if use_fast_path:
        processed = preprocess_already_cropped(original_bgr)
    else:
        processed = preprocess_fundus(original_bgr)

    # Save side-by-side comparison
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    original_resized = cv2.resize(original_rgb, (512, 512))

    comparison = np.concatenate([original_resized, processed], axis=1)
    out_path = os.path.join(os.path.dirname(__file__), "preview.png")
    cv2.imwrite(out_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"[OK] Before/after comparison saved to: {out_path}")
    print(f"     Input shape:  {original_bgr.shape}")
    print(f"     Output shape: {processed.shape}")
