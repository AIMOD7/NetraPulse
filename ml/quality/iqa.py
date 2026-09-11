"""
ml/quality/iqa.py

Fundus Image Quality Assessment (IQA) for Diabetic Retinopathy screening.

Evaluates uploaded retinal images prior to classification:
  - Sharpness / Blur: Laplacian variance on retinal foreground.
  - Illumination / Exposure: Mean luminance of retinal area (detecting underexposed/overexposed).
  - Contrast: Standard deviation of pixel intensities within retinal mask.
  - Retinal Disc Coverage: Non-black foreground coverage check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image


@dataclass
class IQAResult:
    is_gradable: bool
    quality_score: float  # 0.0 to 1.0
    blur_score: float     # Higher = sharper
    illumination_score: float # 0.0 to 1.0 (centered around optimal exposure)
    contrast_score: float     # Standard deviation of foreground
    issues: list[str] = field(default_factory=list)


def _get_retinal_mask(img_rgb: np.ndarray, tol: int = 15) -> tuple[np.ndarray, float]:
    """Segment the retinal area from the black background."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > tol).astype(np.uint8)
    foreground_ratio = float(np.mean(mask))
    return mask, foreground_ratio


def assess(
    image: Union[np.ndarray, bytes, str, Path, Image.Image],
    blur_threshold: float = 60.0,
    min_illumination: float = 40.0,
    max_illumination: float = 210.0,
    min_contrast: float = 20.0,
) -> IQAResult:
    """
    Assess quality of a fundus photograph for clinical DR grading.

    Args:
        image: RGB or BGR numpy array, image file bytes, filepath, or PIL Image.
        blur_threshold: Minimum Laplacian variance to be considered sharp.
        min_illumination: Minimum mean intensity in retinal area.
        max_illumination: Maximum mean intensity in retinal area.
        min_contrast: Minimum standard deviation of pixels in retinal area.

    Returns:
        IQAResult with gradability flag, overall score, and detected issues.
    """
    if isinstance(image, (str, Path)):
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise ValueError(f"Could not load image from {image}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, bytes):
        nparr = np.frombuffer(image, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image bytes.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2RGB)
    elif isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        # Assume RGB if 3 channels
        img_rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask, fg_ratio = _get_retinal_mask(img_rgb)

    issues: list[str] = []

    # 1. Retinal field coverage check
    if fg_ratio < 0.15:
        issues.append("insufficient_retinal_field")

    # 2. Sharpness / Blur (Laplacian variance on masked foreground)
    # Compute on the bounding box of non-black area to avoid artificial border edges
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Margin inward to stay inside retinal disc
        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        roi_gray = gray[y + pad_y : y + h - pad_y, x + pad_x : x + w - pad_x]
        if roi_gray.size > 0:
            lap_var = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
        else:
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    else:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if lap_var < blur_threshold:
        issues.append("image_blurred")

    # 3. Illumination / Exposure
    fg_pixels = gray[mask > 0]
    if len(fg_pixels) > 0:
        mean_lum = float(np.mean(fg_pixels))
        contrast_std = float(np.std(fg_pixels))
    else:
        mean_lum = float(np.mean(gray))
        contrast_std = float(np.std(gray))

    if mean_lum < min_illumination:
        issues.append("underexposed")
    elif mean_lum > max_illumination:
        issues.append("overexposed")

    # 4. Contrast
    if contrast_std < min_contrast:
        issues.append("low_contrast")

    # Composite Quality Score [0.0, 1.0]
    # Sharpness sub-score (sigmoid-like scaling around blur_threshold)
    sharp_norm = min(1.0, lap_var / (blur_threshold * 2.5))
    # Illumination sub-score (optimal around 110-140)
    opt_illum = 125.0
    illum_norm = max(0.0, 1.0 - abs(mean_lum - opt_illum) / 100.0)
    # Contrast sub-score
    contrast_norm = min(1.0, contrast_std / 50.0)

    quality_score = round(float(0.4 * sharp_norm + 0.35 * illum_norm + 0.25 * contrast_norm), 3)
    is_gradable = len(issues) == 0

    return IQAResult(
        is_gradable=is_gradable,
        quality_score=quality_score,
        blur_score=round(lap_var, 2),
        illumination_score=round(illum_norm, 3),
        contrast_score=round(contrast_std, 2),
        issues=issues,
    )
