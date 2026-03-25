"""
utils.py — Shared image-processing helpers.

Import with:
    from utils import enhance_image, gentle_sharpen, laplacian_variance
"""

import cv2
import numpy as np


def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    CLAHE enhancement on the L channel of LAB colour space.

    WHEN TO USE:
        Only on low-contrast faces: np.std(face) < 50
        Do NOT apply universally — it shifts the pixel distribution for
        normal-contrast images and can hurt model accuracy.

    Args:
        image: RGB uint8 numpy array, any size.
    Returns:
        Enhanced RGB uint8 numpy array, same shape.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def gentle_sharpen(image: np.ndarray) -> np.ndarray:
    """
    Unsharp mask sharpening — lifts fine texture detail.

    Uses cv2.addWeighted which clips to [0, 255] safely.
    A raw filter2D with a centre=9 kernel overflows uint8 before clipping.

    WHEN TO USE:
        On normal-contrast faces: np.std(face) >= 50
        Do NOT apply to already-blurry crops (makes ringing artefacts).

    Args:
        image: RGB uint8 numpy array, any size.
    Returns:
        Sharpened RGB uint8 numpy array, same shape.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def laplacian_variance(rgb_image: np.ndarray) -> float:
    """
    Blur score via Laplacian variance on grayscale.

    IMPORTANT: always convert to grayscale first — Laplacian on a 3-channel
    RGB array gives inconsistent variance values across channels.

    Args:
        rgb_image: RGB uint8 numpy array.
    Returns:
        Float variance. Lower = blurrier. Typical threshold: 10–40.
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()