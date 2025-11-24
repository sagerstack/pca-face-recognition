from typing import Tuple

import numpy as np


def flatten_images(images: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Flatten image stack to (N, H*W) and return original (H, W)."""
    if images.ndim != 3:
        raise ValueError("Expected images with shape (N, H, W)")
    n, h, w = images.shape
    return images.reshape(n, h * w), (h, w)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] for visualization."""
    img = image.astype(np.float32)
    min_v, max_v = img.min(), img.max()
    if max_v - min_v < 1e-8:
        return np.zeros_like(img)
    return (img - min_v) / (max_v - min_v)
