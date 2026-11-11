"""Extraction de patches carrés centrés sur les candidats."""
from __future__ import annotations

import cv2
import numpy as np

PATCH_SIZE = 48  # les parasites mesurent ~40x40 px


def extract_patch(image: np.ndarray, bbox: tuple, size: int = PATCH_SIZE) -> np.ndarray:
    """Patch (size, size, 3) uint8 centré sur le centre de `bbox` = (x, y, w, h).

    image : RGB uint8 (H, W, 3). Aux bords, le contenu est complété par réflexion.
    """
    assert image.ndim == 3 and image.dtype == np.uint8, "image attendue : RGB uint8 (H, W, 3)"
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = int(np.floor(cx - size / 2.0 + 0.5))
    y0 = int(np.floor(cy - size / 2.0 + 0.5))
    pad = size  # marge suffisante pour tout centre situé dans l'image
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    return np.ascontiguousarray(padded[y0 + pad : y0 + pad + size, x0 + pad : x0 + pad + size])