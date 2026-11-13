"""Features de couleur (préfixe `col_`) : le candidat comparé à son fond local.

Pour chaque canal : moyenne et écart-type dans le masque du candidat, dans un anneau de fond
autour, leur différence, et (canaux positifs) leur rapport. Histogramme à 8 classes sur R, G, B.
La teinte H est circulaire (les roses/rouges chevauchent 0/179) : on la représente par
cos(H) et sin(H) au lieu de H.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.color import to_gray, to_hsv

CHANNELS = ("r", "g", "b", "gray", "s", "v", "hcos", "hsin")
RATIO_CHANNELS = ("r", "g", "b", "gray", "s", "v")
HIST_CHANNELS = ("r", "g", "b")
HIST_BINS = 8
MARGIN = 8  # pixels autour de la bbox pour l'anneau de fond
_K_IN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_K_OUT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))


def _channel_arrays(win_rgb: np.ndarray) -> dict:
    hsv = to_hsv(win_rgb)
    ang = hsv[..., 0].astype(np.float32) * (2.0 * np.pi / 180.0)
    return {
        "r": win_rgb[..., 0], "g": win_rgb[..., 1], "b": win_rgb[..., 2],
        "gray": to_gray(win_rgb), "s": hsv[..., 1], "v": hsv[..., 2],
        "hcos": np.cos(ang), "hsin": np.sin(ang),
    }


def color_features(image_rgb: np.ndarray, bbox: tuple, mask: np.ndarray) -> dict:
    """image_rgb : RGB uint8 (éventuellement normalisée) ; bbox (x, y, w, h) ; mask (h, w) 0/1."""
    H, W = image_rgb.shape[:2]
    x, y, w, h = (int(v) for v in bbox)
    x0, y0 = max(0, x - MARGIN), max(0, y - MARGIN)
    x1, y1 = min(W, x + w + MARGIN), min(H, y + h + MARGIN)
    win = image_rgb[y0:y1, x0:x1]

    m = np.zeros(win.shape[:2], np.uint8)
    m[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = mask
    inside = m > 0
    ring = (cv2.dilate(m, _K_OUT) > 0) & ~(cv2.dilate(m, _K_IN) > 0)
    if not ring.any():
        ring = ~inside
    if not ring.any():
        ring = np.ones_like(inside)

    arrays = _channel_arrays(win)
    out: dict = {}
    means: dict = {}
    for c in CHANNELS:
        a = arrays[c].astype(np.float32)
        mi, ri = a[inside], a[ring]
        out[f"col_{c}_mean"], out[f"col_{c}_std"] = float(mi.mean()), float(mi.std())
        out[f"col_{c}_ring_mean"], out[f"col_{c}_ring_std"] = float(ri.mean()), float(ri.std())
        out[f"col_{c}_diff"] = out[f"col_{c}_mean"] - out[f"col_{c}_ring_mean"]
        means[c] = (out[f"col_{c}_mean"], out[f"col_{c}_ring_mean"])
    for c in RATIO_CHANNELS:
        out[f"col_{c}_ratio"] = (means[c][0] + 1.0) / (means[c][1] + 1.0)
    for c in HIST_CHANNELS:
        hist, _ = np.histogram(arrays[c][inside], bins=HIST_BINS, range=(0, 256))
        for k, v in enumerate(hist / max(1, inside.sum())):
            out[f"col_{c}_hist{k}"] = float(v)
    return out