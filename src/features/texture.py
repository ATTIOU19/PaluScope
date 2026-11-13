"""Gradients (`grd_`) et texture (`tex_`) sur un patch en niveaux de gris (uint8, 48x48)."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

LBP_POINTS, LBP_RADIUS = 8, 1
LBP_BINS = LBP_POINTS + 2  # motifs « uniformes » : 10 classes
GLCM_LEVELS = 32
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")


def gradient_features(gray: np.ndarray) -> dict:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return {
        "grd_sobel_mean": float(mag.mean()),
        "grd_sobel_std": float(mag.std()),
        "grd_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
    }


def texture_features(gray: np.ndarray) -> dict:
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform").astype(int)
    hist = np.bincount(lbp.ravel(), minlength=LBP_BINS)[:LBP_BINS] / lbp.size
    out = {f"tex_lbp{k}": float(v) for k, v in enumerate(hist)}

    q = (gray // (256 // GLCM_LEVELS)).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    for p in GLCM_PROPS:
        # corrélation indéfinie (NaN) sur un patch uniforme : on la remplace par 0
        out[f"tex_glcm_{p}"] = float(np.nan_to_num(np.mean(graycoprops(glcm, p))))
    return out