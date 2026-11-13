"""Normalisation de couleur d'une image entière (option paramétrable).

`none` est la référence : aucune normalisation n'est supposée meilleure a priori, le choix se
justifie par ablation (Phase 2). Entrée et sortie : RGB uint8 (H, W, 3).
Les candidats sont toujours détectés sur l'image BRUTE (réglages figés) ; la normalisation ne
s'applique qu'au calcul des features.
"""
from __future__ import annotations

import numpy as np

NORM_MODES = ("none", "gray_world", "standardize")


def _check(image_rgb: np.ndarray) -> None:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("image attendue : tableau RGB uint8 de forme (H, W, 3)")


def gray_world(image_rgb: np.ndarray) -> np.ndarray:
    """Hypothèse du monde gris : on égalise les moyennes des trois canaux."""
    _check(image_rgb)
    f = image_rgb.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1e-6)
    return np.clip(f * gain, 0, 255).round().astype(np.uint8)


def standardize(image_rgb: np.ndarray, target_mean: float = 128.0, target_std: float = 40.0) -> np.ndarray:
    """Standardisation par canal et par image, puis recadrage sur (target_mean, target_std)."""
    _check(image_rgb)
    f = image_rgb.astype(np.float32)
    flat = f.reshape(-1, 3)
    out = (f - flat.mean(axis=0)) / np.maximum(flat.std(axis=0), 1e-6) * target_std + target_mean
    return np.clip(out, 0, 255).round().astype(np.uint8)


def normalize_image(image_rgb: np.ndarray, mode: str = "none") -> np.ndarray:
    _check(image_rgb)
    if mode == "none":
        return image_rgb
    if mode == "gray_world":
        return gray_world(image_rgb)
    if mode == "standardize":
        return standardize(image_rgb)
    raise ValueError(f"mode de normalisation inconnu : {mode!r} (attendu : {', '.join(NORM_MODES)})")