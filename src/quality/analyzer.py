"""Point d'entrée du Quality Analyzer (Membre 2).

Le format du retour est défini dans docs/conventions.md.
Entrée : image RGB uint8 (tableau NumPy H×W×3) ou chemin vers un fichier image.
"""
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np

from . import blur, color, contrast, exposure

# Valeurs par défaut. Un seuil à None n'est pas encore calibré : le drapeau correspondant vaut alors None.
# Calibration : notebooks/phase_0/02_quality_analyzer.ipynb (sections 5 à 7).
DEFAULT_CONFIG = {
    "ref_size": 750,               # côté long (px) auquel on ramène l'image avant de mesurer la netteté
    "sharpness_method": "laplacian",
    "contrast_method": "percentile",
    "sharpness_min": 7.381,         # en dessous : image floue 
    "contrast_min": 23,          # en dessous : contraste insuffisant 
    "brightness_min": 80.0,        # en dessous : trop sombre 
    "brightness_max": 200.0,       # au-dessus : trop claire 
    "overexposure_thr": 250,       # un pixel de gris ≥ ce niveau est saturé
    "overexposure_max": 0.05,      # au-dessus de 5 % de pixels saturés : surexposée 
    "underexposure_thr": 5,        # un pixel de gris ≤ ce niveau est quasi noir
    "underexposure_max": 0.005,    # au-dessus de 0,5 % de pixels quasi noirs : bords noirs probables
}


def load_image(path):
    """Lit un fichier image et retourne un tableau RGB uint8 (OpenCV lit en BGR)."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Image illisible ou introuvable : {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _check_rgb(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("L'image doit être un tableau NumPy RGB uint8 de forme (H, W, 3).")
    return image


def _resize_long_side(gray, size):
    """Ramène le côté long à `size` px (la netteté dépend de la résolution). Sans effet si déjà à cette taille."""
    long_side = max(gray.shape)
    if long_side == size:
        return gray
    scale = size / long_side
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)


def _below(value, threshold):
    return None if threshold is None else bool(value < threshold)


def analyze_image(image, config=None):
    """Analyse la qualité technique d'une image de frottis et retourne des indicateurs simples.

    Retour : dict avec sharpness, brightness, contrast, overexposure_ratio, underexposure_ratio,
    color_statistics (dict) et flags (dict de booléens ; None = seuil non calibré).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if isinstance(image, (str, Path)):
        image = load_image(image)
    rgb = _check_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    sharpness = blur.sharpness(_resize_long_side(gray, cfg["ref_size"]), cfg["sharpness_method"])
    brightness = exposure.brightness(gray)
    contrast_value = contrast.contrast(gray, cfg["contrast_method"])
    over = exposure.overexposure_ratio(gray, cfg["overexposure_thr"])
    under = exposure.underexposure_ratio(gray, cfg["underexposure_thr"])

    return {
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast_value,
        "overexposure_ratio": over,
        "underexposure_ratio": under,
        "color_statistics": color.color_statistics(rgb),
        "flags": {
            "blurry": _below(sharpness, cfg["sharpness_min"]),
            "low_contrast": _below(contrast_value, cfg["contrast_min"]),
            "too_dark": bool(brightness < cfg["brightness_min"]),
            "too_bright": bool(brightness > cfg["brightness_max"]),
            "overexposed": bool(over > cfg["overexposure_max"]),
            "dark_borders": bool(under > cfg["underexposure_max"]),
        },
    }