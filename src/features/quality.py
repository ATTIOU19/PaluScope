"""Variables de contexte au niveau de l'IMAGE (`qual_`), issues du Quality Analyzer.

Même valeur pour tous les candidats d'une image. Exclues des modèles par défaut : elles risquent
d'encoder l'appareil ou la lame plutôt que le parasite (ablation explicite en Phase 2).
"""
from __future__ import annotations

import numpy as np

QUAL_KEYS = ("sharpness", "brightness", "contrast", "overexposure_ratio", "underexposure_ratio")


def _analyze(image_rgb: np.ndarray) -> dict:
    try:
        from src.quality.analyzer import analyze_image
    except ImportError:
        from src.quality import analyze_image
    return analyze_image(image_rgb)


def quality_features(image_rgb: np.ndarray) -> dict:
    """image_rgb : image BRUTE (la qualité décrit l'acquisition, pas l'image normalisée)."""
    q = _analyze(image_rgb)
    return {f"qual_{k}": float(q[k]) for k in QUAL_KEYS}