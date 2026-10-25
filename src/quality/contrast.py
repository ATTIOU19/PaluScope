"""Mesure du contraste (JusteAgbo05).

Entrée : image en niveaux de gris, uint8, 2D.
"""
import numpy as np


def contrast_percentile(gray, low=1, high=99):
    """Étendue des niveaux de gris entre deux centiles : p99 − p1 (robuste à quelques pixels extrêmes)."""
    p_low, p_high = np.percentile(gray, [low, high])
    return float(p_high - p_low)


def contrast_std(gray):
    """Écart-type des niveaux de gris : σ = √((1/N) Σ (xᵢ − μ)²)."""
    return float(gray.std())


METHODS = {"percentile": contrast_percentile, "std": contrast_std}


def contrast(gray, method="percentile"):
    """Contraste d'une image en niveaux de gris selon la méthode choisie (voir METHODS)."""
    if method not in METHODS:
        raise ValueError(f"Méthode de contraste inconnue : {method!r} (choix : {sorted(METHODS)})")
    return METHODS[method](gray)