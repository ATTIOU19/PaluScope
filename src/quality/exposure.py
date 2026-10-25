"""Mesure d'exposition et de surexposition (JusteAgbo05).

Entrée : image en niveaux de gris, uint8, 2D (0 = noir, 255 = blanc).
"""


def brightness(gray):
    """Luminosité moyenne : μ = (1/N) Σ xᵢ, entre 0 et 255."""
    return float(gray.mean())


def overexposure_ratio(gray, thr=250):
    """Part des pixels saturés (valeur ≥ thr), entre 0 et 1."""
    return float((gray >= thr).mean())


def underexposure_ratio(gray, thr=5):
    """Part des pixels quasi noirs (valeur ≤ thr), entre 0 et 1. Détecte aussi les bords noirs (vignette)."""
    return float((gray <= thr).mean())