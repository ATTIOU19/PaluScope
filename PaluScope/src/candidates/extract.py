"""Extraction de régions candidates (Membre 3).

Pipeline : Image -> Gris/HSV -> Seuillage -> Morphologie
           -> Composantes connexes -> Contours -> Mesures géométriques
"""


def extract_candidates(image) -> list:
    """Retourne une liste de régions candidates (voir docs/conventions.md)."""
    raise NotImplementedError("À implémenter (Membre 3).")
