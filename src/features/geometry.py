"""Features géométriques d'un candidat (famille geo_), bloc P1-B, Membre 3.

Entrée : un candidat obtenu avec extract_candidates(image, params, with_contour=True).
Sortie : un dictionnaire {nom: nombre}, mêmes clés dans le même ordre pour tous les candidats,
jamais de NaN ni d'infini.
"""
import math

import cv2
import numpy as np

GEO_KEYS = (
    "geo_area", "geo_perimeter", "geo_circularity", "geo_width", "geo_height",
    "geo_aspect_ratio", "geo_extent", "geo_solidity", "geo_eccentricity",
)


def _contour_cv(candidate):
    """Remet le contour au format attendu par OpenCV : tableau (N, 1, 2) d'entiers."""
    return np.asarray(candidate["contour"]).reshape(-1, 1, 2).astype(np.int32)


def solidity(contour):
    """Aire de la forme / aire de son enveloppe convexe (1 = forme pleine, sans creux)."""
    enveloppe = cv2.convexHull(contour)
    aire_enveloppe = cv2.contourArea(enveloppe)
    if aire_enveloppe <= 0:          # forme trop fine pour avoir une aire : valeur de repli
        return 1.0
    return min(1.0, cv2.contourArea(contour) / aire_enveloppe)


def eccentricity(contour):
    """0 = forme ronde, proche de 1 = forme très allongée (calculée avec les moments)."""
    m = cv2.moments(contour)
    if m["m00"] <= 0:                # forme dégénérée : valeur de repli
        return 0.0
    a = m["mu20"] / m["m00"]
    b = m["mu11"] / m["m00"]
    c = m["mu02"] / m["m00"]
    racine = math.sqrt(((a - c) / 2) ** 2 + b ** 2)
    grand = (a + c) / 2 + racine     # variance le long du grand axe
    petit = (a + c) / 2 - racine     # variance le long du petit axe
    if grand <= 0:
        return 0.0
    return math.sqrt(max(0.0, 1.0 - petit / grand))


def geometry_features(candidate):
    """Calcule les 9 features géométriques d'un candidat."""
    if "contour" not in candidate:
        raise ValueError("candidat sans contour : appeler extract_candidates(..., with_contour=True)")
    x, y, w, h = candidate["bbox"]
    contour = _contour_cv(candidate)
    return {
        "geo_area": float(candidate["area"]),
        "geo_perimeter": float(candidate["perimeter"]),
        "geo_circularity": float(candidate["circularity"]),
        "geo_width": float(w),
        "geo_height": float(h),
        "geo_aspect_ratio": float(w) / float(h),
        "geo_extent": float(candidate["area"]) / float(w * h),
        "geo_solidity": float(solidity(contour)),
        "geo_eccentricity": float(eccentricity(contour)),
    }