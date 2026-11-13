"""Features géométriques par candidat (préfixe `geo_`).

Entrée : les candidats d'UNE image et le masque nettoyé qui les a produits
(`run_pipeline_steps(image, params)["clean_mask"]`). On retrouve la composante de chaque
candidat par (bbox, aire), ce qui évite de modifier `src/candidates/extract.py`.
"""
from __future__ import annotations

import cv2
import numpy as np

GEO_FEATURES = (
    "geo_area", "geo_perimeter", "geo_circularity", "geo_width", "geo_height",
    "geo_aspect_ratio", "geo_extent", "geo_solidity", "geo_eccentricity",
)


def _shape_features(cand: dict, sub: np.ndarray) -> dict:
    """sub : masque binaire uint8 de la composante, recadré sur sa bbox."""
    h, w = sub.shape
    area = float(sub.sum())
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=len)
    canvas = np.zeros_like(sub)
    cv2.fillConvexPoly(canvas, cv2.convexHull(cnt), 1)
    hull_px = float(canvas.sum())
    solidity = area / hull_px if hull_px > 0 else 1.0

    m = cv2.moments(sub, binaryImage=True)
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    common = float(np.sqrt(((mu20 - mu02) / 2.0) ** 2 + mu11**2))
    l1, l2 = (mu20 + mu02) / 2.0 + common, (mu20 + mu02) / 2.0 - common
    eccentricity = float(np.sqrt(max(0.0, 1.0 - l2 / l1))) if l1 > 1e-12 else 0.0

    return {
        "geo_area": float(cand["area"]),
        "geo_perimeter": float(cand["perimeter"]),
        "geo_circularity": float(cand["circularity"]),
        "geo_width": float(w),
        "geo_height": float(h),
        "geo_aspect_ratio": float(min(w, h) / max(w, h)),  # dans ]0, 1], indépendant de l'orientation
        "geo_extent": area / float(w * h),
        "geo_solidity": float(min(1.0, solidity)),
        "geo_eccentricity": eccentricity,
    }


def candidate_masks(candidates: list[dict], clean_mask: np.ndarray) -> list[np.ndarray]:
    """Masque binaire uint8 (recadré sur la bbox) de chaque candidat, dans le même ordre."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    index: dict[tuple, list[int]] = {}
    for lab in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[lab])
        index.setdefault((x, y, w, h, a), []).append(lab)

    masks = []
    for c in candidates:
        x, y, w, h = (int(v) for v in c["bbox"])
        labs = index.get((x, y, w, h, int(round(c["area"]))))
        if not labs:
            raise ValueError(
                f"candidat {c.get('id')} introuvable dans le masque : utiliser le `clean_mask` "
                "de run_pipeline_steps avec les MÊMES paramètres que les candidats"
            )
        lab = labs.pop(0)  # les candidats sont produits par étiquette croissante
        masks.append((labels[y : y + h, x : x + w] == lab).astype(np.uint8))
    return masks


def geometry_features(candidates: list[dict], clean_mask: np.ndarray) -> list[dict]:
    """Une ligne de features `geo_*` par candidat, dans le même ordre que `candidates`."""
    return [_shape_features(c, m) for c, m in zip(candidates, candidate_masks(candidates, clean_mask))]