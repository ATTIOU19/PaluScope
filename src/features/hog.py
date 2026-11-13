"""HOG (`hog_`) sur un patch en niveaux de gris 48x48 : 9 orientations, cellules de 12 px,
blocs 2x2 -> 3x3 blocs x 4 cellules x 9 = 324 valeurs. Famille volumineuse : évaluée par ablation."""
from __future__ import annotations

import numpy as np
from skimage.feature import hog

N_HOG = 324


def hog_features(gray: np.ndarray) -> dict:
    v = hog(gray, orientations=9, pixels_per_cell=(12, 12), cells_per_block=(2, 2),
            block_norm="L2-Hys", feature_vector=True)
    assert v.shape[0] == N_HOG, f"HOG attendu sur un patch 48x48 ({N_HOG} valeurs), obtenu {v.shape[0]}"
    return {f"hog_{k:03d}": float(x) for k, x in enumerate(v)}