import math

import numpy as np
import pytest

from src.candidates.extract import CandidateParams, extract_candidates, measure_regions
from src.features.geometry import GEO_KEYS, geometry_features

TOUT = CandidateParams(min_area=1, max_area=10000, min_circularity=0.0)


def masque_avec(forme):
    """Masque noir 60x60 avec une forme blanche ; forme = (ligne_debut, ligne_fin, col_debut, col_fin)."""
    masque = np.zeros((60, 60), dtype=np.uint8)
    l0, l1, c0, c1 = forme
    masque[l0:l1, c0:c1] = 255
    return masque


def un_candidat(masque):
    regions = measure_regions(masque, TOUT, with_contour=True)
    assert len(regions) == 1
    return regions[0]


def test_carre_10_sur_10():
    f = geometry_features(un_candidat(masque_avec((10, 20, 10, 20))))
    assert f["geo_area"] == 100 and f["geo_width"] == 10 and f["geo_height"] == 10
    assert f["geo_aspect_ratio"] == 1.0
    assert f["geo_extent"] == 1.0
    assert f["geo_solidity"] == pytest.approx(1.0)
    assert f["geo_eccentricity"] == pytest.approx(0.0, abs=1e-9)


def test_barre_20_sur_4_est_tres_allongee():
    f = geometry_features(un_candidat(masque_avec((10, 14, 10, 30))))   # 4 lignes, 20 colonnes
    assert f["geo_area"] == 80
    assert f["geo_aspect_ratio"] == 5.0
    assert f["geo_extent"] == 1.0
    assert f["geo_eccentricity"] == pytest.approx(math.sqrt(1 - 9 / 361), abs=1e-6)


def test_forme_en_L_a_un_creux():
    masque = masque_avec((10, 30, 10, 30))      # carré 20 x 20
    masque[20:30, 20:30] = 0                    # on enlève un coin 10 x 10
    f = geometry_features(un_candidat(masque))
    assert f["geo_area"] == 300
    assert f["geo_extent"] == pytest.approx(0.75)
    assert f["geo_solidity"] < 0.95


def test_le_contour_est_optionnel_et_dans_la_boite():
    img = np.full((80, 80, 3), (215, 170, 165), dtype=np.uint8)
    img[40:46, 30:36] = (70, 30, 90)            # petite tache sombre
    sans = extract_candidates(img)
    avec = extract_candidates(img, with_contour=True)
    assert len(sans) == len(avec) == 1
    assert "contour" not in sans[0]
    x, y, w, h = avec[0]["bbox"]
    pts = avec[0]["contour"]
    assert pts.shape[1] == 2
    assert pts[:, 0].min() >= x and pts[:, 0].max() <= x + w - 1
    assert pts[:, 1].min() >= y and pts[:, 1].max() <= y + h - 1


def test_cles_stables_valeurs_finies_et_erreur_sans_contour():
    c = un_candidat(masque_avec((10, 20, 10, 20)))
    f = geometry_features(c)
    assert tuple(f) == GEO_KEYS
    assert all(math.isfinite(v) for v in f.values())
    sans_contour = {k: v for k, v in c.items() if k != "contour"}
    with pytest.raises(ValueError):
        geometry_features(sans_contour)