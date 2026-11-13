import cv2
import numpy as np
import pytest

from src.candidates.extract import CandidateParams, measure_regions, run_pipeline_steps
from src.features.geometry import GEO_FEATURES, geometry_features

OPEN = CandidateParams(min_area=1.0, max_area=10_000.0, min_circularity=0.0)


def _features(mask):
    cands = measure_regions(mask, OPEN)
    return cands, geometry_features(cands, mask)


def _mask():
    return np.zeros((80, 80), np.uint8)


def test_circle_is_round_and_solid():
    m = _mask()
    cv2.circle(m, (40, 40), 8, 255, -1)
    _, (f,) = _features(m)
    assert set(f) == set(GEO_FEATURES)
    assert f["geo_eccentricity"] < 0.1 and f["geo_solidity"] > 0.9 and f["geo_aspect_ratio"] > 0.9


def test_elongated_bar_is_eccentric_with_low_aspect_ratio():
    m = _mask()
    cv2.rectangle(m, (10, 40), (49, 43), 255, -1)  # 40 x 4
    _, (f,) = _features(m)
    assert f["geo_eccentricity"] > 0.95 and f["geo_aspect_ratio"] < 0.15 and f["geo_extent"] > 0.95


def test_l_shape_has_low_solidity():
    m = _mask()
    cv2.rectangle(m, (10, 10), (39, 15), 255, -1)
    cv2.rectangle(m, (10, 10), (15, 39), 255, -1)
    _, (f,) = _features(m)
    assert f["geo_solidity"] < 0.75


def test_values_are_finite_and_candidate_fields_are_copied():
    m = _mask()
    cv2.circle(m, (20, 20), 5, 255, -1)
    m[60:63, 60:63] = 255  # très petite composante (3x3)
    cands, feats = _features(m)
    assert len(feats) == len(cands) == 2
    for c, f in zip(cands, feats):
        assert all(np.isfinite(v) for v in f.values())
        assert f["geo_area"] == c["area"] and f["geo_perimeter"] == c["perimeter"]
        assert 0.0 <= f["geo_solidity"] <= 1.0 and 0.0 <= f["geo_eccentricity"] <= 1.0


def test_wrong_mask_raises():
    m = _mask()
    cv2.circle(m, (20, 20), 5, 255, -1)
    cands = measure_regions(m, OPEN)
    with pytest.raises(ValueError):
        geometry_features(cands, _mask())


def test_pipeline_candidates_match_geometry_features():
    img = np.full((200, 200, 3), 200, np.uint8)
    for cx, cy in [(40, 40), (120, 60), (60, 150)]:
        cv2.circle(img, (cx, cy), 4, (60, 60, 60), -1)
    steps = run_pipeline_steps(img, CandidateParams())
    feats = geometry_features(steps["candidates"], steps["clean_mask"])
    assert len(feats) == len(steps["candidates"]) == 3