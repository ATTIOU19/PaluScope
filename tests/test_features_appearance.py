import cv2
import numpy as np
import pytest

import src.features.quality as quality
from src.features.color import color_features
from src.features.hog import N_HOG, hog_features
from src.features.texture import gradient_features, texture_features


def _scene():
    """Fond rose avec une tache sombre ronde au centre (candidat)."""
    rng = np.random.default_rng(0)
    img = np.clip(np.array([210, 150, 170]) + rng.normal(0, 4, (100, 100, 3)), 0, 255).astype(np.uint8)
    cv2.circle(img, (50, 50), 5, (70, 50, 90), -1)
    mask = np.zeros((11, 11), np.uint8)
    cv2.circle(mask, (5, 5), 5, 1, -1)
    return img, (45, 45, 11, 11), mask


def test_color_candidate_darker_than_ring():
    img, bbox, mask = _scene()
    f = color_features(img, bbox, mask)
    assert f["col_gray_diff"] < -40 and f["col_gray_ratio"] < 0.6
    assert f["col_r_mean"] < f["col_r_ring_mean"]
    assert abs(sum(f[f"col_r_hist{k}"] for k in range(8)) - 1.0) < 1e-6


def test_color_feature_names_are_stable_and_finite():
    img, bbox, mask = _scene()
    f = color_features(img, bbox, mask)
    assert len(f) == 8 * 5 + 6 + 3 * 8  # 70
    assert all(k.startswith("col_") for k in f) and all(np.isfinite(v) for v in f.values())


def test_color_at_image_corner_does_not_crash():
    img, _, mask = _scene()
    f = color_features(img, (0, 0, 11, 11), mask)
    assert all(np.isfinite(v) for v in f.values())


@pytest.mark.parametrize("patch", [
    np.zeros((48, 48), np.uint8),                                              # patch uniforme
    np.random.default_rng(1).integers(0, 255, (48, 48), dtype=np.uint8),       # bruit
])
def test_gradient_texture_hog_shapes_names_and_no_nan(patch):
    g, t, h = gradient_features(patch), texture_features(patch), hog_features(patch)
    assert len(g) == 3 and len(t) == 10 + 4 and len(h) == N_HOG == 324
    assert all(k.startswith("grd_") for k in g) and all(k.startswith("tex_") for k in t)
    for d in (g, t, h):
        assert all(np.isfinite(v) for v in d.values())


def test_features_are_deterministic():
    patch = np.random.default_rng(2).integers(0, 255, (48, 48), dtype=np.uint8)
    assert texture_features(patch) == texture_features(patch)
    assert hog_features(patch) == hog_features(patch)


def test_quality_features_use_analyzer(monkeypatch):
    fake = {"sharpness": 12.0, "brightness": 150.0, "contrast": 30.0,
            "overexposure_ratio": 0.0, "underexposure_ratio": 0.01, "flags": {}}
    monkeypatch.setattr(quality, "_analyze", lambda img: fake)
    f = quality.quality_features(np.zeros((8, 8, 3), np.uint8))
    assert set(f) == {f"qual_{k}" for k in quality.QUAL_KEYS} and f["qual_brightness"] == 150.0