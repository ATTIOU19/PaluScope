"""Tests du Quality Analyzer sur des images synthétiques (aucune donnée réelle requise)."""
import cv2
import numpy as np
import pytest

from src.quality import analyze_image
from src.quality import blur, color, contrast, exposure
from src.quality.analyzer import DEFAULT_CONFIG, load_image

RNG = np.random.default_rng(0)


def noisy_rgb(size=200, base=(180, 140, 150), noise=12):
    img = np.ones((size, size, 3)) * np.array(base) + RNG.normal(0, noise, (size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


@pytest.mark.parametrize("method", sorted(blur.METHODS))
def test_sharpness_drops_when_image_is_blurred(method):
    sharp = to_gray(noisy_rgb())
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2.0)
    assert blur.sharpness(sharp, method) > blur.sharpness(blurred, method)


def test_sharpness_unknown_method():
    with pytest.raises(ValueError):
        blur.sharpness(to_gray(noisy_rgb()), "inconnue")


def test_reblur_ratio_is_bounded_and_zero_on_flat_image():
    flat = np.full((100, 100), 120, dtype=np.uint8)
    assert blur.reblur_ratio(flat) == 0.0
    assert 0.0 <= blur.reblur_ratio(to_gray(noisy_rgb())) <= 1.0


def test_brightness_and_exposure_ratios():
    gray = np.zeros((10, 10), dtype=np.uint8)
    gray[:, :5] = 255
    assert exposure.brightness(gray) == pytest.approx(127.5)
    assert exposure.overexposure_ratio(gray) == pytest.approx(0.5)
    assert exposure.underexposure_ratio(gray) == pytest.approx(0.5)


def test_contrast_is_zero_on_flat_image_and_large_on_two_levels():
    flat = np.full((50, 50), 100, dtype=np.uint8)
    two = np.zeros((50, 50), dtype=np.uint8)
    two[:, 25:] = 200
    for method in contrast.METHODS:
        assert contrast.contrast(flat, method) == 0.0
        assert contrast.contrast(two, method) > 50


def test_contrast_percentile_ignores_a_single_extreme_pixel():
    gray = np.full((100, 100), 120, dtype=np.uint8)
    gray[0, 0] = 0
    assert contrast.contrast_percentile(gray) == 0.0
    assert contrast.contrast_std(gray) > 0.0


def test_color_statistics_pure_red():
    red = np.zeros((20, 20, 3), dtype=np.uint8)
    red[..., 0] = 255
    stats = color.color_statistics(red)
    assert stats["mean_R"] == 255 and stats["mean_G"] == 0 and stats["mean_B"] == 0
    assert stats["mean_H_circ"] == pytest.approx(0.0, abs=1e-6)


def test_hue_mean_is_circular_around_red():
    # moitié des pixels à H = 2, moitié à H = 177 (deux rouges de part et d'autre de 0) : la moyenne circulaire reste proche de 0
    hsv = np.zeros((20, 20, 3), dtype=np.uint8)
    hsv[..., 1:] = 255
    hsv[:, :10, 0] = 2
    hsv[:, 10:, 0] = 177
    stats = color.color_statistics(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    assert abs(stats["mean_H_circ"]) < 5       # une moyenne arithmétique donnerait ≈ 90


def test_analyze_image_contract_and_types():
    result = analyze_image(noisy_rgb())
    for key in ("sharpness", "brightness", "contrast", "overexposure_ratio", "color_statistics"):
        assert key in result                    # clés de docs/conventions.md
    assert isinstance(result["sharpness"], float)
    assert isinstance(result["color_statistics"], dict)
    assert set(result["flags"]) == {"blurry", "low_contrast", "too_dark", "too_bright", "overexposed", "dark_borders"}


def test_uncalibrated_thresholds_give_none_flags():
    # Un seuil à None (non calibré) donne un drapeau None, quelles que soient les valeurs par défaut du module.
    flags = analyze_image(noisy_rgb(), {"sharpness_min": None, "contrast_min": None})["flags"]
    assert flags["blurry"] is None and flags["low_contrast"] is None


def test_flags_react_to_thresholds_and_exposure():
    bright = np.full((100, 100, 3), 255, dtype=np.uint8)
    flags = analyze_image(bright)["flags"]
    assert flags["overexposed"] and flags["too_bright"] and not flags["too_dark"]
    dark = np.zeros((100, 100, 3), dtype=np.uint8)
    flags = analyze_image(dark)["flags"]
    assert flags["too_dark"] and flags["dark_borders"]
    sharp = noisy_rgb()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3.0)
    s_sharp = analyze_image(sharp)["sharpness"]
    cfg = {"sharpness_min": s_sharp / 2}
    assert analyze_image(blurred, cfg)["flags"]["blurry"] is True
    assert analyze_image(sharp, cfg)["flags"]["blurry"] is False


def test_accepts_a_file_path(tmp_path):
    rgb = noisy_rgb()
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert analyze_image(path)["brightness"] == pytest.approx(analyze_image(rgb)["brightness"])
    assert load_image(path).shape == rgb.shape


def test_sharpness_is_resolution_independent_thanks_to_ref_size():
    small = noisy_rgb(size=300)
    large = cv2.resize(small, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
    s_small, s_large = analyze_image(small)["sharpness"], analyze_image(large)["sharpness"]
    assert s_small == pytest.approx(s_large, rel=0.35)
    assert DEFAULT_CONFIG["ref_size"] == 750


@pytest.mark.parametrize("bad", [np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.float32),
                                 np.zeros((10, 10, 4), dtype=np.uint8), "pas une image"])
def test_invalid_inputs_raise(bad):
    with pytest.raises((ValueError, FileNotFoundError)):
        analyze_image(bad)