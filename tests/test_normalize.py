import numpy as np
import pytest

from src.preprocessing.normalize import NORM_MODES, normalize_image


def _tinted(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 200, (64, 64, 3)).astype(np.float32)
    return np.clip(img * np.array([1.3, 0.8, 0.9]), 0, 255).astype(np.uint8)  # dominante rouge


def test_none_returns_the_same_image():
    img = _tinted()
    assert normalize_image(img, "none") is img


def test_gray_world_equalizes_channel_means():
    out = normalize_image(_tinted(), "gray_world")
    means = out.reshape(-1, 3).mean(axis=0)
    assert out.dtype == np.uint8 and out.shape == (64, 64, 3)
    assert means.max() - means.min() < 2.0


def test_standardize_targets_mean_and_std():
    out = normalize_image(_tinted(), "standardize").astype(float).reshape(-1, 3)
    assert np.allclose(out.mean(axis=0), 128, atol=2.0)
    assert np.allclose(out.std(axis=0), 40, atol=3.0)


@pytest.mark.parametrize("mode", NORM_MODES)
def test_all_modes_keep_shape_and_dtype_and_handle_flat_image(mode):
    flat = np.full((32, 32, 3), 120, np.uint8)
    out = normalize_image(flat, mode)
    assert out.shape == flat.shape and out.dtype == np.uint8


def test_invalid_mode_or_input_raises():
    with pytest.raises(ValueError):
        normalize_image(_tinted(), "reinhard")
    with pytest.raises(ValueError):
        normalize_image(np.zeros((8, 8), np.uint8), "none")