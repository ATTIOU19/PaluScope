import numpy as np
import pytest

from src.data.patches import coin_haut_gauche, extract_patch, reflechir


def image_degradee(hauteur=100, largeur=120):
    """Image où le rouge = numéro de colonne et le vert = numéro de ligne."""
    img = np.zeros((hauteur, largeur, 3), dtype=np.uint8)
    img[:, :, 0] = np.arange(largeur)[None, :]
    img[:, :, 1] = np.arange(hauteur)[:, None]
    return img


def test_coin_haut_gauche():
    assert coin_haut_gauche((50, 40, 10, 10)) == (31, 21)


def test_reflechir_comme_un_miroir():
    assert reflechir(5, 750) == 5
    assert reflechir(-3, 750) == 3
    assert reflechir(750, 750) == 748
    assert reflechir(751, 750) == 747


def test_forme_et_position_du_patch():
    patch = extract_patch(image_degradee(), (50, 40, 10, 10))
    assert patch.shape == (48, 48, 3)
    assert patch.dtype == np.uint8
    assert patch[0, 0, 0] == 31    # colonne du coin haut-gauche
    assert patch[0, 0, 1] == 21    # ligne du coin haut-gauche


def test_patch_au_bord_comme_numpy_pad():
    img = image_degradee()
    patch = extract_patch(img, (0, 0, 10, 10))      # le coin tombe en (-19, -19)
    attendu = np.pad(img, ((19, 0), (19, 0), (0, 0)), mode="reflect")[0:48, 0:48]
    assert np.array_equal(patch, attendu)


def test_image_non_modifiee_et_mauvaise_entree():
    img = image_degradee()
    avant = img.copy()
    patch = extract_patch(img, (50, 40, 10, 10))
    patch[:] = 0
    assert np.array_equal(img, avant)
    with pytest.raises(ValueError):
        extract_patch(np.zeros((10, 10), dtype=np.uint8), (1, 1, 2, 2))