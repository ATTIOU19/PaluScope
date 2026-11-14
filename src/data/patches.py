"""Extraction de patches autour des candidats."""
import math

import numpy as np


def coin_haut_gauche(bbox, size=48):
    """Renvoie le coin haut-gauche (x0, y0) du patch centré sur la boîte."""
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    x0 = math.floor(cx - size / 2 + 0.5)
    y0 = math.floor(cy - size / 2 + 0.5)
    return x0, y0


def reflechir(i, n):
    """Ramène l'indice i dans [0, n-1] en le reflétant comme un miroir."""
    if n == 1:
        return 0
    periode = 2 * (n - 1)
    m = i % periode
    if m >= n:
        m = periode - m
    return m


def extract_patch(image, bbox, size=48):
    """Patch carré (size, size, 3) centré sur la boîte, rempli par réflexion aux bords."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image attendue : tableau RGB uint8 de forme (H, W, 3)")
    x0, y0 = coin_haut_gauche(bbox, size)
    hauteur, largeur = image.shape[:2]
    lignes = [reflechir(y0 + k, hauteur) for k in range(size)]
    colonnes = [reflechir(x0 + k, largeur) for k in range(size)]
    return image[lignes][:, colonnes]