"""Conversions de couleur et lissage (Membre 3).

Convention du projet : une image est un tableau NumPy RGB, uint8, de forme (H, W, 3).
ATTENTION : cv2.imread() renvoie du BGR, pas du RGB. Si tu charges avec OpenCV,
convertis d'abord : cv2.cvtColor(img, cv2.COLOR_BGR2RGB).
"""
import cv2
import numpy as np


def to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """RGB (H, W, 3) -> niveaux de gris (H, W), uint8."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """RGB -> HSV (H, W, 3), uint8. Avec OpenCV : H in [0,179], S et V in [0,255]."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)


def smooth(channel: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Flou gaussien léger pour atténuer le bruit (JPEG, capteur) avant de seuiller.

    ksize doit être impair. ksize <= 1 : pas de lissage.
    """
    if ksize <= 1:
        return channel
    return cv2.GaussianBlur(channel, (ksize, ksize), 0)
