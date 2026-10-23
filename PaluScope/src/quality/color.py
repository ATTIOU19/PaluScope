"""Statistiques de couleur RGB / HSV (JusteAgbo05)."""
import cv2
import numpy as np


def color_statistics(rgb):
    """Statistiques de couleur d'une image RGB uint8 (H×W×3). Retourne un dict de flottants.

    - mean_R/G/B, std_R/G/B : moyenne et écart-type de chaque canal (0-255) ;
    - ratio_R_G, ratio_B_G : rapports des moyennes (indicateur de teinte de la coloration) ;
    - mean_H_circ : teinte moyenne CIRCULAIRE en unités OpenCV (−90 à 90, 0 = rouge). La teinte est un angle :
      0 et 179 sont tous deux du rouge, une moyenne arithmétique serait fausse ;
    - mean_S, mean_V : saturation et valeur moyennes (0-255).
    """
    mean, std = (a.ravel() for a in cv2.meanStdDev(rgb))               # moyenne et écart-type par canal (R, G, B)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Teinte moyenne circulaire, calculée sur l'histogramme des 180 valeurs de H (même résultat, bien plus rapide)
    freq = np.bincount(hsv[..., 0].ravel(), minlength=180) / hsv[..., 0].size
    theta = np.arange(180) * (2 * np.pi / 180)                          # H (0-179) → angle (0-2π)
    mean_h = np.degrees(np.arctan2((freq * np.sin(theta)).sum(), (freq * np.cos(theta)).sum())) / 2

    return {
        "mean_R": float(mean[0]), "mean_G": float(mean[1]), "mean_B": float(mean[2]),
        "std_R": float(std[0]), "std_G": float(std[1]), "std_B": float(std[2]),
        "ratio_R_G": float(mean[0] / max(mean[1], 1e-9)), "ratio_B_G": float(mean[2] / max(mean[1], 1e-9)),
        "mean_H_circ": float(mean_h),
        "mean_S": float(hsv[..., 1].mean()), "mean_V": float(hsv[..., 2].mean()),
    }