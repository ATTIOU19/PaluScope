"""Mesure de netteté / flou (JusteAgbo05).

Entrée : image en niveaux de gris, uint8, 2D. Plus la valeur est élevée, plus l'image est nette.
Trois mesures candidates, comparées dans notebooks/phase_0/02_quality_analyzer.ipynb.
"""
import cv2
import numpy as np


def laplacian_variance(gray):
    """Variance du Laplacien : s = Var(∇²I). Un bord net donne un Laplacien de forte amplitude."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray):
    """Énergie du gradient de Sobel : s = moyenne de (Gx² + Gy²)."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def reblur_ratio(gray, sigma=2.0):
    """Perte de netteté après un flou supplémentaire : s = 1 − Var(∇²(G_σ ∗ I)) / Var(∇²I).

    Une image déjà floue change peu quand on la floute encore (s proche de 0) ;
    une image nette perd beaucoup de détails (s proche de 1). Valeur entre 0 et 1.
    """
    v0 = cv2.Laplacian(gray, cv2.CV_64F).var()
    if v0 <= 0:
        return 0.0
    v1 = cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), sigma), cv2.CV_64F).var()
    return float(max(0.0, 1.0 - v1 / v0))


METHODS = {"laplacian": laplacian_variance, "tenengrad": tenengrad, "reblur": reblur_ratio}


def sharpness(gray, method="laplacian"):
    """Netteté d'une image en niveaux de gris selon la méthode choisie (voir METHODS)."""
    if method not in METHODS:
        raise ValueError(f"Méthode de netteté inconnue : {method!r} (choix : {sorted(METHODS)})")
    return METHODS[method](gray)