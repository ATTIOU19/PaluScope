"""Extraction de régions candidates (Membre 3).

Pipeline : Image -> Gris/HSV -> Seuillage -> Morphologie
           -> Composantes connexes -> Contours -> Mesures géométriques

Idée générale : au grossissement x1000, un parasite apparaît comme une petite tache
SOMBRE sur un fond plus clair. On cherche donc de petites zones sombres, puis on
garde celles dont la taille et la forme sont plausibles. Ce sont des CANDIDATS :
on ne dit pas encore « c'est un parasite » (ce sera le rôle de la classification, Phase 2).

Format de sortie : voir docs/conventions.md.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from src.preprocessing.color import smooth, to_gray


@dataclass(frozen=True)
class CandidateParams:
    """Tous les réglages au même endroit : faciles à modifier et à documenter."""

    channel: str = "gray"        # "gray" ou "green" (canal vert : souvent plus contrasté sur frottis colorés)
    smooth_ksize: int = 3        # lissage gaussien (impair)
    blackhat_ksize: int = 15     # doit être PLUS GRAND que les taches à détecter (en px)
    min_contrast: float = 20.0   # seuil plancher (niveaux de gris). Trop bas : des taches voisines fusionnent (puis rejetées car trop grosses) et le bruit passe
    open_ksize: int = 3          # ouverture : supprime les petits pixels isolés (0 = désactivé)
    close_ksize: int = 3         # fermeture : rebouche les petits trous dans une tache (0 = désactivé)
    min_area: float = 10.0       # px² - en dessous : bruit (5 % des vrais parasites ont < 16 px²)
    max_area: float = 150.0      # px² - au dessus : cellule blanche, amas... (99 % des vrais parasites ont < 146 px² sur 200 images)
    min_circularity: float = 0.30  # 1 = cercle parfait. À cette taille (~9 px de diamètre) elle sépare peu : presque tout vaut ~1


# ----------------------------------------------------------------------------
# Étapes du pipeline (chacune est testable et visualisable séparément)
# ----------------------------------------------------------------------------
def get_channel(image_rgb: np.ndarray, params: CandidateParams) -> np.ndarray:
    """Étape 1 : choisir l'image 1-canal sur laquelle on travaille."""
    if params.channel == "green":
        ch = image_rgb[:, :, 1]
    elif params.channel == "gray":
        ch = to_gray(image_rgb)
    else:
        raise ValueError(f"channel inconnu : {params.channel!r} (attendu : 'gray' ou 'green')")
    return smooth(ch, params.smooth_ksize)


def darkness_map(channel: np.ndarray, ksize: int) -> np.ndarray:
    """Étape 2 : carte « de sombreur locale » par black-hat.

    black-hat = fermeture(image) - image. Une fermeture « comble » les petites taches
    sombres avec la couleur du fond voisin ; en soustrayant l'image d'origine il ne reste
    que ces petites taches sombres, indépendamment de la teinte globale du frottis.
    C'est ce qui nous protège de la forte variabilité de couleur entre images.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(channel, cv2.MORPH_BLACKHAT, kernel)


def threshold_map(dark: np.ndarray, min_contrast: float) -> tuple:
    """Étape 3 : seuillage. Retourne (masque binaire 0/255, seuil utilisé).

    On prend le seuil d'Otsu (choisi automatiquement) mais jamais en dessous de
    `min_contrast`, sinon sur une image sans rien de sombre Otsu couperait du bruit.
    """
    otsu_value, _ = cv2.threshold(dark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(otsu_value), float(min_contrast))
    _, mask = cv2.threshold(dark, thr, 255, cv2.THRESH_BINARY)
    return mask, thr


def clean_mask(mask: np.ndarray, open_ksize: int, close_ksize: int) -> np.ndarray:
    """Étape 4 : morphologie. Ouverture (enlève le bruit) puis fermeture (rebouche les trous)."""
    out = mask
    if open_ksize and open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out


def measure_regions(mask: np.ndarray, params: CandidateParams) -> list:
    """Étapes 5-6-7 : composantes connexes, contours, aire / périmètre / circularité."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for lab in range(1, n_labels):  # 0 = fond
        x, y, w, h, area = (int(v) for v in stats[lab])
        if area < params.min_area or area > params.max_area:
            continue
        # contour externe de CETTE composante (on travaille sur un petit recadrage)
        sub = (labels[y:y + h, x:x + w] == lab).astype(np.uint8)
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=len)
        perimeter = float(cv2.arcLength(cnt, True))
        if perimeter <= 0:
            continue
        circularity = min(1.0, 4.0 * np.pi * area / perimeter ** 2)
        if circularity < params.min_circularity:
            continue
        regions.append({
            "id": len(regions),
            "bbox": (x, y, w, h),
            "area": float(area),
            "perimeter": perimeter,
            "circularity": float(circularity),
        })
    return regions


# ----------------------------------------------------------------------------
# Point d'entrée
# ----------------------------------------------------------------------------
def extract_candidates(image, params: CandidateParams = None) -> list:
    """Retourne la liste des régions candidates (voir docs/conventions.md).

    image : tableau NumPy RGB uint8 (H, W, 3).
    Chaque candidat : {"id", "bbox": (x, y, w, h), "area", "perimeter", "circularity"}.
    """
    params = params or CandidateParams()
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image attendue : tableau RGB uint8 de forme (H, W, 3)")
    channel = get_channel(image, params)
    dark = darkness_map(channel, params.blackhat_ksize)
    mask, _ = threshold_map(dark, params.min_contrast)
    mask = clean_mask(mask, params.open_ksize, params.close_ksize)
    return measure_regions(mask, params)


def run_pipeline_steps(image, params: CandidateParams = None) -> dict:
    """Comme extract_candidates, mais renvoie aussi les images intermédiaires
    (pour les visualiser dans un notebook et EXPLIQUER chaque étape)."""
    params = params or CandidateParams()
    channel = get_channel(image, params)
    dark = darkness_map(channel, params.blackhat_ksize)
    raw_mask, thr = threshold_map(dark, params.min_contrast)
    mask = clean_mask(raw_mask, params.open_ksize, params.close_ksize)
    return {
        "channel": channel, "darkness": dark, "threshold": thr,
        "raw_mask": raw_mask, "clean_mask": mask,
        "candidates": measure_regions(mask, params),
    }


__all__ = ["CandidateParams", "extract_candidates", "run_pipeline_steps"]
