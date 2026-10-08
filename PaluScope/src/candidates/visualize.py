"""Visualisation des candidats (Membre 3)."""
import cv2
import numpy as np


def draw_candidates(image_rgb: np.ndarray, candidates: list, gt_boxes: list = None,
                    pad: int = 4) -> np.ndarray:
    """Retourne une COPIE de l'image avec les candidats (rouge) et, si fournies,
    les boîtes de vérité terrain (vert). gt_boxes : liste de dict xmin, ymin, xmax, ymax."""
    out = image_rgb.copy()
    for b in gt_boxes or []:
        cv2.rectangle(out, (int(b["xmin"]), int(b["ymin"])), (int(b["xmax"]), int(b["ymax"])),
                      (0, 255, 0), 1)
    for c in candidates:
        x, y, w, h = c["bbox"]
        cv2.rectangle(out, (x - pad, y - pad), (x + w + pad, y + h + pad), (255, 0, 0), 1)
    return out
