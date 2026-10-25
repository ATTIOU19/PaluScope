"""Évaluation simple des candidats par rapport aux boîtes annotées (Membre 3).

Question posée : « parmi les parasites annotés, combien ont AU MOINS UN candidat
dans leur boîte ? » (c'est le RAPPEL des candidats). On veut qu'il soit élevé :
un parasite non candidat est définitivement perdu pour les phases suivantes.
On regarde aussi combien de candidats NE tombent dans aucune boîte (faux positifs,
acceptables à ce stade : la classification de la Phase 2 les éliminera).
"""
import re
from pathlib import Path


def parse_boxes(path) -> list:
    """Lit un fichier d'annotation (format Pascal VOC) -> liste de dict xmin, ymin, xmax, ymax."""
    txt = Path(path).read_text(errors="ignore")
    boxes = []
    for obj in re.findall(r"<object>(.*?)</object>", txt, flags=re.S | re.I):
        def g(tag):
            m = re.search(rf"<{tag}>\s*([^<\s]+)\s*</{tag}>", obj, flags=re.I)
            return m.group(1) if m else None
        try:
            boxes.append({k: float(g(k)) for k in ("xmin", "ymin", "xmax", "ymax")})
        except (TypeError, ValueError):
            pass
    return boxes


def clip_boxes(boxes: list, width: int, height: int) -> list:
    """Ramène les boîtes dans l'image (certaines dépassent, cf. note de synthèse)."""
    out = []
    for b in boxes:
        c = {"xmin": max(0.0, b["xmin"]), "ymin": max(0.0, b["ymin"]),
             "xmax": min(float(width), b["xmax"]), "ymax": min(float(height), b["ymax"])}
        if c["xmax"] > c["xmin"] and c["ymax"] > c["ymin"]:
            out.append(c)
    return out


def _center(c):
    x, y, w, h = c["bbox"]
    return x + w / 2.0, y + h / 2.0


def match_candidates(candidates: list, gt_boxes: list) -> dict:
    """Un parasite est « retrouvé » si le centre d'au moins un candidat est dans sa boîte."""
    found = 0
    for b in gt_boxes:
        if any(b["xmin"] <= _center(c)[0] <= b["xmax"] and b["ymin"] <= _center(c)[1] <= b["ymax"]
               for c in candidates):
            found += 1
    useful = sum(any(b["xmin"] <= _center(c)[0] <= b["xmax"] and b["ymin"] <= _center(c)[1] <= b["ymax"]
                     for b in gt_boxes) for c in candidates)
    return {"n_gt": len(gt_boxes), "n_found": found,
            "n_candidates": len(candidates), "n_false_candidates": len(candidates) - useful}
