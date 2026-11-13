"""Étiquetage des candidats à partir des boîtes annotées (déjà clippées).

Règle (docs/decisions.md) : un candidat est « parasite » (1) si son centre tombe dans une boîte
annotée, « non-parasite / artefact » (0) sinon. Même règle que `match_candidates`
(src/candidates/evaluate.py), mais on garde l'identifiant de la boîte associée.

Attention : les boîtes font ~40x40 px alors que les parasites détectés font ~10-150 px² : toute
tache sombre dans la boîte reçoit l'étiquette 1. `shrink` < 1 resserre la boîte autour de son
centre (analyse de sensibilité en Phase 2). Valeur par défaut 1.0 = règle officielle.
"""
from __future__ import annotations


def candidate_center(cand: dict) -> tuple[float, float]:
    x, y, w, h = cand["bbox"]
    return x + w / 2.0, y + h / 2.0


def _bounds(b: dict, shrink: float) -> tuple[float, float, float, float]:
    if shrink >= 1.0:
        return b["xmin"], b["ymin"], b["xmax"], b["ymax"]
    cx, cy = (b["xmin"] + b["xmax"]) / 2.0, (b["ymin"] + b["ymax"]) / 2.0
    hw, hh = (b["xmax"] - b["xmin"]) / 2.0 * shrink, (b["ymax"] - b["ymin"]) / 2.0 * shrink
    return cx - hw, cy - hh, cx + hw, cy + hh


def label_candidates(candidates: list[dict], gt_boxes: list[dict], shrink: float = 1.0) -> list[dict]:
    """Retourne des copies des candidats avec `label` (0/1) et `match_box_id`.

    gt_boxes : boîtes clippées au format {xmin, ymin, xmax, ymax} (sortie de `clip_boxes`).
    match_box_id : indice de la boîte dans `gt_boxes` (-1 si aucune). Si le centre tombe dans
    plusieurs boîtes (boîtes qui se chevauchent), on retient la boîte dont le centre est le plus
    proche : choix déterministe.
    """
    assert 0.0 < shrink <= 1.0, "shrink doit être dans ]0, 1]"
    bounds = [_bounds(b, shrink) for b in gt_boxes]
    out = []
    for c in candidates:
        cx, cy = candidate_center(c)
        best, best_d = -1, float("inf")
        for i, (x0, y0, x1, y1) in enumerate(bounds):
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                d = (cx - (x0 + x1) / 2.0) ** 2 + (cy - (y0 + y1) / 2.0) ** 2
                if d < best_d:
                    best, best_d = i, d
        out.append({**c, "label": int(best >= 0), "match_box_id": best})
    return out