"""Métriques de la Phase 2 (bloc P2-A, Membre 1).

Contrat (docs/conventions.md) :
    evaluate(y_true, y_score, thr) -> dict
Prédiction positive si ``y_score >= thr``. Dénominateur nul -> ``None`` (jamais de NaN
silencieux). ``roc_auc`` et ``pr_auc`` valent ``None`` si une seule classe est présente.

Niveau objet (PRÉLIMINAIRE, formalisé en Phase 4 ; définition du comptage « ouverte »,
à valider avec le mentor) :
- une boîte annotée est retrouvée si au moins un candidat prédit positif lui est associé
  (``match_box_id``) ; les positifs multiples sur une même boîte sont fusionnés ;
- chaque candidat positif sans boîte associée compte comme un objet (faux positif) ;
- comptage prédit d'une image = boîtes retrouvées + faux positifs ;
- MAE = moyenne sur les images de |comptage prédit - comptage vrai|.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _div(a: float, b: float):
    return float(a / b) if b > 0 else None


def evaluate(y_true, y_score, thr: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true et y_score doivent avoir la même forme")
    pred = y_score >= thr
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int((~pred & (y_true == 1)).sum())
    tn = int((~pred & (y_true == 0)).sum())

    precision, recall, specificity = _div(tp, tp + fp), _div(tp, tp + fn), _div(tn, tn + fp)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    roc_auc = pr_auc = None
    if len(y_true) > 0 and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

    return {
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "prevalence": float(y_true.mean()) if len(y_true) else None,
        "thr": float(thr),
    }


def choose_threshold(y_true, y_score, target_recall: float = 0.95) -> dict:
    """Seuil à choisir sur la VALIDATION : sensibilité >= cible avec spécificité maximale.

    Le critère exact est « ouvert » dans le journal des décisions : la cible est un
    paramètre (défaut 0,95) à confirmer. Égalités : on garde le seuil le plus élevé.
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Choix du seuil impossible : une seule classe dans y_true")
    fpr, tpr, thr = roc_curve(y_true, np.asarray(y_score, dtype=float))
    ok = np.where((tpr >= target_recall) & np.isfinite(thr))[0]
    if len(ok) == 0:                      # ne devrait pas arriver : le plus petit seuil donne tpr = 1
        ok = np.where(np.isfinite(thr))[0][-1:]
    best = ok[np.argmin(fpr[ok])]         # fpr minimal ; à égalité, premier = seuil le plus haut
    return {"thr": float(thr[best]), "recall": float(tpr[best]),
            "specificity": float(1 - fpr[best]), "target_recall": float(target_recall)}


def object_level(candidates: pd.DataFrame, y_score, thr: float, images: pd.DataFrame) -> dict:
    """Comptage et rappel au niveau objet, sur UNE partition.

    candidates : colonnes image_id, match_box_id (une ligne par candidat de la partition)
    images     : registre restreint à la même partition (colonnes image_id, n_boxes)
    """
    df = candidates[["image_id", "match_box_id"]].copy()
    df["pred"] = np.asarray(y_score, dtype=float) >= thr
    pos = df[df["pred"]]

    matched = (pos.dropna(subset=["match_box_id"]).groupby("image_id")["match_box_id"].nunique())
    false_pos = pos[pos["match_box_id"].isna()].groupby("image_id").size()

    true_count = images.set_index("image_id")["n_boxes"].astype(int)
    pred_count = (matched.reindex(true_count.index, fill_value=0)
                  + false_pos.reindex(true_count.index, fill_value=0))
    found = int(matched.reindex(true_count.index, fill_value=0).sum())
    n_true = int(true_count.sum())
    return {
        "n_images": int(len(true_count)),
        "n_true_objects": n_true,
        "n_found_objects": found,
        "object_recall": _div(found, n_true),
        "count_mae": float((pred_count - true_count).abs().mean()) if len(true_count) else None,
    }
