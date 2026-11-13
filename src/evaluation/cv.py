"""Validation croisée groupée et utilitaires d'entraînement (bloc P2-A, Membre 1).

Règles : plis groupés par ``group_id`` (jamais par candidat), 5 plis, graine fixe ;
le sous-échantillonnage des négatifs (si demandé) ne touche que le TRAIN du pli.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluation.metrics import evaluate

SEED = 42


def positive_score(model, X) -> np.ndarray:
    """Score de la classe positive : predict_proba si disponible, sinon decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def undersample_indices(y, ratio: float | None, seed: int = SEED) -> np.ndarray:
    """Garde tous les positifs et au plus ``ratio`` négatifs par positif. ratio=None : tout garder."""
    y = np.asarray(y).astype(int)
    idx = np.arange(len(y))
    if ratio is None:
        return idx
    pos, neg = idx[y == 1], idx[y == 0]
    n_keep = int(min(len(neg), ratio * len(pos)))
    rng = np.random.default_rng(seed)
    keep_neg = rng.choice(neg, size=n_keep, replace=False)
    return np.sort(np.concatenate([pos, keep_neg]))


def fit_model(estimator, X, y, undersample_ratio: float | None = None, seed: int = SEED):
    keep = undersample_indices(y, undersample_ratio, seed)
    model = clone(estimator)
    model.fit(X[keep], np.asarray(y)[keep])
    return model


def grouped_cv(estimator, X, y, groups, n_splits: int = 5, seed: int = SEED,
               undersample_ratio: float | None = None) -> dict:
    """Validation croisée groupée sur le TRAIN. Renvoie les scores hors-pli et les métriques par pli.

    Seules les métriques indépendantes du seuil (ROC-AUC, PR-AUC) sont rapportées ici ;
    le seuil se choisit sur la validation, pas dans la validation croisée.
    """
    X, y, groups = np.asarray(X), np.asarray(y).astype(int), np.asarray(groups)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    folds = []
    for k, (tr, va) in enumerate(skf.split(X, y, groups)):
        if set(groups[tr]) & set(groups[va]):          # garde-fou anti-fuite
            raise RuntimeError("FUITE entre plis : un group_id est dans train et validation")
        model = fit_model(estimator, X[tr], y[tr], undersample_ratio, seed)
        oof[va] = positive_score(model, X[va])
        m = evaluate(y[va], oof[va], thr=0.0)
        folds.append({"fold": k, "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                      "prevalence": m["prevalence"], "n": int(len(va))})
    return {"oof_scores": oof, "folds": folds}


def summarize_folds(folds: list[dict], keys=("roc_auc", "pr_auc")) -> dict:
    """Moyenne et écart-type entre plis (les plis sans valeur sont ignorés)."""
    out = {}
    for key in keys:
        vals = [f[key] for f in folds if f[key] is not None]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{key}_std"] = float(np.std(vals)) if vals else None
    return out
