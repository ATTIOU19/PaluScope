"""Contrôles de qualité et anti-fuite du jeu de données de candidats."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.split import SPLITS
from src.features.builder import FAMILIES, ID_COLUMNS, META_COLUMNS, feature_columns


def check_dataset(df: pd.DataFrame, registry: pd.DataFrame | None = None) -> list[str]:
    """Retourne la liste des problèmes détectés (liste vide = tout est conforme)."""
    problems: list[str] = []
    if df.empty:
        return ["table vide"]

    if (df.groupby("group_id")["split"].nunique() > 1).any():
        problems.append("fuite : un group_id figure dans plusieurs partitions")
    if (df.groupby("image_id")["split"].nunique() > 1).any():
        problems.append("fuite : une image figure dans plusieurs partitions")
    if not df["split"].isin(SPLITS).all():
        problems.append("split inconnu ou manquant")
    if not df["candidate_id"].is_unique:
        problems.append("candidate_id dupliqué")
    if not df["label"].isin([0, 1]).all():
        problems.append("label hors {0, 1}")
    if ((df["label"] == 1) != (df["match_box_id"] >= 0)).any():
        problems.append("label et match_box_id incohérents")

    feats = feature_columns(df, FAMILIES)
    values = df[feats].to_numpy(dtype=float)
    n_bad = int((~np.isfinite(values)).sum())
    if n_bad:
        problems.append(f"{n_bad} valeurs NaN/infinies dans les features")

    if registry is not None:
        dims = registry.set_index("image_id")[["width", "height"]]
        d = df.join(dims, on="image_id")
        out = (d["x"] < 0) | (d["y"] < 0) | (d["x"] + d["w"] > d["width"]) | (d["y"] + d["h"] > d["height"])
        if out.any():
            problems.append(f"{int(out.sum())} candidats dont la boîte dépasse de l'image")
    return problems


def dataset_summary(df: pd.DataFrame) -> dict:
    """Résumé chiffré (écrit dans reports/phase_1/dataset_summary.json)."""
    per_split = {}
    for s in SPLITS:
        d = df[df["split"] == s]
        per_split[s] = {
            "n_images_with_candidates": int(d["image_id"].nunique()),
            "n_candidates": int(len(d)),
            "n_positive": int(d["label"].sum()),
            "prevalence": float(d["label"].mean()) if len(d) else float("nan"),
        }
    return {
        "n_candidates": int(len(df)),
        "per_split": per_split,
        "n_features": {f: len(feature_columns(df, (f,))) for f in FAMILIES},
        "candidates_per_image": {k: float(v) for k, v in
                                 df.groupby("image_id").size().describe().round(2).to_dict().items()},
        "n_nan_features": int(df[feature_columns(df, FAMILIES)].isna().sum().sum()),
        "norm_mode": str(df["norm_mode"].iloc[0]) if len(df) else None,
        "columns_identification": ID_COLUMNS,
        "columns_meta": META_COLUMNS,
    }