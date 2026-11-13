"""Lecture et contrôle de la table de candidats (sortie de build_dataset, Phase 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["image_id", "candidate_id", "split", "group_id", "label", "match_box_id"]
SPLITS = {"train", "val", "test"}

# Jeux de features nommés (feuille de route, section 4.3). qual_ : testées à part.
FEATURE_SETS = {
    "F1": ["geo_"],
    "F2": ["geo_", "col_"],
    "F3": ["geo_", "col_", "grd_", "tex_"],
    "F4": ["geo_", "col_", "grd_", "tex_", "hog_"],
}


def feature_columns(df: pd.DataFrame, feature_set: str = "F3", use_qual: bool = False) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Jeu inconnu : {feature_set!r} (attendu : {sorted(FEATURE_SETS)})")
    prefixes = list(FEATURE_SETS[feature_set]) + (["qual_"] if use_qual else [])
    cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    if not cols:
        raise ValueError(f"Aucune colonne de features pour {feature_set} (préfixes {prefixes})")
    return cols


def load_table(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    bad = set(df["split"].dropna().unique()) - SPLITS
    if bad or df["split"].isna().any():
        raise ValueError(f"Valeurs de split invalides : {bad or 'NaN'} (attendu : {sorted(SPLITS)})")
    feats = [c for c in df.columns if c.startswith(("geo_", "col_", "grd_", "tex_", "hog_", "qual_"))]
    arr = df[feats].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        cols = [f for f, ok in zip(feats, np.isfinite(arr).all(axis=0)) if not ok]
        raise ValueError(f"NaN ou infini dans les features : {cols[:10]}")
    # Contrôle anti-fuite : un group_id ne doit jamais apparaître dans deux partitions.
    n_splits_per_group = df.groupby("group_id")["split"].nunique()
    if (n_splits_per_group > 1).any():
        raise ValueError("FUITE : des group_id apparaissent dans plusieurs partitions")
    return df
