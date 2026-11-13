"""Séparation train / val / test PAR GROUPE, stratifiée sur le nombre de parasites."""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def make_split(
    registry: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    n_bins: int = 4,
) -> pd.DataFrame:
    """Retourne un DataFrame (image_id, group_id, split).

    Principe : un groupe n'est jamais coupé. Les groupes sont rangés en strates selon leur
    nombre moyen de boîtes par image (strate « zero » = aucune annotation), puis, dans chaque
    strate, affectés au split qui a le plus grand déficit par rapport à sa cible (en images).
    Entièrement déterministe pour (registry, ratios, seed, n_bins).
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "les proportions doivent sommer à 1"
    g = (
        registry.groupby("group_id")
        .agg(n_images=("image_id", "size"), n_boxes=("n_boxes", "sum"))
        .reset_index()
    )
    g["mean_boxes"] = g["n_boxes"] / g["n_images"]
    g["stratum"] = "zero"
    nz = g["n_boxes"] > 0
    if nz.sum() > 0:
        q = min(n_bins, int(nz.sum()))
        ranks = g.loc[nz, "mean_boxes"].rank(method="first")
        g.loc[nz, "stratum"] = "b" + pd.qcut(ranks, q=q, labels=False).astype(str)

    assign: dict[str, str] = {}
    for _, gs in g.groupby("stratum", sort=True):
        gs = gs.sample(frac=1.0, random_state=seed)                      # mélange déterministe
        gs = gs.sort_values("n_images", ascending=False, kind="stable")  # gros groupes d'abord
        total = gs["n_images"].sum()
        target = {s: r * total for s, r in zip(SPLITS, ratios)}
        current = dict.fromkeys(SPLITS, 0)
        for gid, n in zip(gs["group_id"], gs["n_images"]):
            s = max(SPLITS, key=lambda k: target[k] - current[k])
            assign[gid] = s
            current[s] += int(n)

    out = registry[["image_id", "group_id"]].copy()
    out["split"] = out["group_id"].map(assign)
    return out.sort_values("image_id").reset_index(drop=True)


def assert_no_leakage(split_df: pd.DataFrame) -> None:
    """Lève AssertionError si un groupe ou une image figure dans plusieurs partitions."""
    per_group = split_df.groupby("group_id")["split"].nunique()
    assert (per_group == 1).all(), f"groupes présents dans plusieurs splits : {list(per_group[per_group > 1].index[:5])}"
    assert split_df["image_id"].is_unique, "image_id dupliqué"
    assert split_df["split"].isin(SPLITS).all(), "split manquant ou inconnu"


def split_report(registry: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    """Tableau récapitulatif par partition (images, groupes, boîtes, non annotées)."""
    df = registry.merge(split_df[["image_id", "split"]], on="image_id")
    rep = df.groupby("split").agg(
        n_images=("image_id", "size"),
        n_groups=("group_id", "nunique"),
        n_boxes=("n_boxes", "sum"),
        n_unannotated=("image_unannotated", "sum"),
    )
    rep["share_images"] = (rep["n_images"] / rep["n_images"].sum()).round(3)
    rep["boxes_per_image"] = (rep["n_boxes"] / rep["n_images"]).round(2)
    return rep.reindex(list(SPLITS))