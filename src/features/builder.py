"""Assemblage du jeu de données : une ligne par candidat, avec features et étiquette.

Chaîne par image : image brute -> candidats (réglages figés) -> étiquettes -> features.
Les candidats sont détectés sur l'image BRUTE ; la normalisation ne sert qu'aux features.
Les patches sauvegardés sont BRUTS (la normalisation peut se refaire plus tard).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.candidates.evaluate import clip_boxes, parse_boxes
from src.candidates.extract import CandidateParams, run_pipeline_steps
from src.data.candidate_eval import load_rgb
from src.data.labeling import label_candidates
from src.data.patches import PATCH_SIZE, extract_patch
from src.features.color import color_features
from src.features.geometry import candidate_masks, geometry_features
from src.features.hog import hog_features
from src.features.quality import quality_features
from src.features.texture import gradient_features, texture_features
from src.preprocessing.color import to_gray
from src.preprocessing.normalize import normalize_image

FAMILIES = ("geo", "col", "grd", "tex", "hog", "qual")
ID_COLUMNS = [
    "image_id", "candidate_id", "split", "group_id", "dataset_source", "device", "smear_type",
    "image_unannotated", "x", "y", "w", "h", "label", "match_box_id",
]
META_COLUMNS = ["norm_mode", "candidates_cfg", "features_cfg"]
FEATURE_PREFIXES = tuple(f"{f}_" for f in FAMILIES)


def feature_columns(df: pd.DataFrame, families=FAMILIES) -> list[str]:
    """Colonnes de features appartenant aux familles demandées (ex. ('geo', 'col'))."""
    prefixes = tuple(f"{f}_" for f in families)
    return [c for c in df.columns if c.startswith(prefixes)]


def load_features_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(cfg.get("families", FAMILIES)) - set(FAMILIES)
    if unknown:
        raise ValueError(f"familles inconnues : {sorted(unknown)}")
    return cfg


def candidate_rows(image_rgb, gt_boxes, params: CandidateParams, *, norm_mode="none",
                   families=FAMILIES, shrink=1.0, patch_size=PATCH_SIZE):
    """Lignes (liste de dict) et patches bruts (N, p, p, 3) pour UNE image."""
    steps = run_pipeline_steps(image_rgb, params)
    cands = steps["candidates"]
    if not cands:
        return [], np.empty((0, patch_size, patch_size, 3), np.uint8)

    labeled = label_candidates(cands, gt_boxes, shrink)
    masks = candidate_masks(cands, steps["clean_mask"])
    geo = geometry_features(cands, steps["clean_mask"]) if "geo" in families else [{} for _ in cands]
    norm = normalize_image(image_rgb, norm_mode)
    qual = quality_features(image_rgb) if "qual" in families else {}
    need_patch = any(f in families for f in ("grd", "tex", "hog"))

    rows, patches = [], []
    for c, lab, mask, g in zip(cands, labeled, masks, geo):
        x, y, w, h = c["bbox"]
        row = {"candidate_index": c["id"], "x": x, "y": y, "w": w, "h": h,
               "label": lab["label"], "match_box_id": lab["match_box_id"], **g}
        if "col" in families:
            row.update(color_features(norm, c["bbox"], mask))
        if need_patch:
            gray = to_gray(extract_patch(norm, c["bbox"], patch_size))
            if "grd" in families:
                row.update(gradient_features(gray))
            if "tex" in families:
                row.update(texture_features(gray))
            if "hog" in families:
                row.update(hog_features(gray))
        row.update(qual)
        rows.append(row)
        patches.append(extract_patch(image_rgb, c["bbox"], patch_size))
    return rows, np.stack(patches)


def build_dataset(registry: pd.DataFrame, params: CandidateParams, *, norm_mode="none",
                  families=FAMILIES, shrink=1.0, patch_size=PATCH_SIZE, root=".",
                  candidates_cfg="candidates_v1", features_cfg="features_v1", progress_every=0):
    """Retourne (DataFrame, patches). Déterministe : mêmes entrées -> même table."""
    all_rows, all_patches = [], []
    for i, r in enumerate(registry.sort_values("image_id").itertuples(), 1):
        img = load_rgb(Path(root) / r.image_path)
        gt = clip_boxes(parse_boxes(Path(root) / r.xml_path), r.width, r.height) if r.xml_path else []
        rows, patches = candidate_rows(img, gt, params, norm_mode=norm_mode, families=families,
                                       shrink=shrink, patch_size=patch_size)
        for row in rows:
            row.update(
                image_id=r.image_id, candidate_id=f"{r.image_id}_{row.pop('candidate_index'):03d}",
                split=r.split, group_id=r.group_id, dataset_source=r.dataset_source, device=r.device,
                smear_type=r.smear_type, image_unannotated=bool(r.image_unannotated),
                norm_mode=norm_mode, candidates_cfg=candidates_cfg, features_cfg=features_cfg,
            )
        all_rows.extend(rows)
        all_patches.append(patches)
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(registry)} images, {len(all_rows)} candidats", flush=True)

    df = pd.DataFrame(all_rows)
    feats = [c for c in df.columns if c not in ID_COLUMNS and c not in META_COLUMNS]
    df = df[ID_COLUMNS + feats + META_COLUMNS]
    patches = np.concatenate(all_patches) if all_patches else np.empty((0, patch_size, patch_size, 3), np.uint8)
    return df, patches