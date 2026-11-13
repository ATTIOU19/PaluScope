"""Évaluation du détecteur de candidats : rappel R_cand et nombre de candidats par image.

R_cand = boîtes annotées contenant le centre d'au moins un candidat / boîtes annotées
(même définition que `match_candidates`). Il borne la sensibilité de toute la chaîne.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import pandas as pd

from src.candidates.evaluate import clip_boxes, match_candidates, parse_boxes
from src.candidates.extract import CandidateParams, extract_candidates


def load_rgb(path: str | Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"image illisible : {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def iter_samples(registry: pd.DataFrame, root: str | Path = ".") -> Iterator[tuple]:
    """Génère (image RGB, boîtes clippées) pour chaque ligne du registre."""
    for r in registry.itertuples():
        img = load_rgb(Path(root) / r.image_path)
        gt = clip_boxes(parse_boxes(Path(root) / r.xml_path), r.width, r.height) if r.xml_path else []
        yield img, gt


def evaluate_samples(samples: Iterable[tuple], params: CandidateParams) -> dict:
    n_img = n_gt = n_found = n_cand = n_useful = n_empty = 0
    for img, gt in samples:
        cands = extract_candidates(img, params)
        m = match_candidates(cands, gt)
        n_img += 1
        n_gt += m["n_gt"]
        n_found += m["n_found"]
        n_cand += m["n_candidates"]
        n_useful += m["n_candidates"] - m["n_false_candidates"]
        n_empty += int(m["n_candidates"] == 0)
    return {
        "n_images": n_img,
        "n_gt": n_gt,
        "n_found": n_found,
        "recall": n_found / n_gt if n_gt else float("nan"),
        "n_candidates": n_cand,
        "cand_per_image": n_cand / n_img if n_img else float("nan"),
        "candidate_precision": n_useful / n_cand if n_cand else float("nan"),
        "images_without_candidate": n_empty,
    }


def params_to_dict(params: CandidateParams) -> dict:
    return dataclasses.asdict(params)


def save_params(params: CandidateParams, path: str | Path, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps({"params": params_to_dict(params), "meta": meta or {}}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_params(path: str | Path) -> CandidateParams:
    """Lit configs/candidates_v1.json. Un champ inconnu lève TypeError (pas d'erreur silencieuse)."""
    return CandidateParams(**json.loads(Path(path).read_text(encoding="utf-8"))["params"])