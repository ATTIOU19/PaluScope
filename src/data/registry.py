"""Registre des images : une ligne par image, métadonnées + signature + groupe."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .dedup import bits_to_hex, group_near_duplicates, nearest_neighbor_distance, phash
from .voc import read_voc_boxes, summarize_boxes

DATASET_SOURCE = "makerere_uganda_mobile_thick"
DEVICE = "smartphone_on_microscope"
SMEAR_TYPE = "thick"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

DEFAULT_DATA_DIR = "data/raw/plasmodium_phonecamera"
DEFAULT_REGISTRY = "data/registry/images.csv"
DEFAULT_SPLIT = "data/splits/split_v1.csv"


def build_registry(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    root: str | Path = ".",
    hash_size: int = 8,
    threshold: int = 8,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Construit le registre. Retourne (registre, matrice des distances de Hamming).

    Les chemins sont stockés relatifs à `root` (séparateurs POSIX) pour rester portables.
    """
    root = Path(root)
    data_dir = Path(data_dir)
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"aucune image dans {data_dir}")

    rows, hashes = [], []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)  # BGR : seule la forme et le gris servent ici
        if bgr is None:
            raise ValueError(f"image illisible : {p}")
        h, w = bgr.shape[:2]
        bits = phash(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), hash_size=hash_size)
        hashes.append(bits)

        xml = p.with_suffix(".xml")
        has_xml = xml.exists()
        counts = summarize_boxes(read_voc_boxes(xml) if has_xml else [], w, h)
        rows.append(
            {
                "image_id": p.stem,
                "image_path": p.relative_to(root).as_posix(),
                "xml_path": xml.relative_to(root).as_posix() if has_xml else "",
                "width": w,
                "height": h,
                **counts,
                "image_unannotated": counts["n_boxes_raw"] == 0,
                "dataset_source": DATASET_SOURCE,
                "device": DEVICE,
                "smear_type": SMEAR_TYPE,
                "phash": bits_to_hex(bits),
            }
        )

    reg = pd.DataFrame(rows)
    labels, D = group_near_duplicates(np.vstack(hashes), threshold)
    reg["group_id"] = [f"G{l:04d}" for l in labels]
    reg["nn_distance"] = nearest_neighbor_distance(D).astype(int)
    return reg, D


def load_registry(
    registry_csv: str | Path = DEFAULT_REGISTRY,
    split_csv: str | Path | None = DEFAULT_SPLIT,
) -> pd.DataFrame:
    """Contrat d'interface : DataFrame (une ligne par image) avec group_id et split."""
    reg = pd.read_csv(registry_csv, keep_default_na=False, dtype={"xml_path": str})
    if split_csv is not None and Path(split_csv).exists():
        sp = pd.read_csv(split_csv)
        merged = reg.merge(sp, on="image_id", suffixes=("", "_split"), validate="one_to_one")
        assert (merged["group_id"] == merged["group_id_split"]).all(), (
            "group_id du registre et de split_v1 divergent : relancer build_registry puis make_split"
        )
        reg = merged.drop(columns="group_id_split")
    return reg