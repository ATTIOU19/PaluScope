"""Audit visuel des quasi-doublons : planche de paires par niveau de distance.

Usage : python scripts/audit_duplicates.py [--per-level 3]
Sorties : reports/phase_1/duplicate_audit.csv, reports/phase_1/duplicate_contact_sheet.png

Constat sur le dataset : les vrais doublons ne sont PAS des copies exactes. Même champ, mais
cadrage légèrement décalé, donc boîtes annotées différentes. Comparer les annotations ne
permet donc pas de trancher : le contrôle est visuel (planche) sur quelques paires par niveau.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CELL = 240


def _cell(path: str, text: str):
    img = cv2.resize(cv2.imread(path), (CELL, CELL), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (CELL, 18), (255, 255, 255), -1)
    cv2.putText(img, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default="data/registry/images.csv")
    ap.add_argument("--pairs", default="data/registry/duplicate_pairs.csv")
    ap.add_argument("--out-dir", default="reports/phase_1")
    ap.add_argument("--per-level", type=int, default=3, help="paires affichées par niveau de distance")
    args = ap.parse_args()

    reg = pd.read_csv(args.registry, keep_default_na=False).set_index("image_id")
    pairs = pd.read_csv(args.pairs).sort_values(["hamming", "image_a", "image_b"]).reset_index(drop=True)
    pairs["n_boxes_a"] = [int(reg.loc[i, "n_boxes_raw"]) for i in pairs["image_a"]]
    pairs["n_boxes_b"] = [int(reg.loc[i, "n_boxes_raw"]) for i in pairs["image_b"]]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "duplicate_audit.csv", index=False)
    print(pairs.groupby("hamming").size().rename("n_pairs").to_string())

    rows = []
    for d, g in pairs.groupby("hamming"):
        # échantillon réparti sur tout le niveau (pas seulement les premiers identifiants)
        idx = np.unique(np.linspace(0, len(g) - 1, min(args.per_level, len(g))).round().astype(int))
        for _, r in g.iloc[idx].iterrows():
            a = _cell(reg.loc[r.image_a, "image_path"], f"{r.image_a[-4:]} d={d} boxes {r.n_boxes_a}/{r.n_boxes_b}")
            b = _cell(reg.loc[r.image_b, "image_path"], f"{r.image_b[-4:]}")
            rows.append(cv2.hconcat([a, b]))
    if rows:
        if len(rows) % 2:
            rows.append(np.full_like(rows[-1], 255))
        sheet = cv2.vconcat([cv2.hconcat(rows[i : i + 2]) for i in range(0, len(rows), 2)])
        cv2.imwrite(str(out / "duplicate_contact_sheet.png"), sheet)
        print(f"planche : {out / 'duplicate_contact_sheet.png'}  ({len(rows)} paires)")


if __name__ == "__main__":
    main()