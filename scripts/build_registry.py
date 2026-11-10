"""Construit le registre des images et les groupes de quasi-doublons.

Usage (depuis la racine du dépôt) :
    python scripts/build_registry.py --threshold 8
Sorties : data/registry/images.csv, data/registry/duplicate_pairs.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.registry import DEFAULT_DATA_DIR, DEFAULT_REGISTRY, build_registry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", default=DEFAULT_REGISTRY)
    ap.add_argument("--threshold", type=int, default=8, help="distance de Hamming max (64 bits) pour fusionner")
    ap.add_argument("--audit-margin", type=int, default=4, help="paires listées jusqu'à threshold + marge")
    args = ap.parse_args()

    reg, D = build_registry(args.data_dir, root=".", threshold=args.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(out, index=False)

    # Paires proches, pour contrôle visuel (seuil + marge)
    ii, jj = np.where(np.triu(D <= args.threshold + args.audit_margin, k=1))
    pairs = reg[["image_id"]].iloc[ii].reset_index(drop=True).rename(columns={"image_id": "image_a"})
    pairs["image_b"] = reg["image_id"].iloc[jj].to_numpy()
    pairs["hamming"] = D[ii, jj]
    pairs["merged"] = pairs["hamming"] <= args.threshold
    pairs.sort_values("hamming").to_csv(out.parent / "duplicate_pairs.csv", index=False)

    sizes = reg.groupby("group_id").size()
    print(f"images                : {len(reg)}")
    print(f"boîtes (clippées)     : {int(reg['n_boxes'].sum())}  (brutes {int(reg['n_boxes_raw'].sum())}, "
          f"dépassant {int(reg['n_boxes_clipped'].sum())}, dégénérées {int(reg['n_boxes_degenerate'].sum())})")
    print(f"images non annotées   : {int(reg['image_unannotated'].sum())}")
    print(f"dimensions distinctes : {sorted(set(zip(reg['width'], reg['height'])))}")
    print(f"groupes               : {len(sizes)}  (singletons {(sizes == 1).sum()}, max {sizes.max()} images)")
    print("taille des groupes    :", sizes.value_counts().sort_index().to_dict())
    print("distance au plus proche voisin (quantiles 0/5/25/50) :",
          np.percentile(reg["nn_distance"], [0, 5, 25, 50]).tolist())
    print(f"paires fusionnées     : {int(pairs['merged'].sum())} -> {out.parent / 'duplicate_pairs.csv'}")
    if sizes.max() > 0.05 * len(reg):
        print("ATTENTION : un groupe dépasse 5 % des images -> seuil probablement trop haut (chaînage).")


if __name__ == "__main__":
    main()