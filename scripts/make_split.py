"""Fige la séparation train/val/test par groupe -> data/splits/split_v1.csv.

Usage : python scripts/make_split.py            (graine 42, 70/15/15)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.registry import DEFAULT_REGISTRY, DEFAULT_SPLIT  # noqa: E402
from src.data.split import assert_no_leakage, make_split, split_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--out", default=DEFAULT_SPLIT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    ap.add_argument("--force", action="store_true", help="écraser un split existant (déconseillé : il est figé)")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} existe déjà : le split est figé. Utiliser --force pour le régénérer volontairement.")

    reg = pd.read_csv(args.registry, keep_default_na=False)
    sp = make_split(reg, ratios=tuple(args.ratios), seed=args.seed)
    assert_no_leakage(sp)
    out.parent.mkdir(parents=True, exist_ok=True)
    sp.to_csv(out, index=False)
    print(split_report(reg, sp).to_string())
    print(f"\nécrit : {out}  (graine {args.seed}, ratios {tuple(args.ratios)})")


if __name__ == "__main__":
    main()