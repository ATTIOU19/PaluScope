"""Construit la table de candidats (features + étiquettes) en une commande.

Usage :
    python scripts/build_dataset.py                     # normalisation 'none' (référence)
    python scripts/build_dataset.py --norm gray_world   # variante pour l'ablation de la Phase 2
    python scripts/build_dataset.py --limit 5           # test rapide : 5 images par partition
Sorties (hors dépôt) : data/processed/candidates_v1[_<norm>].csv.gz, patches_v1.npz
Sortie (dépôt)       : reports/phase_1/dataset_summary.json
Le script s'arrête avec un code d'erreur si un contrôle anti-fuite / qualité échoue.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.candidate_eval import load_params  # noqa: E402
from src.data.registry import load_registry  # noqa: E402
from src.features.builder import build_dataset, load_features_config  # noqa: E402
from src.features.checks import check_dataset, dataset_summary  # noqa: E402
from src.preprocessing.normalize import NORM_MODES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates-config", default="configs/candidates_v1.json")
    ap.add_argument("--features-config", default="configs/features_v1.json")
    ap.add_argument("--norm", choices=NORM_MODES, default=None, help="remplace norm_mode du fichier de config")
    ap.add_argument("--limit", type=int, default=0, help="N images par partition (test rapide)")
    ap.add_argument("--no-patches", action="store_true")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--summary", default="reports/phase_1/dataset_summary.json")
    args = ap.parse_args()

    cfg = load_features_config(args.features_config)
    norm = args.norm or cfg.get("norm_mode", "none")
    params = load_params(args.candidates_config)
    reg = load_registry()
    if args.limit:
        reg = reg.sort_values("image_id").groupby("split").head(args.limit)

    print(f"construction : {len(reg)} images, normalisation '{norm}', familles {cfg['families']}")
    df, patches = build_dataset(
        reg, params, norm_mode=norm, families=tuple(cfg["families"]), shrink=cfg.get("label_shrink", 1.0),
        patch_size=cfg.get("patch_size", 48), candidates_cfg=Path(args.candidates_config).stem,
        features_cfg=Path(args.features_config).stem, progress_every=100,
    )

    problems = check_dataset(df, reg)
    summary = dataset_summary(df)
    suffix = "" if norm == "none" else f"_{norm}"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"candidates_v1{suffix}.csv.gz", index=False, float_format="%.6g")
    if not args.no_patches:
        np.savez_compressed(out / "patches_v1.npz", patches=patches, candidate_id=df["candidate_id"].to_numpy(dtype=str))
    if not args.limit:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("n_candidates", "per_split", "n_features", "n_nan_features")},
                     indent=2, ensure_ascii=False))
    if problems:
        print("\nCONTRÔLES EN ÉCHEC :\n- " + "\n- ".join(problems))
        raise SystemExit(1)
    print("\ncontrôles anti-fuite et qualité : OK")


if __name__ == "__main__":
    main()