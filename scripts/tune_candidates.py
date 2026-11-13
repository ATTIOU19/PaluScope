"""Cherche les réglages du détecteur de candidats SUR LE TRAIN UNIQUEMENT (sous-échantillon).

Usage :
    python scripts/tune_candidates.py --quick                 # test rapide du montage
    python scripts/tune_candidates.py --max-cand 80           # grille complète
    python scripts/tune_candidates.py --from-csv --max-cand 60 --write   # rechoisir sans recalculer
    python scripts/tune_candidates.py --max-cand 80 --write   # fige configs/candidates_v1.json

Règle de choix : parmi les réglages avec au plus --max-cand candidats par image, on prend
celui qui a le moins de candidats parmi ceux dont le rappel est à moins de 0,5 point du meilleur.
Le tableau complet est écrit dans reports/phase_1/candidate_grid.csv : le regarder avant --write.
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.candidates.extract import CandidateParams  # noqa: E402
from src.data.candidate_eval import evaluate_samples, iter_samples, params_to_dict, save_params  # noqa: E402
from src.data.registry import load_registry  # noqa: E402

GRID = {
    "channel": ["gray", "green"],
    "blackhat_ksize": [11, 15, 21],
    "min_contrast": [10.0, 15.0, 20.0],
    "min_area": [6.0, 10.0],
    "max_area": [150.0, 250.0],
}
QUICK_GRID = {"channel": ["gray", "green"], "min_contrast": [15.0, 20.0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-images", type=int, default=150, help="taille du sous-échantillon du train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cand", type=float, default=80.0, help="budget : candidats par image au maximum")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--from-csv", action="store_true", help="ne pas recalculer : reprendre --out et rechoisir")
    ap.add_argument("--write", action="store_true", help="écrire configs/candidates_v1.json")
    ap.add_argument("--out", default="reports/phase_1/candidate_grid.csv")
    ap.add_argument("--config-out", default="configs/candidates_v1.json")
    args = ap.parse_args()

    grid = QUICK_GRID if args.quick else GRID
    if args.from_csv:
        df = pd.read_csv(args.out)
        n_used = args.n_images
    else:
        reg = load_registry()
        train = reg[reg["split"] == "train"]
        sub = train.sample(n=min(args.n_images, len(train)), random_state=args.seed).sort_values("image_id")
        n_used = len(sub)
        samples = list(iter_samples(sub))  # chargées une fois pour toute la grille
        rows = []
        combos = list(itertools.product(*grid.values()))
        for k, combo in enumerate(combos, 1):
            overrides = dict(zip(grid, combo))
            res = evaluate_samples(samples, CandidateParams(**overrides))
            rows.append({**overrides, **res})
            print(f"[{k}/{len(combos)}] {overrides} -> rappel {res['recall']:.3f}, {res['cand_per_image']:.1f} cand/img", flush=True)
        df = pd.DataFrame(rows)

    df = df.sort_values(["recall", "cand_per_image"], ascending=[False, True]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    grid_cols = [c for c in df.columns if c in GRID]
    cols = grid_cols + ["recall", "cand_per_image", "candidate_precision"]
    print("\nMeilleurs rappels (tous budgets) :\n", df[cols].head(8).round(3).to_string())

    ok = df[df["cand_per_image"] <= args.max_cand]
    if ok.empty:
        raise SystemExit(f"aucun réglage sous {args.max_cand} candidats/image : relever --max-cand")
    near = ok[ok["recall"] >= ok["recall"].max() - 0.005]
    best = near.sort_values("cand_per_image").iloc[0]
    chosen = CandidateParams(**{k: best[k].item() if hasattr(best[k], "item") else best[k] for k in grid_cols})
    print(f"\nChoisi (budget {args.max_cand}) : {params_to_dict(chosen)}")
    print(f"  rappel {best['recall']:.3f} ; {best['cand_per_image']:.1f} candidats/image ; "
          f"pureté {best['candidate_precision']:.3f}")
    if args.write:
        meta = {"selected_on": "train (sous-échantillon)", "n_images": n_used, "seed": args.seed,
                "max_cand_per_image": args.max_cand, "recall_sample": float(best["recall"]),
                "cand_per_image_sample": float(best["cand_per_image"])}
        save_params(chosen, args.config_out, meta)
        print(f"écrit : {args.config_out}")


if __name__ == "__main__":
    main()