"""Mesure R_cand du détecteur figé (configs/candidates_v1.json) sur train et val.

Usage : python scripts/eval_candidates.py
Le jeu de test n'est PAS évalué ici (il n'est utilisé qu'une fois, en fin de Phase 2).
Sortie : reports/phase_1/candidate_recall.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.candidate_eval import evaluate_samples, iter_samples, load_params, params_to_dict  # noqa: E402
from src.data.registry import load_registry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/candidates_v1.json")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--out", default="reports/phase_1/candidate_recall.json")
    args = ap.parse_args()

    params = load_params(args.config)
    reg = load_registry()
    result = {"config": args.config, "params": params_to_dict(params), "splits": {}}
    for s in args.splits:
        res = evaluate_samples(iter_samples(reg[reg["split"] == s]), params)
        result["splits"][s] = res
        print(f"{s:5s}: {res['n_images']} images | R_cand = {res['recall']:.4f} "
              f"({res['n_found']}/{res['n_gt']}) | {res['cand_per_image']:.1f} candidats/image | "
              f"pureté {res['candidate_precision']:.3f} | images sans candidat {res['images_without_candidate']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"écrit : {args.out}")


if __name__ == "__main__":
    main()