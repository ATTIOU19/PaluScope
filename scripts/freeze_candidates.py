"""Gel du détecteur de candidats (bloc P1-B, Membre 3).

Ce script :
  1. lit le découpage data/splits/split_v1.csv ;
  2. essaie quelques réglages du détecteur sur les images de TRAIN seulement ;
  3. garde celui qui retrouve assez de parasites (Rcand >= RCAND_MIN) avec le moins de candidats ;
  4. l'écrit dans configs/candidates_v1.json ;
  5. mesure Rcand sur le train et la validation (le test n'est JAMAIS utilisé) et écrit un petit rapport.

Lancer depuis la racine du dépôt :  python scripts/freeze_candidates.py
"""
import csv
import json
import sys
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.candidates.evaluate import clip_boxes, match_candidates, parse_boxes
from src.candidates.extract import CandidateParams, extract_candidates

SPLIT_FILE = ROOT / "data" / "splits" / "split_v1.csv"
RAW_DIR = ROOT / "data" / "raw"
OUT_CONFIG = ROOT / "configs" / "candidates_v1.json"
OUT_REPORT = ROOT / "reports" / "phase_1" / "rcand_v1.md"
RCAND_MIN = 0.98                       # objectif : au moins 98 % des parasites ont un candidat
GRILLE = {                             # petits réglages à comparer (sur le train)
    "min_contrast": [16.0, 20.0, 24.0],
    "max_area": [150.0, 200.0],
}


def lire_decoupage():
    """Renvoie {image_id: split}."""
    with open(SPLIT_FILE, newline="", encoding="utf-8") as f:
        return {ligne["image_id"]: ligne["split"] for ligne in csv.DictReader(f)}


def preparer(split_par_image, nom_split):
    """Liste de (chemin_image, boîtes_clippées) pour les images d'un paquet."""
    images = {p.stem: p for p in RAW_DIR.rglob("*.jpg")}
    xmls = {p.stem: p for p in RAW_DIR.rglob("*.xml")}
    items = []
    for image_id, split in sorted(split_par_image.items()):
        if split != nom_split or image_id not in images:
            continue
        largeur, hauteur = Image.open(images[image_id]).size
        brutes = parse_boxes(xmls[image_id]) if image_id in xmls else []
        items.append((images[image_id], clip_boxes(brutes, largeur, hauteur)))
    return items


def evaluer(params, items):
    """Rcand = parasites annotés ayant au moins un candidat / parasites annotés."""
    n_gt = n_trouves = n_cand = 0
    for chemin, boites in items:
        image = np.array(Image.open(chemin).convert("RGB"))
        r = match_candidates(extract_candidates(image, params), boites)
        n_gt += r["n_gt"]
        n_trouves += r["n_found"]
        n_cand += r["n_candidates"]
    return {"rcand": n_trouves / max(1, n_gt), "cand_par_image": n_cand / max(1, len(items)),
            "n_images": len(items), "n_boites": n_gt, "n_boites_trouvees": n_trouves}


def main():
    split_par_image = lire_decoupage()
    noms = sorted(set(split_par_image.values()))
    print("Paquets trouvés dans split_v1.csv :", noms)
    nom_val = next((n for n in noms if n in ("val", "valid", "validation")), None)
    if "train" not in noms or nom_val is None:
        sys.exit("Il me faut les paquets 'train' et 'val' dans split_v1.csv (voir la ligne ci-dessus).")
    train, val = preparer(split_par_image, "train"), preparer(split_par_image, nom_val)
    if not train:
        sys.exit(f"Aucune image du train trouvée dans {RAW_DIR}. Le dataset est-il bien copié dans data/raw ?")
    print(f"Images : train = {len(train)}, {nom_val} = {len(val)} (le test n'est pas utilisé)\n")

    resultats = []
    for valeurs in product(*GRILLE.values()):
        params = replace(CandidateParams(), **dict(zip(GRILLE, valeurs)))
        m = evaluer(params, train)
        resultats.append((params, m))
        print(f"  {dict(zip(GRILLE, valeurs))} -> Rcand = {m['rcand']:.4f}, candidats/image = {m['cand_par_image']:.1f}")

    bons = [r for r in resultats if r[1]["rcand"] >= RCAND_MIN]
    choix = min(bons, key=lambda r: r[1]["cand_par_image"]) if bons else max(resultats, key=lambda r: r[1]["rcand"])
    params, m_train = choix
    if not bons:
        print(f"\nATTENTION : aucun réglage n'atteint Rcand >= {RCAND_MIN} ; on garde le meilleur Rcand.")
    m_val = evaluer(params, val)

    OUT_CONFIG.parent.mkdir(exist_ok=True)
    OUT_CONFIG.write_text(json.dumps(asdict(params), indent=2) + "\n", encoding="utf-8")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    lignes = ["# Détecteur de candidats figé (candidates_v1)", "",
              f"Réglage choisi sur le TRAIN uniquement (objectif Rcand >= {RCAND_MIN}, puis le moins de candidats).", "",
              "```json", json.dumps(asdict(params), indent=2), "```", "",
              "| Paquet | Images | Boîtes annotées | Boîtes avec >= 1 candidat | Rcand | Candidats / image |",
              "|---|---|---|---|---|---|"]
    for nom, mm in (("train", m_train), (nom_val, m_val)):
        lignes.append(f"| {nom} | {mm['n_images']} | {mm['n_boites']} | {mm['n_boites_trouvees']} | "
                      f"{mm['rcand']:.4f} | {mm['cand_par_image']:.1f} |")
    lignes += ["", "Rcand = parasites annotés ayant au moins un candidat dont le centre tombe dans leur boîte.",
               "Les parasites sans candidat sont perdus pour toute la chaîne : Rcand est le plafond de la sensibilité.",
               "Le test n'a pas été utilisé."]
    OUT_REPORT.write_text("\n".join(lignes) + "\n", encoding="utf-8")

    print(f"\nRéglage retenu : {asdict(params)}")
    print(f"Rcand train = {m_train['rcand']:.4f} | Rcand {nom_val} = {m_val['rcand']:.4f}")
    print(f"Écrit : {OUT_CONFIG.relative_to(ROOT)} et {OUT_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()