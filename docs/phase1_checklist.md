# Phase 1 : liste de contrôle de fin de phase

Rendu : dimanche 15/11/2026. Prototype de recherche, aucun diagnostic clinique.

## A. Dépôt Git

- [x] `ag/phase1-dataset-registry` fusionnée dans `main` (PR, commit de fusion, sans squash)
- [x] `ag/phase1-candidate-labeling` fusionnée
- [x] `ag/phase1-geometry-features` fusionnée
- [x] `feature/phase1-appearance-features` fusionnée
- [x] `ag/phase1-build-dataset` fusionnée
- [x] `main` à jour en local, `git status` propre
- [x] `python -m pytest -q` passe en entier
- [x] Tag `v0.2-phase1` posé sur `main` et poussé

## B. Registre et découpage

- [x] `data/registry/images.csv` : 1 182 images, `group_id`, `split`
- [x] `data/registry/duplicate_pairs.csv` et audit visuel `reports/phase_1/duplicate_audit.csv`
- [x] `data/splits/split_v1.csv` : 70/15/15, graine 42, aucun groupe sur deux partitions
- [x] `docs/dataset_registry.md` : section 5 complétée (tableau du découpage, limites)

## C. Détecteur de candidats

- [x] Paramètres gelés dans `configs/candidates_v1.json`
- [x] Rappel mesuré : train 0,991, val 0,994 (`reports/phase_1/candidate_recall.json`)
- [x] Grille de réglage conservée : `reports/phase_1/candidate_grid.csv`
- [x] Test non consulté pour le réglage

## D. Table de candidats

- [x] `python scripts/build_dataset.py` : 21 611 candidats, 425 variables, 0 NaN, contrôles anti-fuite OK
- [x] `reports/phase_1/dataset_summary.json` versionné
- [x] `data/processed/` hors dépôt (vérifier `.gitignore`)
- [ ] Lecture par `src/evaluation/dataset.load_table` testée sur la vraie table

## E. Documentation

- [x] `docs/phase1_synthese.md` (note de synthèse, sans crochets)
- [x] `docs/decisions.md` : lignes registre canonique, départage des boîtes, contrat avec `src/evaluation`, variables constantes, gel du détecteur
- [x] `docs/conventions.md` : section Phase 1 cohérente avec le code réel
- [x] `README.md` : état de la Phase 1, commandes de reconstruction
- [x] `docs/dataset_registry.md` complété (voir B)

## F. Reproductibilité

- [x] Depuis un clone propre : `pip install -r requirements.txt`, puis `build_registry`, `make_split`, `build_dataset` reproduisent les mêmes chiffres
- [x] Les données brutes ne sont pas dans le dépôt ; leur emplacement est documenté
- [x] Source et licence du dataset vérifiées et citées dans le README
