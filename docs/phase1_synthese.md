# Phase 1 : note de synthèse du jeu de candidats (v1)

Date : 13/11/2026 · Prototype de recherche, aucun diagnostic clinique.
Table de référence : `data/processed/candidates_v1.csv.gz` (hors dépôt, reconstruite par `python scripts/build_dataset.py`).
Résumé chiffré versionné : `reports/phase_1/dataset_summary.json`.

## 1. Ce qui a été produit

- Une ligne par candidat : **21 611 candidats** issus de **1 144 images** (sur 1 182).
- **425 variables** en 5 familles : `geo_` 9, `col_` 70, `grd_` 3, `tex_` 14, `hog_` 324, plus 5 variables de contexte image `qual_` (comptées dans les 425, exclues des modèles par défaut).
- Détecteur de candidats gelé (`configs/candidates_v1.json`) : rappel des boîtes annotées 0,991 (train) et 0,994 (val). Le test n'a pas servi à régler le détecteur.
- Découpage `split_v1` : groupes de quasi-doublons (pHash), 70/15/15, graine 42. Aucun groupe ni aucune image ne figure dans deux partitions.
- Contrôles automatiques passés : aucune valeur NaN ou infinie, étiquettes cohérentes avec `match_box_id`, boîtes contenues dans l'image, `candidate_id` unique.

## 2. Répartition

| Partition | Images avec candidats | Candidats | Positifs | Part de positifs |
|---|---|---|---|---|
| train | 801 | 15 180 | 5 954 | 39,2 % |
| val | 172 | 3 181 | 1 329 | 41,8 % |
| test | 171 | 3 250 | 1 257 | 38,7 % |
| total | 1 144 | 21 611 | 8 540 | 39,5 % |

Candidats par image : moyenne 18,9, médiane 17, quartiles 9 et 26, maximum 92.

## 3. Déséquilibre

- Au niveau des candidats, les classes sont peu déséquilibrées (environ 40 % de positifs, 60 % de faux candidats). Ce n'est pas une prévalence de parasites : c'est la pureté du détecteur.
- La part de positifs diffère de 3 points entre val (41,8 %) et test (38,7 %). Le découpage est stratifié sur le nombre de boîtes par image, pas sur la part de candidats positifs. Toute métrique sensible à la prévalence (précision, PR-AUC) doit être comparée à la ligne de base de sa propre partition.
- La répartition par image est très inégale (1 à 92 candidats). Les images denses pèsent davantage dans toute moyenne calculée sur les candidats : les résultats seront aussi rapportés par image.
- Conséquence pour la Phase 2 : pas de rééchantillonnage au départ, poids de classes en option ; métriques retenues : PR-AUC, rappel à nombre fixé de faux positifs par image, jamais l'exactitude seule.

## 4. Étiquettes

- Règle : un candidat est positif si son centre tombe dans une boîte annotée clippée (boîte au centre le plus proche en cas de recouvrement).
- 8 540 candidats positifs pour 7 628 boîtes, soit environ 1,12 candidat positif par boîte. Les boîtes (environ 40×40 px, 1 600 px²) sont bien plus grandes que les candidats (10 à 150 px²) : plusieurs candidats, dont du bruit de coloration, peuvent tomber dans une même boîte. Les étiquettes sont donc généreuses. Le paramètre `label_shrink` (boîte réduite) permet de mesurer cette sensibilité en Phase 2.
- Les images sans annotation (234) reçoivent des candidats étiquetés 0 (`image_unannotated`) : 1 598 candidats sur 196 de ces images. On ne sait pas si elles sont réellement sans parasite ou simplement non annotées : à tester par une expérience avec et sans ces images.

## 5. Variables les plus séparatrices

AUC univariée sur le train uniquement (0,5 = aucune information ; une valeur sous 0,5 signifie que la variable est plus élevée chez les non-parasites).

| Variable | AUC | Lecture |
|---|---|---|
| `col_b_std`, `col_g_std`, `col_gray_std` | 0,774 / 0,773 / 0,772 | variation de couleur à l'intérieur du candidat |
| `col_v_std`, `col_r_std`, `col_s_std` | 0,768 / 0,761 / 0,741 | idem (autres canaux) |
| `col_g_diff`, `col_gray_diff`, `col_v_diff`, `col_r_diff`, `col_b_diff` | 0,228 à 0,267 | candidat plus sombre que son anneau de fond |
| `col_v_ratio`, `col_gray_ratio` | 0,267 / 0,277 | rapport au fond |
| `geo_area` | 0,718 | les candidats plus grands sont plus souvent positifs |
| `col_hsin_diff` | 0,283 | teinte (sinus) vs fond |

Lecture prudente :
- Aucune variable seule ne dépasse 0,78 en séparation. Ce n'est pas le signe d'une fuite d'étiquette (on verrait des valeurs proches de 1).
- Ces variables sont fortement corrélées entre elles (écart-type par canal, différence au fond par canal) : elles représentent peu de signaux indépendants, pas quinze.
- Les AUC sont descriptives. Les candidats d'une même image ne sont pas indépendants : ce ne sont pas des tests statistiques.
- Les variables HOG (324 colonnes) ne figurent pas dans ce classement : elles sont évaluées par ablation.

## 6. Anomalies et points d'attention

- 38 images (1 182 − 1 144) n'ont aucun candidat : 26 en train, 7 en val, 5 en test. Toutes sont des images non annotées. Les 948 images annotées ont toutes au moins un candidat.
- 1 598 candidats (7,4 % du total) se trouvent sur 196 images non annotées, soit 8,2 candidats par image contre 21,1 sur les images annotées (20 013 pour 948). Ces images ont donc nettement moins de candidats, ce qui est compatible avec des images en majorité sans parasite, sans le prouver. Tous ces candidats sont étiquetés 0.
- 7 variables sont constantes sur le train : six classes hautes des histogrammes de couleur (`col_r_hist7`, `col_g_hist6`, `col_g_hist7`, `col_b_hist5`, `col_b_hist6`, `col_b_hist7`), jamais atteintes parce que les candidats sont sombres, et `qual_overexposure_ratio`, nulle partout (aucune saturation dans le dataset, comme en Phase 0). Elles restent dans la table pour garder un contrat de colonnes stable ; la Phase 2 les retire avec un filtre de variance appris sur le train.
- Les variables `qual_` décrivent l'image entière et non le candidat. Leur AUC univariée sur le train va de 0,455 à 0,501 : `qual_brightness` 0,501, `qual_overexposure_ratio` 0,500 (constante), `qual_underexposure_ratio` 0,494, `qual_sharpness` 0,475, `qual_contrast` 0,455. Il n'y a pas d'effet d'image fort, mais le contraste et la netteté montrent un léger écart : les candidats négatifs viennent un peu plus souvent d'images contrastées et nettes. Les `qual_` restent exclues par défaut ; une ablation avec et sans elles mesurera ce signal en Phase 2.
- HOG représente 324 des 425 colonnes (76 %). Le jeu F4 est dominé par HOG : l'ablation F3 contre F4 est obligatoire.
- Normalisation couleur : `none` par défaut. `gray_world` et `standardize` sont comparés en Phase 2 et pour la robustesse (Phase 3).

## 7. Limites

- Pas d'identifiant de lame ni de patient. Les groupes sont des quasi-doublons d'images (pHash, distance de Hamming ≤ 8). Un risque résiduel de fuite subsiste pour des doublons très décalés.
- Un seul dataset, un seul type d'appareil (smartphone sur microscope), un seul type de frottis (épais) : aucune conclusion de généralisation n'est possible à ce stade.
- Les étiquettes sont dérivées de boîtes grossières, pas d'annotations au niveau du parasite.
- Le détecteur est réglé sur train et val ; le test reste à utiliser une seule fois, en fin de Phase 2.
