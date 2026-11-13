# Dataset registry : PaluScope

> Prototype de recherche. Aucun diagnostic clinique.

## 1. Source
- **Nom** : Uganda Mobile Thick / Makerere `plasmodium-phonecamera` (dossier `data/raw/plasmodium_phonecamera/`).
- **Source officielle, référence, licence** : à compléter et à vérifier sur la page officielle du jeu de données avant toute diffusion.
- **Domaine** : goutte épaisse, images acquises avec un smartphone sur microscope (`dataset_source = makerere_uganda_mobile_thick`, `device = smartphone_on_microscope`, `smear_type = thick`).

## 2. Contenu
| Élément | Valeur |
|---|---|
| Images | 1 182 JPEG, toutes en 750×750 |
| Annotations | Pascal VOC, une classe (`plasmodium`), pas de balise `<size>` |
| Boîtes | 7 628 (7 245 annoncées par les auteurs), ~40×40 px |
| Boîtes dépassant de l'image | 408, à clipper (jamais arrondies sur place) |
| Images sans annotation | 234 (négatives ou simplement non annotées : inconnu) |

## 3. Registre (`data/registry/images.csv`)
Une ligne par image : `image_id`, `image_path`, `xml_path`, `width`, `height`, `n_boxes_raw`, `n_boxes` (après clipping), `n_boxes_clipped`, `n_boxes_degenerate`, `image_unannotated`, `dataset_source`, `device`, `smear_type`, `phash`, `group_id`, `nn_distance`. Généré par `scripts/build_registry.py`.

## 4. Quasi-doublons et groupes
- Aucun identifiant de patient ou de lame n'est fourni : la règle « pas de vues d'un même patient/lame dans plusieurs partitions » ne peut pas être appliquée exactement.
- Approximation : signature pHash (64 bits) ; deux images sont dans le même groupe si leur distance de Hamming est ≤ 8 (composantes connexes).
- Résultat : 1 131 groupes (1 087 images isolées, 39 paires, 3 triplets, 2 quadruplets).
- Contrôle visuel (`scripts/audit_duplicates.py`, 22 paires) : les distances 0 et 2 correspondent à de vrais doublons (même champ, cadrage décalé, annotations non identiques) ; à 6 et 8, beaucoup de fausses fusions ; à 10 et 12, aucune vraie dans l'échantillon. Seuil 8 conservé par prudence : une fausse fusion est sans conséquence, un doublon manqué crée une fuite.
- **Limite connue** : des doublons plus fortement décalés peuvent échapper au regroupement. Les performances restent des performances *in-domain* sur un seul jeu d'images.

## 5. Partitions (`data/splits/split_v1.csv`)
Séparation par groupe, stratifiée sur le nombre moyen de boîtes par image, 70/15/15, graine 42, figée (`--force` obligatoire pour la régénérer). Le test n'est utilisé qu'une fois, en fin de Phase 2.

| Partition | Images | Groupes | Boîtes | Non annotées | Boîtes/image |
|---|---|---|---|---|---|
| train | 827 | 776 | 5 354 | 166 | 6,47 |
| val | 179 | 179 | 1 162 | 34 | 6,49 |
| test | 176 | 176 | 1 112 | 34 | 6,32 |

Tous les groupes de plus d'une image sont dans `train` (les gros groupes sont placés en premier) : `val` et `test` ne contiennent que des images sans quasi-doublon détecté.

## 6. Reproduire
```bash
python scripts/build_registry.py --threshold 8
python scripts/audit_duplicates.py
python scripts/make_split.py
```