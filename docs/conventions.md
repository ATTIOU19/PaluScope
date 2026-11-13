# Conventions des modules

## Entrée
Image chargée en tableau NumPy (RGB, uint8), ou chemin vers le fichier.

## Sortie du Quality Analyzer : `analyze_image(image) -> dict`
```python
{
    "sharpness": float,
    "brightness": float,
    "contrast": float,
    "overexposure_ratio": float,
    "color_statistics": dict,
}
```

## Sortie de la détection de candidats : liste de dict
```python
{
    "id": int,
    "bbox": (x, y, w, h),
    "area": float,
    "perimeter": float,
    "circularity": float,   # 4*pi*A / P^2
}
```
# Phase 1 : contrats d'interface

Conventions générales (inchangées) : image = tableau NumPy RGB, uint8, (H, W, 3) ;
`cv2.imread` donne du BGR, il faut convertir à l'entrée. Boîtes d'un candidat : `(x, y, w, h)`.
Boîtes annotées : dictionnaires `xmin, ymin, xmax, ymax` (format de `clip_boxes`).

## Registre (P1-A) : `src/data/register.py`

```python
load_registry(path=None) -> pd.DataFrame
# une ligne par image : image_id, path, width, height, n_boxes, n_boxes_clipped,
# n_boxes_dropped, image_unannotated, has_xml, read_error,
# dataset_source, device, smear_type, group_id, split

load_boxes(path=None, image_id=None) -> pd.DataFrame
# une ligne par boîte clippée : box_id, image_id, label, x, y, w, h,
# raw_xmin, raw_ymin, raw_xmax, raw_ymax, was_clipped
```

`group_id` et `split` sont vides tant que `split_v1` n'est pas figé.

## Candidats (P1-B) : `src/candidates/extract.py` (existant)

```python
extract_candidates(image_rgb, params=CandidateParams()) -> list[dict]
# {"id", "bbox": (x, y, w, h), "area", "perimeter", "circularity"}
# Extension proposée (à valider avec Aichatou) : champ OPTIONNEL "contour",
# sans changer les champs existants (compatibilité démo V0 et tests).
```

Les paramètres figés sont lus dans `configs/candidates_v1.json` :
`CandidateParams(**json.load(f))`.

## Étiquetage (P1-B) : `src/data/labeling.py`

```python
label_candidates(candidates, boxes) -> list[dict]
# candidates : sortie de extract_candidates
# boxes      : liste de {"box_id", "xmin", "ymin", "xmax", "ymax"} (boîtes DÉJÀ clippées)
# renvoie une COPIE de chaque candidat avec en plus :
#   "label"        : int, 1 si le centre du candidat est dans une boîte, sinon 0
#   "match_box_id" : box_id de la boîte retenue, ou None
```

Règles : bornes incluses (comme `match_candidates`) ; si le centre tombe dans
plusieurs boîtes, on retient celle dont le centre est le plus proche, puis le
`box_id` le plus petit (résultat déterministe). Les entrées ne sont pas modifiées.

## Patches (P1-B) : `src/data/patches.py`

```python
extract_patch(image, bbox, size=48) -> np.ndarray   # uint8, (size, size, 3)
```

Centre du candidat : `cx = x + w/2`, `cy = y + h/2`. Coin haut gauche du patch :
`floor(cx - size/2 + 0.5)` (idem pour y). Remplissage par réflexion si le patch
dépasse l'image. Aucune modification de l'image d'entrée.

## Normalisation (P1-C) : `src/preprocessing/normalize.py`

```python
normalize_image(image, mode="none") -> np.ndarray   # uint8, même forme
# mode : "none" | "gray_world" | "channel_standardize" | ("reinhard", optionnel)
```

La normalisation s'applique à l'image ENTIÈRE avant extraction des features.
Tout objet appris (si Reinhard avec une référence) est ajusté sur le train uniquement.

## Features (P1-B et P1-C) : `src/features/`

```python
compute_features(image, candidate, cfg) -> dict[str, float]
# image : RGB uint8, DÉJÀ normalisée selon cfg["norm_mode"]
# clés préfixées : geo_, col_, grd_, tex_, hog_ (qual_ seulement si cfg l'active)
```

Règles : fonction pure ; mêmes clés, dans le même ordre, pour tous les
candidats d'une même configuration ; jamais de NaN ni d'infini (valeur de
repli explicite, comptée par le builder) ; indépendante du jeu de données.
Jeux nommés : F1 = geo_, F2 = F1 + col_, F3 = F2 + grd_ + tex_, F4 = F3 + hog_.

Configuration : `configs/features_v1.json`, par exemple
`{"norm_mode": "none", "families": ["geo", "col", "grd", "tex"], "patch_size": 48, "use_qual": false}`.

## Assemblage (P1-D) : `src/features/builder.py`, `scripts/build_dataset.py`

```python
build_dataset(cfg) -> pd.DataFrame
```

Écrit `data/processed/dataset_v1.csv` (non versionné) et un rapport de
contrôles dans `reports/phase_1/`. Une ligne = un candidat. Colonnes :
`image_id, candidate_id, split, group_id, dataset_source, device, smear_type,
image_unannotated, x, y, w, h, label, match_box_id, <features>, norm_mode,
candidates_cfg, features_cfg`. Déterministe : deux exécutions, même table.

## Évaluation (P2-A) : `src/evaluation/metrics.py`

```python
evaluate(y_true, y_score, thr) -> dict
# precision, recall, specificity, f1, tn, fp, fn, tp,
# roc_auc, pr_auc, prevalence, thr
```

Prédiction positive si `y_score >= thr`. Dénominateur nul : la métrique vaut
`None` (jamais NaN silencieux). `roc_auc` et `pr_auc` valent `None` si une seule
classe est présente dans `y_true`.
