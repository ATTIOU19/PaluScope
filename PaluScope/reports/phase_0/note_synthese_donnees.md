# PaluScope - Phase 0 - Note de synthèse sur les données

**Bloc A · Responsable : Membre 1 · Notebook associé : `notebooks/phase_0/01_data_exploration.ipynb`**

## 1. Dataset de départ

- **Nom :** Microscopy Malaria Dataset, split `plasmodium-phonecamera`
- **Auteurs :** J. Quinn, R. Nakasi, P. K. B. Mugagga, P. Byanyima, W. Lubega, A. Andama (Makerere University AI Lab, Ouganda), 2016
- **Source :** https://air.ug/microscopy_dataset/
- **Licence :** CC0 1.0
- **Nature des images :** frottis sanguins épais (Field stain), grossissement ×1000, photographiés avec un smartphone monté sur un microscope
- **Provenance indiquée dans les annotations :** Makerere Automated Lab Diagnostics Database, Mulago National Referral Hospital

## 2. Inventaire

| Élément | Résultat |
|---|---|
| Images (`.jpg`) | 1182 |
| Annotations (`.xml`) | 1182 |
| Images sans annotation | 0 |
| Annotations sans image | 0 |
| Fichier sans extension | 1 (à identifier) |

La correspondance entre images et annotations est complète (même nom de fichier, ex. `plasmodium-phone-0001`).

## 3. Images

- Dimensions uniques : **750 × 750 pixels**, mode **RGB**, format **JPEG**
- Poids : 84 à 125 Ko par image (moyenne 102 Ko)
- Aucune harmonisation de taille nécessaire

## 4. Annotations

- Format de type **Pascal VOC** (XML) : une balise `<object>` par parasite, avec `<label>` et `<bndbox>` (`xmin`, `ymin`, `xmax`, `ymax`)
- **Une seule classe** : `plasmodium`
- **7628 boîtes** lues par notre parseur
- **948 images** contiennent au moins un parasite, **234 n'en contiennent aucun**
- Parasites par image : moyenne 6,45 · médiane 6 · maximum 27 (environ 8 par image positive)

> À confirmer : la description du dataset mentionne 7245 plasmodiums, alors que le comptage de nos annotations donne 7628. Nous retenons notre comptage et signalons l'écart.

### Particularités des boîtes
- **Taille quasi constante :** médiane et quartiles à 40 × 40 px (écart-type d'environ 2,5 px). Ces boîtes semblent être de taille fixe centrées sur chaque parasite (hypothèse). **Elles ne donnent donc pas la taille réelle des parasites** et ne doivent pas servir à calibrer les filtres de taille.
- **Boîtes qui sortent de l'image :** `xmin` descend jusqu'à −19,3, `ymin` jusqu'à −20, `xmax` et `ymax` montent jusqu'à 770, alors que l'image fait 750 px. Certaines boîtes sont tronquées (largeur minimale 1,27 px). Elles devront être **clippées** aux bornes de l'image avant toute comparaison avec des candidats.
  - Nombre de boîtes hors image : *[à compléter après exécution de la cellule de clipping]*

## 5. Variabilité visuelle

- Les colorations varient fortement d'une image à l'autre : certaines sont rosées ou beiges, d'autres violettes ou bleutées.
- Les parasites apparaissent comme de très petites taches sombres, à l'intérieur de boîtes de 40 px.
- Certaines images sont très pâles et peu contrastées, comme `plasmodium-phone-0502`.
- Des images sans parasite annoté existent, comme `plasmodium-phone-0564`.

### Mesures exploratoires (échantillon de 200 images)

| Mesure | Moyenne | Min – Max |
|---|---|---|
| Luminosité (niveaux de gris) | 147 | 126 – 165 |
| Contraste (écart-type) | 15,7 | 4,2 – 38,5 |
| Variance du Laplacien (netteté) | 9,9 | 7,0 – 16,7 |

- La luminosité varie peu : pas d'image franchement sombre ou surexposée dans l'échantillon.
- Le contraste varie davantage.
- La variance du Laplacien est faible et resserrée : elle sépare mal les images nettes des floues. D'autres mesures de netteté seront à tester (Membre 2).
- Ces chiffres sont **indicatifs** : l'échantillon est aléatoire et le choix des mesures finales relève du Quality Analyzer.

## 6. Anomalies et limites

1. Écart entre le nombre de parasites annoncé (7245) et celui compté (7628)
2. Boîtes qui dépassent les bornes de l'image (408 boîtes hors image)
3. Boîtes de taille fixe : pas de taille réelle de parasite
4. Un fichier sans extension à identifier (fichier .gitkeep)
5. Aucune étiquette de qualité (flou, exposition) : la validation des mesures se fera par comparaison visuelle
6. Frottis épais uniquement, un seul type de coloration et de dispositif : la généralisation à d'autres domaines n'est pas démontrée

## 7. Conséquences pour la suite

- **Quality Analyzer (Membre 2) :** prévoir des mesures de netteté plus discriminantes que le Laplacien seul, et tenir compte de la forte variabilité de couleur.
- **Candidats (Membre 3) :** calibrer la taille des régions à partir des parasites réels sur l'image, pas à partir des boîtes de 40 px. Les 234 images sans parasite serviront à estimer les faux positifs.
- **Évaluation (phases ultérieures) :** clipper les boîtes avant comparaison.
- **Intégration (Membre 1) :** conserver `data/samples/` avec des images variées (pâles, très colorées, sans parasite, très chargées).

## 8. À étudier plus tard

- Origine exacte de l'écart 7245 / 7628
- Contenu du fichier sans extension
- Normalisation de la couleur entre images (Phase 1)
