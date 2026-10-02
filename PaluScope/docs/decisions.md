# Journal des décisions
| 2026-10-01 | Dataset de départ | plasmodium-phonecamera (Makerere AI Lab) : images smartphone de frottis épais, annotations par boîtes, licence CC0. Adapté à l'étude de la qualité d'image et à l'extraction de candidats sans ML. |

| Date | Sujet | Décision et justification |
| à compléter | Méthode de mesure du flou | laplacian, seuil 7.381 (centile 5 du dataset). Comparée à tenengrad et reblur sur flou gaussien synthétique (σ de 0.5 à 3.0 px), section 5. |
| à compléter | Indicateur de contraste | percentile, seuil 23 (centile 5 du dataset). |
| à compléter | Détection de surexposition | part de pixels de gris ≥ 250 > 5%, et luminosité hors de [80, 200]. Aucune saturation dans le dataset (0 images concernées) : validée par éclaircissement synthétique, section 6. |
| à compléter | Convention des sorties | cinq clés de docs/conventions.md + underexposure_ratio + flags (None = seuil non calibré). |
