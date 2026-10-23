# Journal des décisions
| 2026-10-01 | Dataset de départ | plasmodium-phonecamera (Makerere AI Lab) : images smartphone de frottis épais, annotations par boîtes, licence CC0. Adapté à l'étude de la qualité d'image et à l'extraction de candidats sans ML. |
| 2026-10-08 | Stratégie de candidats | Image grise lissée, puis black-hat (noyau elliptique de 15 px), seuil d'Otsu avec plancher à 20, ouverture et fermeture (3 px), composantes connexes (connectivité 8), puis filtres d'aire (10 à 150 px²) et de circularité (≥ 0,3). Le black-hat isole les petites taches sombres quelle que soit la teinte du frottis. |
| 2026-10-08 | Choix des paramètres | Configuration par défaut retenue : rappel 0,99, environ 17,9 candidats par image, environ 6,4 sur les images sans parasite. Comparée au canal vert (0,97), au contraste min 12 (1,00 mais 38,5 candidats par image), au contraste min 30 (0,93) et au black-hat 9 ou 21 (0,85). Priorité au rappel, puis au nombre de candidats. |
| 2026-10-08 | Bornes d'aire | Aire de 10 à 150 px², fixée d'après les aires des candidats tombant dans des boîtes annotées (section 7 du notebook). Les boîtes de 40×40 px ne servent pas à régler la taille. |
| 2026-10-08 | Convention des sorties | Liste de dictionnaires {id, bbox (x, y, w, h), area, perimeter, circularity}, conforme à docs/conventions.md. |
| 2026-10-23 | Méthode de mesure du flou | laplacian, seuil 7.381 (centile 5 du dataset). Comparée à tenengrad et reblur sur flou gaussien synthétique (σ de 0.5 à 3.0 px), section 5. |
| 2026-10-23| Indicateur de contraste | percentile, seuil 23 (centile 5 du dataset). |
| 2026-10-23 | Détection de surexposition | part de pixels de gris ≥ 250 > 5%, et luminosité hors de [80, 200]. Aucune saturation dans le dataset (0 images concernées) : validée par éclaircissement synthétique, section 6. |
| 2026-10-23 | Convention des sorties (qualité) | cinq clés de docs/conventions.md + underexposure_ratio + flags (None = seuil non calibré). |
