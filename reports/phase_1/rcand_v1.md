# Détecteur de candidats figé (candidates_v1)

Réglage choisi sur le TRAIN uniquement (objectif Rcand >= 0.98, puis le moins de candidats).

```json
{
  "channel": "gray",
  "smooth_ksize": 3,
  "blackhat_ksize": 15,
  "min_contrast": 20.0,
  "open_ksize": 3,
  "close_ksize": 3,
  "min_area": 10.0,
  "max_area": 200.0,
  "min_circularity": 0.3
}
```

| Paquet | Images | Boîtes annotées | Boîtes avec >= 1 candidat | Rcand | Candidats / image |
|---|---|---|---|---|---|
| train | 827 | 5354 | 5299 | 0.9897 | 17.4 |
| val | 179 | 1162 | 1154 | 0.9931 | 17.0 |

Rcand = parasites annotés ayant au moins un candidat dont le centre tombe dans leur boîte.
Les parasites sans candidat sont perdus pour toute la chaîne : Rcand est le plafond de la sensibilité.
Le test n'a pas été utilisé.
