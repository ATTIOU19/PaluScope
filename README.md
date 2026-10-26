<p align="center">
  <img src="docs/assets/banner.svg" alt="PaluScope : microscopie intelligente du paludisme" width="100%">
</p>

<p align="center">
  <img alt="Phase" src="https://img.shields.io/badge/phase-0%20%C2%B7%20V0-6f42c1">
  <img alt="Statut" src="https://img.shields.io/badge/statut-en%20d%C3%A9veloppement-orange">
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white">
</p>

> **Prototype de recherche et de formation.** PaluScope n'est pas un dispositif médical : il ne pose aucun diagnostic clinique et ne prescrit aucun traitement. Il assiste la lecture de frottis sanguins par un humain.

PaluScope est un prototype de recherche et d'assistance qui analyse des images de frottis sanguins : il contrôle leur
qualité, repère les parasites suspects, les compte, et mesure la robustesse du système quand on change de microscope,
de téléphone, de laboratoire ou de jeu de données.

**Chaîne de traitement :** Image → Contrôle qualité → Candidats → Classification / localisation → Comptage → Revue humaine

## Phase actuelle : Phase 0, vision classique et mathématiques

- Période : 26 septembre 2026 – début octobre 2026
- Livrable : **V0 · Blood Smear Quality Analyzer**
- Hors périmètre : CNN, Transfer Learning, ML supervisé, application finale

### Ce que contient la V0

| Brique | Rôle | Code |
|---|---|---|
| Contrôle qualité | Netteté, luminosité, contraste, saturation, couleur, et alertes (flou, trop sombre, surexposé…) | `src/quality/` |
| Régions candidates | Taches sombres plausibles (black-hat, seuillage, morphologie, composantes connexes) avec aire, périmètre, circularité | `src/candidates/` |
| Démo | Application Streamlit qui enchaîne les deux briques | `app/streamlit_app.py` |

## Démo Streamlit

```bash
streamlit run app/streamlit_app.py
```

L'application permet de choisir un échantillon du dépôt (`data/samples/`) ou d'importer une image JPEG/PNG, puis :

1. d'afficher les métriques de qualité et les alertes ;
2. de détecter les régions candidates, de régler la détection, et de les comparer aux annotations du jeu de données ;
3. de télécharger les candidats (CSV) et l'image annotée (PNG).

<!-- Quand la capture sera prête : ![Aperçu de la démo](docs/assets/demo.png) -->

## Utiliser le code

```python
from src.quality.analyzer import analyze_image
from src.candidates.extract import extract_candidates

quality = analyze_image("data/samples/plasmodium-phone-0001.jpg")
print(quality["flags"])            # blurry, low_contrast, too_dark, too_bright, overexposed, dark_borders

# extract_candidates attend un tableau RGB uint8 (H, W, 3) : cv2.imread donne du BGR, il faut convertir
candidates = extract_candidates(image_rgb)
print(candidates[0])               # {"id", "bbox": (x, y, w, h), "area", "perimeter", "circularity"}
```

Les formats d'échange entre modules sont décrits dans [docs/conventions.md](docs/conventions.md).

## Installation

```bash
git clone https://github.com/ATTIOU19/PaluScope.git
cd PaluScope
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

### Tests

```bash
python -m pytest
```

## Données

### Dataset de départ : Microscopy Malaria Dataset, split `plasmodium-phonecamera`

- Auteurs : J. Quinn, R. Nakasi, P. K. B. Mugagga, P. Byanyima, W. Lubega, A. Andama
  (Makerere University AI Lab, Ouganda), 2016
- Page officielle : https://air.ug/microscopy/
- Téléchargement : http://air.ug/downloads/plasmodium-phonecamera.zip
- Licence : CC0 1.0
- Contenu : frottis sanguins épais (coloration de Field), ×1000, photographiés au smartphone via un adaptateur de
  microscope ; 1 182 images (750 × 750 px) avec annotations Pascal VOC. Les auteurs annoncent 7 245 plasmodiums
  annotés ; les fichiers XML contiennent 7 628 boîtes (d'environ 40 × 40 px).
- Citation : Quinn et al., Microscopy Malaria Dataset, 2016, https://air.ug/microscopy/

### Organisation

- `data/raw/` : dataset brut, **non versionné**. Dézipper ici le fichier téléchargé.
- `data/processed/` : données transformées, **non versionnées**.
- `data/samples/` : petit jeu d'images d'exemple (versionné), utilisé par les tests et la démo.

Les gros fichiers ne sont **pas** poussés sur GitHub. Voir `data/README.md` pour où les télécharger et où les placer.

## Structure

```
app/         démo Streamlit de la Phase 0 (PaluScope Studio viendra dans une phase ultérieure)
data/        raw/ (non versionné) · processed/ · samples/ (petit jeu de test)
docs/        journal des décisions, conventions, visuels (bannière)
notebooks/   phase_0/ : exploration, contrôle qualité, détection de candidats
reports/     phase_0/ : notes et rapport
src/         quality/ · preprocessing/ · candidates/ : code réutilisable
tests/       tests unitaires et test de fumée de l'application
```

## Équipe

| Membre | Responsabilité principale |
|---|---|
| Narcisse ATTIOU | Coordination, exploration des données, intégration V0 |
| Juste AGBO | Quality Analyzer (`src/quality/`) |
| Aichatou TRAORE | Candidate Detection (`src/preprocessing/`, `src/candidates/`) |

Programme : Académie des Mathématiques Appliquées (AMA), PIIA Cohorte 2, spécial Machine Learning.

## Règles de travail

Voir [CONTRIBUTING.md](CONTRIBUTING.md) et [docs/decisions.md](docs/decisions.md).