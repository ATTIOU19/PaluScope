"""Les six modèles de la feuille de route avec des hyperparamètres PAR DÉFAUT raisonnables.

Sert de point de départ commun pour le tableau de comparaison (bloc P2-A). Les réglages
fins (grilles petites) sont du ressort de Juste (régression logistique, KNN, SVM :
src/models/linear_distance.py) et d'Aichatou (arbre, RF, boosting : src/models/trees.py).
Ils pourront remplacer ces défauts en exposant une fonction ``get_models(seed) -> dict``.

``class_weight="balanced"`` quand le modèle le supporte. KNN et le boosting n'en ont pas :
pour eux, l'option de sous-échantillonnage des négatifs est la manière de traiter le déséquilibre.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def default_models(seed: int = 42, class_weight: str | None = "balanced") -> dict:
    cw = class_weight
    # La standardisation est DANS le pipeline : ajustée sur le train de chaque pli (pas de fuite).
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight=cw, max_iter=2000, random_state=seed)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=15, weights="distance")),
        "svm": make_pipeline(StandardScaler(), SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight=cw)),
        "tree": make_pipeline(StandardScaler(), DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, class_weight=cw, random_state=seed)),
        "rf": make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5, class_weight=cw,
            n_jobs=-1, random_state=seed)),
        "gboost": make_pipeline(StandardScaler(), HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.1, max_iter=200, random_state=seed)),
    }
