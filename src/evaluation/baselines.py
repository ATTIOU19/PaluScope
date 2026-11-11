"""Baselines de référence (feuille de route, section 5.2) : savoir ce que les modèles apportent."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score


class SingleFeatureRule(BaseEstimator, ClassifierMixin):
    """Règle à un seuil sur UNE feature (ex. circularité ou contraste local).

    ``fit`` choisit seulement le sens (la feature, ou son opposé, est le score) selon
    l'AUC sur le train. Le seuil se choisit ensuite sur la validation, comme pour les modèles.
    """

    def __init__(self, feature_index: int = 0):
        self.feature_index = feature_index

    def fit(self, X, y):
        x = np.asarray(X)[:, self.feature_index]
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        self.sign_ = 1.0
        if len(np.unique(y)) == 2 and roc_auc_score(y, x) < 0.5:
            self.sign_ = -1.0
        return self

    def decision_function(self, X):
        return self.sign_ * np.asarray(X)[:, self.feature_index]


def make_dummy() -> DummyClassifier:
    """Classifieur trivial : ignore les features (ROC-AUC = 0,5 ; PR-AUC = prévalence)."""
    return DummyClassifier(strategy="prior")
