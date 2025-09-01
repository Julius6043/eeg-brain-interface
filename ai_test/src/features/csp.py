"""Hilfsfunktionen für CSP + scikit-learn Pipelines."""

from __future__ import annotations
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import ShuffleSplit, cross_val_score
import numpy as np
from typing import Tuple
from config import settings


def build_csp_lda_pipeline(n_components: int | None = None) -> Pipeline:
    if n_components is None:
        n_components = settings.CSP_COMPONENTS
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    lda = LDA()
    return Pipeline([("csp", csp), ("lda", lda)])


def evaluate_csp_pipeline(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5, test_size: float = 0.2
) -> Tuple[float, float]:
    cv = ShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=settings.RANDOM_STATE
    )
    pipeline = build_csp_lda_pipeline()
    scores = cross_val_score(pipeline, X, y, cv=cv)
    return scores.mean(), scores.std()
