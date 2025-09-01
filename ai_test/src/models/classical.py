"""Klassische ML Modelle (Bandpower + Logistic Regression, CSP + LDA)."""

from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def build_bandpower_logreg() -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]
    )


def train_eval_simple(X: np.ndarray, y: np.ndarray) -> float:
    model = build_bandpower_logreg()
    model.fit(X, y)
    preds = model.predict(X)
    return accuracy_score(y, preds)
