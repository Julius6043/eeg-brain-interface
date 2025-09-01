"""Metrik-Utilities (erweiterbar)."""

from __future__ import annotations
from sklearn.metrics import confusion_matrix
import numpy as np


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def class_distribution(y):
    vals, counts = np.unique(y, return_counts=True)
    return dict(zip(vals, counts))
