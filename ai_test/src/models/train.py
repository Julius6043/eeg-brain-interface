"""High-Level Trainings-Helper für klassische & DL Modelle."""

from __future__ import annotations
import numpy as np
from features import bandpower, utils as feat_utils, csp as csp_utils
from models import classical, deep_learning


def train_classical(epochs, use_csp: bool = True, use_bandpower: bool = False):
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y = epochs.events[:, 2]
    results = {}
    if use_csp:
        mean_acc, std_acc = csp_utils.evaluate_csp_pipeline(X, y)
        results["csp+lda_cv_mean"] = mean_acc
        results["csp+lda_cv_std"] = std_acc
    if use_bandpower:
        psd, freqs = bandpower.compute_psd_welch(epochs)
        bands = {"alpha": (8, 12), "beta": (13, 30)}
        bp = bandpower.bandpower(psd, freqs, bands)
        X_bp, _ = feat_utils.stack_band_features(bp)
        acc = classical.train_eval_simple(X_bp, y)
        results["bandpower_logreg_train_acc"] = acc
    return results


def train_deep(epochs, dl_epochs: int = 3):
    X = epochs.get_data()  # (N, C, T)
    y = epochs.events[:, 2]
    res = deep_learning.train_deep_model(X, y, epochs=dl_epochs)
    return res
