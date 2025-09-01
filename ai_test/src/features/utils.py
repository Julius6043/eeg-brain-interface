"""Allgemeine Feature Utilities."""

from __future__ import annotations
import numpy as np


def stack_band_features(
    band_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Konkateniert Bandpower-Features zu einem 2D-Array (Samples × Features)."""
    names = sorted(band_dict.keys())
    mats = [band_dict[n] for n in names]
    # Jede Matrix (epochs, channels) -> flatten Kanäle pro Band
    flat = [m for m in mats]
    X = np.concatenate([f.reshape(f.shape[0], -1) for f in flat], axis=1)
    feature_names = []
    for n, m in zip(names, mats):
        ch_features = [f"{n}_ch{idx}" for idx in range(m.shape[1])]
        feature_names.extend(ch_features)
    return X, feature_names
