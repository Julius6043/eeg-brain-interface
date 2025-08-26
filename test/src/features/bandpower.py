"""Bandpower / PSD Feature-Berechnung."""

from __future__ import annotations
import numpy as np
from scipy.signal import welch
from mne import Epochs
from typing import Dict, Tuple


def compute_psd_welch(
    epochs: Epochs, nperseg: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Berechnet PSD für jede Epoche & Kanal (Welch-Methode).

    Verwendet explizite Schleifen für klare Form (kleine Datensätze -> ok).
    """
    data = epochs.get_data()  # (epochs, channels, times)
    fs = epochs.info["sfreq"]
    n_ep, n_ch, _ = data.shape
    psd_list = []
    freqs = None
    for e in range(n_ep):
        ch_psd = []
        for c in range(n_ch):
            f, p = welch(data[e, c], fs=fs, nperseg=min(nperseg, data.shape[-1]))
            ch_psd.append(p)
            if freqs is None:
                freqs = f
        psd_list.append(ch_psd)
    psd = np.array(psd_list)  # (n_epochs, n_channels, n_freqs)
    return psd, freqs


def bandpower(
    psd: np.ndarray, freqs: np.ndarray, bands: Dict[str, Tuple[float, float]]
) -> dict[str, np.ndarray]:
    """Aggregiert PSD in definierte Frequenzbänder.

    Returns dict mit Schlüssel = Bandname, Wert = Array (n_epochs, n_channels).
    """
    out: dict[str, np.ndarray] = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        out[name] = psd[:, :, mask].mean(axis=-1)
    return out
