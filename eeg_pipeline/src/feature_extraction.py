"""Feature extraction routines for EEG workload decoding.

This module implements computation of spectral bandpower features from
sliding windows and extraction of event‑related potential (ERP)
features such as the P300. Spectral features capture oscillatory
activity in the theta, alpha and beta bands which are known to
correlate with mental workload【634168958309391†L324-L340】. ERP features
provide complementary information on transient responses to discrete
stimuli, with the P300 amplitude and latency serving as proxies for
cognitive processing load【990978214535399†L150-L155】.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Sequence, List

import numpy as np

try:
    import mne  # type: ignore
except ImportError:
    mne = None  # type: ignore

from .config import FeatureConfig


def compute_bandpower_features(
    windows: np.ndarray,
    sfreq: float,
    bands: Dict[str, Tuple[float, float]]
) -> np.ndarray:
    """Compute bandpower features from time‑domain windows.

    For each window and channel, this function estimates the power
    spectral density (PSD) using Welch's method (via
    ``mne.time_frequency.psd_array_welch``) and integrates the PSD
    across the frequency ranges defined in ``bands``. The resulting
    feature vector is of length ``n_channels * n_bands``.

    Parameters
    ----------
    windows : ndarray, shape (n_windows, n_channels, n_samples)
        Time‑domain windows.
    sfreq : float
        Sampling frequency in Hz.
    bands : dict
        Mapping of band names to (low, high) frequency limits in Hz.

    Returns
    -------
    ndarray, shape (n_windows, n_channels * n_bands)
        Array of bandpower features per window.
    """
    if mne is None:
        raise ImportError(
            "MNE-Python is required for PSD computation. Please install MNE before running this function."
        )
    n_windows, n_channels, n_samples = windows.shape
    n_bands = len(bands)
    features = np.zeros((n_windows, n_channels * n_bands), dtype=np.float32)

    # Compute PSD for each window separately to avoid memory blowup
    for i in range(n_windows):
        data = windows[i]
        # mne computes PSD per channel; returns shape (n_channels, n_freqs)
        psd, freqs = mne.time_frequency.psd_array_welch(
            data,
            sfreq=sfreq,
            n_fft=min(256, n_samples),
            n_per_seg=min(128, n_samples),
            n_overlap=0,
            average='mean',
        )
        # Integrate power in each band
        feat_vec = []
        for low, high in bands.values():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = psd[:, idx].mean(axis=1)  # mean over freq bins
            feat_vec.append(band_power)
        # Concatenate features for all bands and channels
        features[i] = np.concatenate(feat_vec, axis=0)

    return features


def compute_p300_features(
    epochs: 'mne.Epochs',
    p300_window: Tuple[float, float],
    picks: Optional[Sequence[str]] = None
) -> np.ndarray:
    """Compute P300 ERP features from epochs.

    For each epoch a few descriptive statistics of the ERP are
    calculated: the peak amplitude (maximum value) within the P300
    window, the latency of this peak relative to stimulus onset, and
    the mean amplitude in the window. The ERP is averaged across the
    selected channels before feature computation. These features
    quantify the magnitude and timing of the P300 component【990978214535399†L150-L155】.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs extracted around stimulus onsets.
    p300_window : tuple[float, float]
        Start and end (in seconds) of the window used to search for
        the P300 peak. Typical values are (0.25, 0.45) seconds after
        stimulus onset.
    picks : list of str, optional
        Channel names to include. If None, all EEG channels are used.

    Returns
    -------
    ndarray, shape (n_epochs, 3)
        Array containing [peak amplitude, peak latency, mean
        amplitude] for each epoch.
    """
    if mne is None:
        raise ImportError(
            "MNE-Python is required to compute P300 features. Please install MNE before running this function."
        )
    data = epochs.get_data(picks=picks)  # shape (n_epochs, n_channels, n_times)
    times = epochs.times  # time axis in seconds
    # Determine sample indices corresponding to p300 window
    start_idx = np.searchsorted(times, p300_window[0])
    end_idx = np.searchsorted(times, p300_window[1])
    n_epochs = data.shape[0]
    features = np.zeros((n_epochs, 3), dtype=np.float32)
    for i in range(n_epochs):
        # average across selected channels
        avg_wave = data[i].mean(axis=0)
        segment = avg_wave[start_idx:end_idx]
        # Peak amplitude and latency
        peak_idx = np.argmax(segment)
        peak_amp = segment[peak_idx]
        peak_latency = times[start_idx + peak_idx]
        mean_amp = segment.mean()
        features[i] = np.array([peak_amp, peak_latency, mean_amp], dtype=np.float32)
    return features


def fuse_features(
    band_features: np.ndarray,
    erp_features: np.ndarray,
    align_indices: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Concatenate spectral and ERP features to form a joint feature vector.

    When both spectral and ERP features are available, they can be
    fused by concatenation. This function optionally aligns ERP
    features to spectral windows via the ``align_indices`` argument,
    which maps each epoch to the corresponding window index. If no
    alignment is provided, the two feature arrays must have the same
    number of samples.

    Parameters
    ----------
    band_features : ndarray, shape (n_samples_band, n_features_band)
        Spectral features extracted from sliding windows.
    erp_features : ndarray, shape (n_samples_erp, n_features_erp)
        ERP features extracted from epochs.
    align_indices : sequence of int, optional
        For each ERP feature vector specify the index of the band
        feature vector that belongs to the same trial. Length must
        equal ``erp_features.shape[0]``.

    Returns
    -------
    ndarray
        Fused feature matrix of shape (n_samples, n_features_band + n_features_erp).
    """
    if align_indices is None:
        if band_features.shape[0] != erp_features.shape[0]:
            raise ValueError(
                "If align_indices is None, band_features and erp_features must have the same number of rows."
            )
        return np.hstack([band_features, erp_features])
    else:
        fused = []
        for i, idx in enumerate(align_indices):
            fused.append(np.concatenate([band_features[idx], erp_features[i]]))
        return np.array(fused)