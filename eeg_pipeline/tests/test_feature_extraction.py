"""Tests for feature extraction functions."""

import numpy as np
import pytest

try:
    import mne
except ImportError:
    mne = None

from eeg_pipeline.src.feature_extraction import compute_bandpower_features, compute_p300_features


@pytest.mark.skipif(mne is None, reason="MNE is required for PSD computations")
def test_compute_bandpower_features_shape():
    """compute_bandpower_features returns array of expected shape."""
    sfreq = 100.0
    # two windows, two channels, 200 samples
    windows = np.random.randn(2, 2, 200)
    bands = {
        'theta': (4.0, 7.0),
        'alpha': (8.0, 12.0),
        'beta': (13.0, 30.0),
    }
    features = compute_bandpower_features(windows, sfreq, bands)
    # Expect shape (n_windows, n_channels * n_bands) = (2, 2*3)
    assert features.shape == (2, 2 * len(bands))


@pytest.mark.skipif(mne is None, reason="MNE is required for ERP computations")
def test_compute_p300_features_detects_peak():
    """compute_p300_features should identify the peak amplitude and latency."""
    # Create synthetic epochs: 5 epochs, 1 channel, 1 s duration, sfreq=100 Hz
    sfreq = 100.0
    times = np.arange(-0.2, 0.8, 1/sfreq)  # -200 ms to 800 ms
    n_times = len(times)
    # We'll set a P300-like positive peak at 300 ms (0.3 s) with amplitude 5
    data = np.zeros((5, 1, n_times))
    peak_time = 0.3
    peak_idx = np.argmin(np.abs(times - peak_time))
    data[:, 0, peak_idx] = 5.0
    info = mne.create_info(['Fz'], sfreq=sfreq, ch_types=['eeg'])
    epochs = mne.EpochsArray(data, info, tmin=-0.2, verbose=False)
    # Compute P300 features with window 0.25–0.35 s
    features = compute_p300_features(epochs, p300_window=(0.25, 0.35), picks=['Fz'])
    # Peak amplitude should be 5 for each epoch
    peak_amps = features[:, 0]
    assert np.allclose(peak_amps, 5.0, atol=1e-6)
    # Peak latency should be close to 0.3 s
    peak_latencies = features[:, 1]
    assert np.allclose(peak_latencies, peak_time, atol=1/sfreq)