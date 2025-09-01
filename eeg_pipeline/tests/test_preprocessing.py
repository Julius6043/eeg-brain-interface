"""Tests for preprocessing utilities."""

import numpy as np
import pytest

try:
    import mne
except ImportError:
    mne = None

from eeg_pipeline.src.preprocessing import zscore_windows, create_sliding_windows


def test_zscore_windows_simple():
    """zscore_windows should standardise each channel within a window."""
    # Create dummy windows: 2 windows, 3 channels, 4 samples
    X = np.array(
        [
            [[1, 2, 3, 4], [10, 20, 30, 40], [-1, -2, -3, -4]],
            [[0, 1, 0, 1], [5, 5, 5, 5], [7, 8, 9, 10]],
        ],
        dtype=float,
    )
    Z = zscore_windows(X)
    # Each channel per window should have zero mean and unit std
    assert np.allclose(Z.mean(axis=2), 0, atol=1e-6)
    stds = Z.std(axis=2)
    # Allow either unit std (normal case) or zero std for constant channels
    assert np.all(
        (np.isclose(stds, 1.0, atol=1e-6)) | (np.isclose(stds, 0.0, atol=1e-6))
    )


@pytest.mark.skipif(mne is None, reason="MNE is required for this test")
def test_create_sliding_windows_shapes_and_labels():
    """create_sliding_windows should return correct shapes and labels."""
    sfreq = 100.0
    times = np.arange(0, 10, 1 / sfreq)
    data = np.vstack(
        [
            np.sin(2 * np.pi * 1 * times),
            np.cos(2 * np.pi * 1 * times),
        ]
    )  # 2 channels
    info = mne.create_info(ch_names=["A", "B"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    # Markers at 2s and 6s with values 'low', 'high'
    markers = [
        {"time_stamp": 2, "onset_s": 2.0, "value": "low", "subject": 1},
        {"time_stamp": 6, "onset_s": 6.0, "value": "high", "subject": 1},
    ]
    window_length = 2.0
    overlap = 0.5
    X, y, groups, starts = create_sliding_windows(raw, markers, window_length, overlap)
    # Ensure windows fit
    assert X.shape[1:] == (2, int(window_length * sfreq))
    # There should be windows starting from 0s to 8s inclusive every 1s (due to 50% overlap)
    expected_starts = np.arange(
        0, 10 - window_length + 1e-9, window_length * (1 - overlap)
    )
    # Only windows with full length should be kept
    expected_valid = [s for s in expected_starts if s + window_length <= 10]
    assert np.allclose(starts, expected_valid)
    # Labels: windows around 2s and 6s should get assigned labels
    # Determine which windows include the markers
    label_map = {"low": 0, "high": 1}
    # Recompute labels quickly for reference
    # At each start compute label assignment; but we only test that labels contain both 0,1 and -1 for no marker
    assert set(y.tolist()) >= {0, 1, -1}
