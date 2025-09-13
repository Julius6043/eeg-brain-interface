"""Epoching Module for EEG Pipeline.

This module converts annotated Raw EEG data into epochs based on
the marker annotations. Each epoch corresponds to an experimental block
or a baseline period.

Functionality:
    * Extraction of epochs from Raw data based on annotations
    * Labeling with block names and n-back difficulty levels
    * Creation of MNE Epochs objects with metadata
    * 3D data structure: (Epochs x Channels x Time)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import mne
from mne.io import Raw


@dataclass
class EpochingConfig:
    """Configuration for epoching parameters.
    """

    tmin: float = 0.0
    tmax: Optional[float] = None  # None = use full annotation duration
    baseline: Optional[Tuple[float, float]] = None  # (None, 0) for pre-stimulus baseline
    picks: Optional[List[str]] = None
    reject: Optional[Dict[str, float]] = None


def create_epochs_from_raw(
        raw: Raw, config: EpochingConfig = None
) -> Optional[mne.Epochs]:
    """Creates epochs from an annotated Raw object with 4s segments and 2s overlap.
    """
    if config is None:
        config = EpochingConfig(tmin=0.0, tmax=4.0)

    if not raw.annotations or len(raw.annotations) == 0:
        print("[WARN] No annotations found in Raw object")
        return None

    segment_length = 4.0
    overlap = 2.0

    all_events = []
    event_id = {
        'baseline': 0,
        '0-back': 1,
        '1-back': 2,
        '2-back': 3,
        '3-back': 4
    }

    for block_idx, annot in enumerate(raw.annotations):
        description = annot['description']
        start_time = annot['onset']
        duration = annot['duration']

        block_events = mne.make_fixed_length_events(
            raw,
            id=event_id[description],
            start=start_time,
            stop=start_time + duration,
            duration=segment_length,
            overlap=overlap
        )

        if len(block_events) == 0:
            print(f"      No segments for block {block_idx} (too short)")
            continue

        print(f"      → {len(block_events)} segments of {segment_length}s")
        all_events.append(block_events)

    if not all_events:
        print("[WARN] No valid events created")
        return None

    events = np.vstack(all_events)
    events = events[events[:, 0].argsort()]

    print(f"\n[INFO] Creating {len(events)} epochs...")
    print(f"   - Event IDs: {event_id}")

    try:
        # Create standard MNE Epochs
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=config.tmin,
            tmax=config.tmax or segment_length,
            baseline=config.baseline,
            picks=config.picks,
            reject=config.reject,
            preload=True,
            verbose=False
        )

        data_shape = epochs.get_data().shape
        print(f"[SUCCESS] {len(epochs)} epochs created")
        print(f"   - Shape: {data_shape}")

        assert epochs is not None, "Epochs object is None"
        assert len(epochs) > 0, "No epochs were created"

        assert len(data_shape) == 3, f"Expected 3D data, but got: {len(data_shape)}D"
        assert data_shape[0] > 0, "No epochs in data matrix"
        assert data_shape[1] > 0, "No channels in data matrix"
        assert data_shape[2] > 0, "No time points in data matrix"

        assert hasattr(epochs, 'event_id'), "Event ID dictionary is missing"
        assert len(epochs.event_id) > 0, "Event ID dictionary is empty"
        assert epochs.info['sfreq'] > 0, "Invalid sampling rate"

        onset_times = epochs.events[:, 0] / epochs.info['sfreq']
        assert np.all(onset_times[:-1] <= onset_times[1:]), "Events are not sorted chronologically"

        return epochs

    except Exception as e:
        print(f"[ERROR] Error during epoching: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_epochs_summary(epochs: mne.Epochs) -> pd.DataFrame:
    """Creates a summary of the epoch distribution.
    """
    summary_data = []

    for event_name, event_id in epochs.event_id.items():
        count = sum(epochs.events[:, 2] == event_id)
        summary_data.append({
            'event_name': event_name,
            'event_id': event_id,
            'count': count,
            'percentage': (count / len(epochs)) * 100
        })

    return pd.DataFrame(summary_data)


def epochs_to_dataframe(epochs: mne.Epochs) -> pd.DataFrame:
    """Converts epochs to a DataFrame for pandas/scikit-learn.
    """
    # Extract data and reshape to 2D
    data = epochs.get_data()  # (n_epochs, n_channels, n_timepoints)
    n_epochs, n_channels, n_timepoints = data.shape

    # Reshape to 2D: (n_epochs, n_features)
    data_2d = data.reshape(n_epochs, n_channels * n_timepoints)

    # Create feature names
    feature_names = []
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        for time_idx in range(n_timepoints):
            feature_names.append(f"{ch_name}_t{time_idx}")

    # Create DataFrame
    df = pd.DataFrame(data_2d, columns=feature_names)

    # Add labels (reverse lookup from event_id)
    id_to_name = {v: k for k, v in epochs.event_id.items()}
    df['label'] = [id_to_name[event_id] for event_id in epochs.events[:, 2]]

    return df


def validate_epochs(epochs: mne.Epochs) -> None:
    """Validates the epoching result using asserts.
    """
    # Basic validation
    assert epochs is not None, "Epochs object is None"
    assert len(epochs) > 0, "No epochs were created"
    
    data_shape = epochs.get_data().shape
    assert len(data_shape) == 3, f"Expected 3D data, but got: {len(data_shape)}D"
    assert data_shape[0] > 0, "No epochs in data matrix"
    assert data_shape[1] > 0, "No channels in data matrix"
    assert data_shape[2] > 0, "No time points in data matrix"
    
    assert hasattr(epochs, 'event_id'), "Event ID dictionary is missing"
    assert len(epochs.event_id) > 0, "Event ID dictionary is empty"
    assert epochs.info['sfreq'] > 0, "Invalid sampling rate"
    
    # Temporal consistency
    onset_times = epochs.events[:, 0] / epochs.info['sfreq']
    assert np.all(onset_times[:-1] <= onset_times[1:]), "Events are not sorted chronologically"
    
    # Check distances between events - more flexible validation
    if len(onset_times) > 1:
        time_diffs = np.diff(onset_times)
        
        # No negative time differences
        assert np.all(time_diffs >= 0), "Negative time differences found"
        
        # Check distances within blocks (should be ~2s due to overlap)
        # and between blocks (can be larger)
        small_diffs = time_diffs[time_diffs <= 5.0]  # Within blocks
        large_diffs = time_diffs[time_diffs > 5.0]   # Between blocks
        
        if len(small_diffs) > 0:
            median_small = np.median(small_diffs)
            assert 1.0 <= median_small <= 3.0, f"Unexpected intra-block distances: {median_small:.2f}s (expected: ~2s)"
        
        # Check that large distances are not too extreme (max 5 minutes between blocks)
        if len(large_diffs) > 0:
            max_large = np.max(large_diffs)
            assert max_large <= 300, f"Inter-block distances too large: {max_large:.1f}s (max: 300s)"
            print(f"   Info: {len(large_diffs)} inter-block transitions found (max: {max_large:.1f}s)")
    
    print(f"✓ Epoch validation successful: {len(epochs)} epochs, shape: {data_shape}")