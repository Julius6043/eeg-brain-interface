"""Preprocessing utilities for the EEG workload decoding pipeline.

This module contains functions to load raw EEG data from file, apply
standard preprocessing steps (notch and band‑pass filtering, re‑
referencing, resampling) and perform windowing. It also provides a
convenient interface to parse marker/event files that accompany the
recordings. The implementation builds upon the MNE‑Python library to
ensure reproducibility and compatibility with existing neuroimaging
tools. The default parameters follow recommendations from the
literature: a 50 Hz notch filter to remove mains interference and a
1–40 Hz band‑pass to isolate the theta, alpha and beta bands【102020619966441†L182-L209】.

The helper functions in this module do not perform any model
training; they simply prepare the data so that feature extraction
modules can operate on clean, segmented signals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Sequence, Optional, Any

import numpy as np

try:
    import mne  # type: ignore
except ImportError:
    mne = None  # type: ignore

from .config import PreprocessingConfig


def load_raw_file(raw_path: str, preload: bool = True):
    """Load a raw EEG file.

    Parameters
    ----------
    raw_path : str
        Path to the raw file (e.g. FIF or XDF). Currently only FIF is
        supported. Users may extend this function to support other
        formats by leveraging MNE's readers (e.g. ``read_raw_xdf``).
    preload : bool
        Whether to preload the data into memory. Preloading makes
        subsequent filtering steps faster but increases memory usage.

    Returns
    -------
    mne.io.BaseRaw
        The loaded raw object.
    """
    if mne is None:
        raise ImportError(
            "MNE-Python is required to load raw files. Please install MNE before running this function."
        )
    if raw_path.endswith(".fif") or raw_path.endswith(".fif.gz"):
        raw = mne.io.read_raw_fif(raw_path, preload=preload)
    elif raw_path.endswith(".xdf"):
        # XDF reading support requires mne.io.read_raw_xdf (>=0.21)
        if not hasattr(mne.io, "read_raw_xdf"):
            raise NotImplementedError(
                "Reading XDF files requires mne>=0.21 with the read_raw_xdf function."
            )
        raw = mne.io.read_raw_xdf(raw_path, preload=preload)
    else:
        raise ValueError(f"Unsupported file format: {raw_path}")
    return raw


def load_markers(markers_path: str) -> List[Dict[str, float]]:
    """Load a JSON file containing event markers.

    The markers file is expected to be a list of dictionaries with at
    least the keys ``'time_stamp'``, ``'value'`` and ``'onset_s'``.
    Additional metadata such as ``'block'`` or ``'trial'`` is
    propagated unchanged. See the example markers file provided with
    this repository for the exact structure.

    Parameters
    ----------
    markers_path : str
        Path to the JSON file.

    Returns
    -------
    list of dict
        List of marker dictionaries.
    """
    with open(markers_path, "r") as f:
        markers = json.load(f)
    return markers


def preprocess_raw(raw, config: PreprocessingConfig):
    """Apply standard preprocessing to a raw EEG recording.

    The following operations are performed in order:

    1. Notch filtering at ``config.notch_freq`` to remove line noise
       (50 Hz in Europe)【102020619966441†L182-L209】.
    2. Band‑pass filtering between ``config.l_freq`` and
       ``config.h_freq`` to suppress slow drifts and high‑frequency
       noise【102020619966441†L182-L209】.
    3. Optional resampling to ``config.resample``.
    4. Re‑referencing using the strategy specified in ``config.reference``.

    The raw object is copied before any modifications to avoid
    altering the original recording.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Input raw data object.
    config : PreprocessingConfig
        Configuration parameters.

    Returns
    -------
    mne.io.BaseRaw
        A new raw object with filters and reference applied.
    """
    if mne is None:
        raise ImportError(
            "MNE-Python is required for preprocessing. Please install MNE before running this function."
        )
    proc = raw.copy().load_data()

    # Apply notch filter for line noise removal
    if config.notch_freq is not None:
        proc.notch_filter(config.notch_freq, verbose="WARNING")

    # Band‑pass filter to isolate task‑relevant frequency bands
    if config.l_freq is not None or config.h_freq is not None:
        proc.filter(l_freq=config.l_freq, h_freq=config.h_freq, verbose="WARNING")

    # Optional resampling to reduce data size
    if config.resample is not None:
        proc.resample(config.resample, npad="auto")

    # Re‑reference
    if isinstance(config.reference, str) and config.reference == "average":
        proc.set_eeg_reference("average", projection=False)
    elif isinstance(config.reference, Sequence):
        # Custom reference: average across specified channels
        proc.set_eeg_reference(ref_channels=list(config.reference), projection=False)
    else:
        # If reference is None or unknown, leave as is
        pass
    return proc


def create_sliding_windows(
    raw,
    markers: List[Dict[str, float]],
    window_length: float,
    window_overlap: float,
    label_map: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Segment the continuous recording into overlapping windows and assign labels.

    This function partitions the continuous raw data into fixed‑length
    windows with a given overlap. Each window is labelled based on the
    marker occurring nearest to its centre. Group identifiers (e.g.
    subject indices) can be supplied as part of the marker metadata
    (field 'subject' in each marker). When no marker falls within a
    window the label is set to ``-1``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed raw object.
    markers : list of dict
        Event markers with at least an ``'onset_s'`` key specifying
        onset time in seconds and a ``'value'`` key specifying the
        label or condition.
    window_length : float
        Window length in seconds.
    window_overlap : float
        Fraction of overlap between consecutive windows (0 ≤ overlap < 1).
    label_map : dict[str,int], optional
        Mapping from marker values (string or numeric) to integer labels.
        If None, unique marker values are enumerated in sorted order.

    Returns
    -------
    X : ndarray, shape (n_windows, n_channels, window_samples)
        Array of windowed data.
    y : ndarray, shape (n_windows,)
        Integer labels for each window; -1 indicates no marker.
    groups : ndarray, shape (n_windows,)
        Group labels (e.g. subject identifiers) extracted from marker
        metadata if present, else zeros.
    window_times : ndarray, shape (n_windows,)
        Start time (in seconds) of each window relative to recording.
    """
    sfreq = raw.info["sfreq"]
    n_channels = len(raw.ch_names)
    data = raw.get_data()  # shape (n_channels, n_samples)
    total_duration = data.shape[1] / sfreq

    step = window_length * (1 - window_overlap)
    if step <= 0:
        raise ValueError("window_overlap must be < 1")
    starts = np.arange(0, total_duration - window_length + 1e-9, step)
    n_windows = len(starts)
    window_samples = int(round(window_length * sfreq))

    # Prepare label mapping
    if label_map is None:
        unique_values = sorted({m["value"] for m in markers})
        label_map = {v: i for i, v in enumerate(unique_values)}

    # Convert markers to arrays for efficient lookup
    marker_times = np.array([m["onset_s"] for m in markers])
    marker_values = [m["value"] for m in markers]
    group_ids = np.array([m.get("subject", 0) for m in markers])

    # Preallocate lists to collect valid windows
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    group_list: List[int] = []
    time_list: List[float] = []

    for start in starts:
        start_sample = int(round(start * sfreq))
        end_sample = start_sample + window_samples
        # Skip window if it would exceed the recording length
        if end_sample > data.shape[1]:
            continue
        X_list.append(data[:, start_sample:end_sample].astype(np.float32))
        # Determine label based on nearest marker to window centre
        centre_time = start + window_length / 2
        if len(marker_times) > 0:
            idx_closest = int(np.argmin(np.abs(marker_times - centre_time)))
            if np.abs(marker_times[idx_closest] - centre_time) <= window_length / 2:
                val = marker_values[idx_closest]
                y_list.append(label_map.get(val, -1))
                group_list.append(group_ids[idx_closest])
            else:
                y_list.append(-1)
                group_list.append(0)
        else:
            y_list.append(-1)
            group_list.append(0)
        time_list.append(start)

    # Convert lists to arrays
    X_arr = (
        np.stack(X_list, axis=0)
        if X_list
        else np.empty((0, n_channels, window_samples), dtype=np.float32)
    )
    y_arr = np.array(y_list, dtype=np.int64)
    groups_arr = np.array(group_list, dtype=np.int64)
    times_arr = np.array(time_list, dtype=float)
    return X_arr, y_arr, groups_arr, times_arr


def zscore_windows(X: np.ndarray) -> np.ndarray:
    """Apply z‑score normalisation across each channel within each window.

    Parameters
    ----------
    X : ndarray, shape (n_windows, n_channels, window_samples)
        Windowed data.

    Returns
    -------
    ndarray
        Z‑scored windows of the same shape as ``X``.
    """
    # Compute mean and std along the temporal axis
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def create_epochs_for_erp(
    raw,
    markers: List[Dict[str, float]],
    tmin: float,
    tmax: float,
    event_id: Optional[Dict[str, int]] = None,
    event_repeated: str = "drop",
) -> Tuple[Any, np.ndarray]:
    """Create MNE epochs around stimulus markers for ERP analysis.

    The markers are converted to an ``events`` array as required by
    MNE. Each marker ``value`` is mapped to an integer code using
    ``event_id``; if ``event_id`` is None, unique values are assigned
    automatically. Epochs are created with the provided ``tmin`` and
    ``tmax`` relative to the marker onset.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed raw object.
    markers : list of dict
        Markers with fields ``'onset_s'`` and ``'value'``.
    tmin, tmax : float
        Start and end time relative to the event in seconds. E.g. -0.2
        and 0.8 for an epoch ranging from −200 ms to 800 ms around
        stimulus onset.
    event_id : dict[str,int], optional
        Mapping from marker values to integer codes. If None, a
        dictionary is generated automatically.

    Parameters
    ----------
    event_repeated : str
        Strategy if events share identical onset sample (passed to
        ``mne.Epochs``). Common choices: 'drop', 'merge', or 'error'.

    Returns
    -------
    epochs : mne.Epochs
        Epoch object containing the time‑locked segments.
    events : ndarray, shape (n_events, 3)
        MNE events array used to construct the epochs.
    """
    if mne is None:
        raise ImportError(
            "MNE-Python is required to create epochs. Please install MNE before running this function."
        )
    # Create events array: (sample index, 0, event_id)
    sfreq = raw.info["sfreq"]
    if event_id is None:
        unique_values = sorted({m["value"] for m in markers})
        event_id = {val: i + 1 for i, val in enumerate(unique_values)}
    events_list: List[Tuple[int, int, int]] = []
    for m in markers:
        onset_sample = int(round(m["onset_s"] * sfreq))
        events_list.append((onset_sample, 0, event_id[m["value"]]))
    events = np.array(events_list, dtype=int)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0),
        preload=True,
        reject_by_annotation=True,
        event_repeated=event_repeated,
    )
    return epochs, events
