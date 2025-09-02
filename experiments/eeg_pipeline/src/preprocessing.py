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
import re
from pathlib import Path
from typing import List, Tuple, Dict, Sequence, Optional, Any, Iterable

import numpy as np

try:
    import mne  # type: ignore
except ImportError:
    mne = None  # type: ignore

from .config import PreprocessingConfig

# Optional heavy dependency used only for raw XDF access when mne.read_raw_xdf
# is unavailable / insufficient.
try:  # pyxdf is tiny, safe to import; if missing we degrade gracefully
    import pyxdf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pyxdf = None  # type: ignore


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


# ---------------------------------------------------------------------------
# Batch XDF ingestion utilities (extension – integrates outer script logic)
# ---------------------------------------------------------------------------
def find_xdf_files(root: str | Path) -> List[Path]:
    """Recursively discover all ``.xdf`` files below *root*.

    Parameters
    ----------
    root : path-like
        Directory under which to search.

    Returns
    -------
    list[Path]
        Sorted list of discovered XDF file paths.
    """
    root_path = Path(root)
    files = sorted(p for p in root_path.rglob("*.xdf") if p.is_file())
    return files


def _load_xdf_safe(xdf_path: Path):  # pragma: no cover (I/O heavy)
    """Robust XDF loader with fallbacks (mirrors notebook helper).

    This uses ``pyxdf`` directly instead of ``mne.io.read_raw_xdf`` to gain
    access to the marker stream separately and apply custom channel name &
    scaling heuristics. If *pyxdf* is missing we raise a clear error.
    """
    if pyxdf is None:
        raise ImportError(
            "pyxdf is required for direct XDF parsing. Install with 'pip install pyxdf'."
        )
    try:
        streams, header = pyxdf.load_xdf(
            str(xdf_path),
            synchronize_clocks=True,
            dejitter_timestamps=True,
            handle_clock_resets=True,
        )
    except Exception as e:  # fallback w/out advanced options
        print(
            f"[WARN] Full XDF load failed ({xdf_path.name}): {e}\n[INFO] Retrying simplified mode …"
        )
        streams, header = pyxdf.load_xdf(
            str(xdf_path),
            synchronize_clocks=False,
            dejitter_timestamps=False,
            handle_clock_resets=False,
        )
    return streams, header


def _safe_get(meta: Dict[str, Any], key: str, default: Any) -> Any:
    try:
        v = meta.get(key, default)
        if isinstance(v, (list, tuple)) and len(v) == 1:
            return v[0]
        return v
    except Exception:
        return default


def _pick_streams(
    streams: Iterable[Dict[str, Any]],
) -> Tuple[
    Dict[str, Any], Optional[Dict[str, Any]]
]:  # pragma: no cover - straightforward
    eeg_stream = None
    marker_stream = None
    for st in streams:
        info = st.get("info", {})
        stype = str(_safe_get(info, "type", "")).lower()
        name = str(_safe_get(info, "name", "")).lower()
        ch_n = int(float(_safe_get(info, "channel_count", "0")))
        if ("eeg" in stype or "unicorn" in name) and ch_n >= 1:
            eeg_stream = st
        if "marker" in stype or "markers" in name:
            marker_stream = st
    if eeg_stream is None:
        raise RuntimeError("No EEG stream found in XDF file.")
    return eeg_stream, marker_stream


def xdf_to_raw_and_markers(
    xdf_path: Path,
    channels_keep: Optional[Sequence[str]] = None,
    montage: str | None = "standard_1020",
) -> Tuple[Any, List[Dict[str, Any]]]:  # raw, markers list
    """Convert an XDF recording to an MNE Raw + structured markers list.

    Heuristics borrowed from the external notebook: auto µV→V scaling if the
    median absolute value suggests microvolt units; optional channel subset; a
    best‑effort extraction of channel names from nested descriptors.
    Marker timing is converted to onset relative to the first EEG timestamp.
    """
    if mne is None:
        raise ImportError("MNE-Python required for building Raw object.")
    streams, _ = _load_xdf_safe(xdf_path)
    eeg_stream, marker_stream = _pick_streams(streams)

    info = eeg_stream["info"]
    fs = float(_safe_get(info, "nominal_srate", 0))
    data = np.asarray(eeg_stream["time_series"], dtype=float).T  # (n_ch, n_times)
    med_abs = float(np.nanmedian(np.abs(data)))
    if med_abs > 1e-3:  # looks like µV
        data *= 1e-6
    # channel names
    ch_names: List[str] = []
    try:
        desc = info.get("desc", {})
        channels = desc.get("channels", {}).get("channel")
        if isinstance(channels, dict):  # single channel case
            channels = [channels]
        if channels:
            for ch in channels:
                label = ch.get("label", "")
                if isinstance(label, list):
                    label = label[0] if label else ""
                ch_names.append(str(label) if label else f"EEG{len(ch_names)+1}")
    except Exception:
        pass
    if not ch_names or len(ch_names) != data.shape[0]:
        ch_names = [f"EEG{i+1}" for i in range(data.shape[0])]
    raw = mne.io.RawArray(data, mne.create_info(ch_names, fs, ch_types="eeg"))
    # restrict channels
    if channels_keep:
        keep = [c for c in channels_keep if c in raw.ch_names]
        if keep:
            raw.pick(keep)
    else:
        # standardize to first 8 if more provided
        if len(raw.ch_names) > 8:
            raw.pick(raw.ch_names[:8])
    if montage:
        try:
            raw.set_montage(montage, on_missing="ignore")
        except Exception:
            pass

    # markers
    markers: List[Dict[str, Any]] = []
    if marker_stream is not None:
        raw_ts0 = float(eeg_stream["time_stamps"][0])
        m_values = marker_stream.get("time_series", [])
        # flatten nested lists
        flat_vals = [v[0] if isinstance(v, (list, tuple)) else v for v in m_values]
        m_stamps = marker_stream.get("time_stamps", [])
        for stamp, val in zip(m_stamps, flat_vals):
            onset = float(stamp) - raw_ts0
            if onset < 0:
                continue
            markers.append({"time_stamp": float(stamp), "value": val, "onset_s": onset})
    return raw, markers


_SUB_RE = re.compile(r"sub-([A-Za-z0-9]+)")
_SES_RE = re.compile(r"ses-([A-Za-z0-9]+)")


def _infer_subject_session(path: Path) -> Tuple[str, str]:
    """Infer subject / session identifiers from the *path* tokens."""
    subj = "unknown"
    ses = "unknown"
    for part in path.parts:
        m1 = _SUB_RE.search(part)
        if m1:
            subj = m1.group(1)
        m2 = _SES_RE.search(part)
        if m2:
            ses = m2.group(1)
    return subj, ses


def batch_preprocess_xdf(
    data_root: str | Path,
    config: PreprocessingConfig,
    out_dir: str | Path,
    channels_keep: Optional[Sequence[str]] = None,
    save_fif: bool = True,
    save_markers: bool = True,
) -> List[Dict[str, Any]]:
    """Process all XDF recordings under *data_root* using *config*.

    For every discovered file we:
      1. Parse XDF into Raw & markers.
      2. Apply standard preprocessing (notch/band‑pass/resample/reference).
      3. Persist optional artefacts (initial + cleaned FIF, markers JSON).

    Parameters
    ----------
    data_root : path-like
        Root directory containing nested subject/session folders.
    config : PreprocessingConfig
        Preprocessing parameters.
    out_dir : path-like
        Base output directory (a subfolder per recording is created).
    channels_keep : sequence[str], optional
        If provided, restrict to these channels (subset present in each file).
    save_fif : bool
        Whether to write raw_initial / raw_clean FIF files.
    save_markers : bool
        Whether to write markers JSON next to processed data.

    Returns
    -------
    list of dict
        Each entry contains keys: 'raw_initial', 'raw_clean', 'markers',
        'subject', 'session', 'source_path', 'window_config' (if windowing is
        later extended – placeholder for future feature integration).
    """
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    xdf_files = find_xdf_files(data_root)
    if not xdf_files:
        print(f"[WARN] No .xdf files found under {data_root}")
        return results
    print(f"[INFO] Found {len(xdf_files)} XDF file(s) under {data_root}")
    for xdf in xdf_files:
        subj, ses = _infer_subject_session(xdf)
        rel_name = xdf.stem
        rec_out = out_base / f"sub-{subj}_ses-{ses}" / rel_name
        rec_out.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Processing {xdf} -> {rec_out}")
        try:
            raw_init, markers = xdf_to_raw_and_markers(xdf, channels_keep)
        except Exception as e:  # pragma: no cover - robust batch
            print(f"[ERROR] Failed to load {xdf.name}: {e}")
            continue
        # Save initial raw & markers (before processing)
        raw_clean = preprocess_raw(raw_init, config)

        # --- Extended metadata & annotation enrichment (inspired by notebook) ---
        # Build marker DataFrame structure (without importing pandas if not installed)
        # We attempt simple pattern extraction for blocks / trials / sequences.
        import math

        try:  # optional pandas usage
            import pandas as _pd  # type: ignore
        except Exception:  # pragma: no cover - optional
            _pd = None  # type: ignore

        block_pattern = re.compile(r"main_block_(\d+)_start")
        trial_pattern = re.compile(r"main_block_(\d+)_trial_(\d+)_on")
        sequence_prefix = "sequence_"
        targets_prefix = "targets_"
        block_info: Dict[int, Dict[str, Any]] = {}
        current_block: Optional[int] = None
        for m in markers:
            val = str(m.get("value", ""))
            onset = float(m.get("onset_s", math.nan))
            mb = block_pattern.match(val)
            if mb:
                current_block = int(mb.group(1))
                block_info.setdefault(current_block, {})["start"] = onset
                continue
            if val.startswith(sequence_prefix) and current_block is not None:
                seq = [
                    s.strip()
                    for s in val[len(sequence_prefix) :].split(",")
                    if s.strip()
                ]
                block_info.setdefault(current_block, {})["seq"] = seq
                continue
            if val.startswith(targets_prefix) and current_block is not None:
                tgt = [
                    t.strip()
                    for t in val[len(targets_prefix) :].split(",")
                    if t.strip().isdigit()
                ]
                block_info.setdefault(current_block, {})["targets"] = [
                    int(t) for t in tgt
                ]
                continue
            mt = trial_pattern.match(val)
            if mt:
                b_id = int(mt.group(1))
                tr_id = int(mt.group(2))
                block_info.setdefault(b_id, {}).setdefault("trials", []).append(
                    {
                        "trial": tr_id,
                        "onset_s": onset,
                    }
                )
        # Close block ends
        rec_dur = raw_init.n_times / raw_init.info["sfreq"]
        sorted_blocks = sorted(block_info.keys())
        for i, b in enumerate(sorted_blocks):
            start = block_info[b].get("start", 0.0)
            end = rec_dur
            if i + 1 < len(sorted_blocks):
                nxt = sorted_blocks[i + 1]
                end = block_info[nxt].get("start", rec_dur)
            block_info[b]["end"] = end
            block_info[b]["dur"] = max(0.0, end - start)

        # Infer n-back difficulty by minimal symmetric difference (1..3)
        def infer_nback(seq: List[str], targets: List[int], max_n=3):
            tgt_set = set(targets or [])
            best = (None, float("inf"), -1)  # (n, diff, overlap)
            best_pred = []
            for n in range(1, max_n + 1):
                pred = []
                for idx in range(n, len(seq)):
                    if seq[idx] == seq[idx - n]:
                        pred.append(idx + 1)  # 1-based index
                pred_set = set(pred)
                diff = len(pred_set ^ tgt_set)
                overlap = len(pred_set & tgt_set)
                if (diff < best[1]) or (diff == best[1] and overlap > best[2]):
                    best = (n, diff, overlap)
                    best_pred = pred
            return best[0], {"diff": best[1], "overlap": best[2], "pred": best_pred}

        for b in sorted_blocks:
            seq = block_info[b].get("seq", [])
            tgs = block_info[b].get("targets", [])
            if seq and tgs:
                nb, info = infer_nback(seq, tgs)
                block_info[b]["nback"] = nb
                block_info[b]["match_info"] = info

        # Create annotations for blocks with inferred n-back
        if mne is not None and sorted_blocks:
            onsets = []
            durs = []
            descs = []
            for b in sorted_blocks:
                nb = block_info[b].get("nback")
                if nb is None:
                    continue
                st = float(block_info[b].get("start", 0.0))
                dur = float(block_info[b].get("dur", 0.0))
                if dur <= 0:
                    continue
                onsets.append(st)
                durs.append(dur)
                descs.append(f"nback_{nb}_block_{b}")
            if onsets:
                anns = mne.Annotations(
                    onset=onsets, duration=durs, description=descs, orig_time=None
                )
                raw_clean.set_annotations(
                    raw_clean.annotations + anns if len(raw_clean.annotations) else anns
                )

        # Optional export of marker tables (CSV/JSON) when pandas present
        if _pd is not None and save_markers and markers:
            df_markers = _pd.DataFrame(markers)
            df_markers.to_csv(rec_out / "markers_all.csv", index=False)
            df_markers.to_json(rec_out / "markers_all.json", orient="records", indent=2)
            # Blocks summary
            if block_info:
                rows = []
                for b in sorted_blocks:
                    rows.append(
                        {
                            "block": b,
                            "start_s": block_info[b].get("start"),
                            "end_s": block_info[b].get("end"),
                            "dur_s": block_info[b].get("dur"),
                            "nback": block_info[b].get("nback"),
                            "n_seq": len(block_info[b].get("seq", [])),
                            "n_targets": len(block_info[b].get("targets", [])),
                            "n_trials": len(block_info[b].get("trials", [])),
                        }
                    )
                _pd.DataFrame(rows).to_csv(rec_out / "blocks_inferred.csv", index=False)
        # --- End enrichment ---
        rec_info: Dict[str, Any] = {
            "raw_initial": raw_init,
            "raw_clean": raw_clean,
            "markers": markers,
            "subject": subj,
            "session": ses,
            "source_path": str(xdf),
        }
        if save_fif:  # pragma: no cover - file I/O
            init_path = rec_out / "raw_initial_eeg.fif"
            clean_path = rec_out / "raw_clean.fif"
            raw_init.save(str(init_path), overwrite=True)
            raw_clean.save(str(clean_path), overwrite=True)
        if save_markers and markers:  # pragma: no cover - file I/O
            with open(rec_out / "markers.json", "w", encoding="utf-8") as f:
                json.dump(markers, f, indent=2)
        results.append(rec_info)
    return results


def _build_arg_parser_batch():  # pragma: no cover - CLI helper
    import argparse

    ap = argparse.ArgumentParser(
        description="Batch preprocessing of all .xdf recordings under a root directory."
    )
    ap.add_argument(
        "--data-root",
        required=True,
        help="Root folder containing .xdf files (recursively)",
    )
    ap.add_argument(
        "--out-dir", required=True, help="Output base directory for processed artefacts"
    )
    ap.add_argument(
        "--notch",
        type=float,
        default=50.0,
        help="Notch frequency (Hz), set 0 to disable",
    )
    ap.add_argument("--l-freq", type=float, default=1.0, help="Lower band-pass (Hz)")
    ap.add_argument("--h-freq", type=float, default=40.0, help="Upper band-pass (Hz)")
    ap.add_argument(
        "--resample",
        type=float,
        default=0.0,
        help="Resample frequency (Hz, 0 = keep native)",
    )
    ap.add_argument(
        "--reference",
        default="average",
        help="Reference mode: 'average' or comma list of channels",
    )
    ap.add_argument(
        "--channels",
        default="",
        help="Comma separated subset of channels to keep (optional)",
    )
    ap.add_argument(
        "--no-save-fif",
        action="store_true",
        help="Do not persist FIF files (in-memory only)",
    )
    ap.add_argument(
        "--no-save-markers", action="store_true", help="Do not write markers JSON files"
    )
    return ap


def main_batch():  # pragma: no cover - CLI entrypoint
    ap = _build_arg_parser_batch()
    args = ap.parse_args()
    reference: str | Sequence[str] | None
    if args.reference.lower() == "none":
        reference = None
    elif "," in args.reference:
        reference = [c.strip() for c in args.reference.split(",") if c.strip()]
    else:
        reference = args.reference
    cfg = PreprocessingConfig(
        notch_freq=None if args.notch in (0, -1) else args.notch,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        resample=None if args.resample in (0, -1) else args.resample,
        reference=reference,
    )
    channels_keep = [c.strip() for c in args.channels.split(",") if c.strip()] or None
    batch_preprocess_xdf(
        data_root=args.data_root,
        config=cfg,
        out_dir=args.out_dir,
        channels_keep=channels_keep,
        save_fif=not args.no_save_fif,
        save_markers=not args.no_save_markers,
    )


if __name__ == "__main__":  # pragma: no cover
    # Provide a lightweight CLI for batch processing independent of the training
    # pipeline. Example:
    #   python -m eeg_pipeline.src.preprocessing --data-root data --out-dir processed
    main_batch()
