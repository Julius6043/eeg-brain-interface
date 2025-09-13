"""Data Loading Utilities for EEG Pipeline.

This module encapsulates the reading of XDF files (EEG + markers) and
provides structured objects for downstream processing.

Design Goals:
    * Robust, fault-tolerant loaders (catch exceptions locally).
    * Clear separation between configuration (`DataLoadingConfig`) and data container (`SessionData`).
    * Minimal heuristics for stream selection (EEG + markers) – can be extended later.

Important Assumptions / Limitations:
    * Currently, only ONE EEG stream and ONE marker stream are selected (if multiple exist,
      the last match overwrites the previous one).
    * The marker file (CSV) is searched for using a generic pattern (first `*.csv` in a parent
      folder) – this might fail if multiple CSVs exist.
    * Channels are generically named EEG1..EEG<N> (original channel names are not
      extracted from the XDF metadata tree).
    * Autoscaling interprets values with median(|x|) > 1e-3 as microvolts and rescales to volts.

Recommended Future Improvements (TODO notes are set in the code):
    - More precise channel name extraction from stream metadata.
    - Selection priority list for competing EEG streams.
    - Validation / logging via a standardized logger interface instead of `print`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import mne
import numpy as np
import pyxdf
import pandas as pd


@dataclass
class DataLoadingConfig:
    """Configuration parameters for loading XDF data.
    """

    channels_keep: Optional[List[str]] = None
    montage: str = "standard_1020"
    auto_scale_to_volts: bool = True
    max_channels: int = 8


@dataclass
class SessionData:
    """Structured collection of a participant's (up to two) sessions.
    """

    participant_name: str
    indoor_session: Optional[mne.io.Raw] = None
    indoor_markers: Optional[pd.DataFrame] = None
    indoor_epochs: Optional[mne.Epochs] = None
    outdoor_session: Optional[mne.io.Raw] = None
    outdoor_markers: Optional[pd.DataFrame] = None
    outdoor_epochs: Optional[mne.Epochs] = None


def load_xdf_safe(path: Path) -> Tuple[Optional[list], Optional[dict]]:
    """Robustly load an XDF file.
    """
    try:
        streams, header = pyxdf.load_xdf(str(path))
        return streams, header
    except Exception as e:  # pragma: no cover - defensive I/O
        print(f"[WARN] XDF load failed for {path}: {e}")
        return None, None


def _safe_get(d: dict, key: str, default):
    """Robust access to (nested) XDF metadata.
    """
    try:
        v = d.get(key, default)
        if isinstance(v, (list, tuple)) and len(v) == 1:
            return v[0]
        return v
    except Exception:  # pragma: no cover - defensive
        return default


def pick_streams(streams: list) -> Tuple[Optional[dict], Optional[dict]]:
    """Heuristically select EEG and marker streams.
    """
    eeg_stream, marker_stream = None, None

    for st in streams:
        info = st.get("info", {})
        stype = str(_safe_get(info, "type", "")).lower()
        sname = str(_safe_get(info, "name", "")).lower()
        ch_n = int(float(_safe_get(info, "channel_count", "0")))

        if ("eeg" in stype or "unicorn" in sname) and ch_n >= 1:
            eeg_stream = st
        if "marker" in stype or "markers" in sname:
            marker_stream = st

    return eeg_stream, marker_stream


def eeg_stream_to_raw(eeg_stream: dict, config: DataLoadingConfig) -> mne.io.Raw:
    """Convert an EEG stream into an `mne.io.Raw` object.
    """
    info = eeg_stream["info"]
    fs = float(_safe_get(info, "nominal_srate", "0"))
    data = np.array(
        eeg_stream["time_series"], dtype=float
    ).T  # shape: (n_channels, n_samples)

    # (Heuristic) Scale to Volts if amplitudes are likely in the µV range.
    if config.auto_scale_to_volts:
        med_abs = float(np.nanmedian(np.abs(data)))
        if med_abs > 1e-3:
            print(f"[INFO] Scaling from µV to V (median={med_abs:.1f})")
            data *= 1e-6

    # Channel mapping
    channel_mapping = {1: "Fz", 2: "C4", 3: "Cz", 4: "C3", 5: "Pz", 6: "PO8", 7: "Oz", 8: "PO7"}
    ch_names = [channel_mapping.get(i + 1, f"EEG{i + 1}") for i in range(data.shape[0])]

    # Create MNE Raw object
    raw = mne.io.RawArray(data, mne.create_info(ch_names, fs, ch_types="eeg"))

    # Channel filtering
    if config.channels_keep:
        keep = [ch for ch in config.channels_keep if ch in raw.ch_names]
    else:
        keep = raw.ch_names[: config.max_channels]
    raw.pick_channels(keep)

    # Apply montage (soft fault tolerance)
    if config.montage:
        try:
            raw.set_montage(config.montage, on_missing="ignore")
        except Exception as e:  # pragma: no cover - GUI/IO dependent
            print(f"[WARN] Setting montage failed: {e}")

    return raw


def get_session_paths(
        experiment_sessions: List[Path],
) -> Tuple[Optional[Path], Optional[Path]]:
    """Assign found XDF files to the expected session codes.
    """
    sess01_path: Optional[Path] = None
    sess02_path: Optional[Path] = None

    for session in experiment_sessions:
        if session.parent.parent.name == "ses-S001":
            sess01_path = session
        elif session.parent.parent.name == "ses-S002":
            sess02_path = session
        else:
            print(f"Unknown path: {session}")

    return sess01_path, sess02_path


def load_session_data(
        session_path: Optional[Path],
        config: DataLoadingConfig,
) -> Tuple[Optional[mne.io.Raw], Optional[pd.DataFrame]]:
    """Load a single session (EEG + marker CSV).
    """
    if session_path is None:
        return None, None

    print(f"Load Session Data: {session_path}")

    streams, _header = load_xdf_safe(session_path)
    if not streams:
        return None, None

    # Stream selection (EEG + markers)
    eeg_stream, marker_stream = pick_streams(streams)
    if eeg_stream is None:
        print("[WARN] No EEG stream found")
        return None, None
    first_eeg_timestamp = (
        eeg_stream["time_stamps"][0] if "time_stamps" in eeg_stream else 0
    )
    raw = eeg_stream_to_raw(eeg_stream, config)

    # convert marker_stream to DataFrame (if present) and adjust timestamps
    markers = None
    if marker_stream and "time_series" in marker_stream:
        marker_data = np.array(marker_stream["time_series"])
        if marker_data.ndim == 1:
            marker_data = marker_data[:, np.newaxis]
        timestamps = np.array(marker_stream["time_stamps"])
        # Adjust marker timestamps relative to EEG start (assumption: EEG starts before the first marker)
        timestamps = timestamps - first_eeg_timestamp
        markers = pd.DataFrame(
            marker_data, columns=[f"Marker{i + 1}" for i in range(marker_data.shape[1])]
        )
        markers.insert(0, "Timestamp", timestamps)
    return raw, markers


def load_single_session(
        experiment_dir: Path, config: DataLoadingConfig = None
) -> SessionData:
    """Load (up to) two sessions for a participant's folder.
    """
    if config is None:
        config = DataLoadingConfig()

    participant_name = experiment_dir.name.split("_")[-1]
    experiment_sessions = list(experiment_dir.rglob("*.xdf"))
    assert (
            len(experiment_sessions) <= 2
    ), "More than two sessions found – adjustment needed"

    indoor_path, outdoor_path = get_session_paths(experiment_sessions)

    indoor_session, indoor_markers = None, None
    try:
        indoor_session, indoor_markers = load_session_data(indoor_path, config)
    except Exception as e:
        print(f"[WARN] Error loading the indoor session for {participant_name}: {e}")

    outdoor_session, outdoor_markers = None, None
    try:
        outdoor_session, outdoor_markers = load_session_data(outdoor_path, config)
    except Exception as e:
        print(f"[WARN] Error loading the outdoor session for {participant_name}: {e}")

    return SessionData(
        participant_name=participant_name,
        indoor_session=indoor_session,
        indoor_markers=indoor_markers,
        outdoor_session=outdoor_session,
        outdoor_markers=outdoor_markers,
    )


def load_all_sessions(
        data_dir: Path, config: DataLoadingConfig = None
) -> List[SessionData]:
    """Load all participant directories within a root folder.

    Errors per participant do not stop the process (best-effort aggregation).
    """
    experiment_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    sessions: List[SessionData] = []

    for experiment_dir in experiment_dirs:
        try:
            session_data = load_single_session(experiment_dir, config)
            sessions.append(session_data)
            print(f"✓ Session loaded: {session_data.participant_name}")
        except Exception as e:  # pragma: no cover
            print(f"✗ Error loading {experiment_dir.name}: {e}")

    return sessions


if __name__ == "__main__":
    # Example usage (when running the module manually)
    data_dir = Path("data")
    config = DataLoadingConfig(max_channels=8)

    if data_dir.exists():
        sessions = load_all_sessions(data_dir, config)
        print(f"A total of {len(sessions)} sessions were loaded")
    else:
        print("Data directory not found")