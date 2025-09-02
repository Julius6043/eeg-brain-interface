from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import mne
import numpy as np
import pyxdf
import pandas as pd


@dataclass
class DataLoadingConfig:
    channels_keep: Optional[List[str]] = None
    montage: str = "standard_1020"
    auto_scale_to_volts: bool = True
    max_channels: int = 8


@dataclass
class SessionData:
    participant_name: str
    indoor_session: Optional[mne.io.Raw] = None
    indoor_markers: Optional[pd.DataFrame] = None
    outdoor_session: Optional[mne.io.Raw] = None
    outdoor_markers: Optional[pd.DataFrame] = None


def load_xdf_safe(path: Path) -> Tuple[Optional[list], Optional[dict]]:
    """Sicheres Laden von XDF-Dateien mit Fallbacks."""
    try:
        streams, header = pyxdf.load_xdf(str(path))
        return streams, header
    except Exception as e:
        print(f"[WARN] XDF-Load fehlgeschlagen für {path}: {e}")
        return None, None


def _safe_get(d: dict, key: str, default):
    """Sicherer Zugriff auf Dictionary-Werte."""
    try:
        v = d.get(key, default)
        if isinstance(v, (list, tuple)) and len(v) == 1:
            return v[0]
        return v
    except Exception:
        return default


def pick_streams(streams: list) -> Tuple[Optional[dict], Optional[dict]]:
    """Wähle EEG- und Marker-Streams aus."""
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
    """Konvertiere XDF EEG zu MNE RawArray."""
    info = eeg_stream["info"]
    fs = float(_safe_get(info, "nominal_srate", "0"))
    data = np.array(eeg_stream["time_series"], dtype=float).T

    # Auto-Skalierung wenn Daten wie Mikrovolt aussehen
    if config.auto_scale_to_volts:
        med_abs = float(np.nanmedian(np.abs(data)))
        if med_abs > 1e-3:
            print(f"[INFO] Skaliere von µV zu V (median={med_abs:.1f})")
            data *= 1e-6

    # Channel-Namen
    ch_names = [f"EEG{i + 1}" for i in range(data.shape[0])]

    # Erstelle Raw-Objekt
    raw = mne.io.RawArray(data, mne.create_info(ch_names, fs, ch_types="eeg"))

    # Wähle Kanäle
    if config.channels_keep:
        keep = [ch for ch in config.channels_keep if ch in raw.ch_names]
    else:
        keep = raw.ch_names[:config.max_channels]

    raw.pick_channels(keep)

    # Setze Montage
    if config.montage:
        try:
            raw.set_montage(config.montage, on_missing="ignore")
        except Exception as e:
            print(f"[WARN] Montage fehlgeschlagen: {e}")

    return raw


def get_session_paths(experiment_sessions: List[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    sess01_path = None
    sess02_path = None

    for session in experiment_sessions:
        if session.parent.parent.name == "ses-S001":
            sess01_path = session
        elif session.parent.parent.name == "ses-S002":
            sess02_path = session
        else:
            print(f"Unbekannter Pfad: {session}")

    return sess01_path, sess02_path


def load_session_data(session_path: Optional[Path], config: DataLoadingConfig) -> Tuple[
    Optional[mne.io.Raw], Optional[pd.DataFrame]]:
    if session_path is None:
        return None, None

    print(f"Load Session Data: {session_path}")

    streams, header = load_xdf_safe(session_path)
    if not streams:
        return None, None

    # Extrahiere Streams
    eeg_stream, marker_stream = pick_streams(streams)
    if eeg_stream is None:
        print("[WARN] Kein EEG-Stream gefunden")
        return None, None

    # Konvertiere zu Raw
    raw = eeg_stream_to_raw(eeg_stream, config)

    # Lade Marker CSV
    marker_csv = list(session_path.parent.parent.glob("*.csv"))
    markers = pd.read_csv(marker_csv[0]) if marker_csv else None

    return raw, markers


def load_single_session(experiment_dir: Path, config: DataLoadingConfig = None) -> SessionData:
    if config is None:
        config = DataLoadingConfig()

    participant_name = experiment_dir.name.split('_')[-1]
    experiment_sessions = list(experiment_dir.rglob("*.xdf"))
    assert (len(experiment_sessions) <= 2)

    indoor_path, outdoor_path = get_session_paths(experiment_sessions)

    # Lade Sessions
    indoor_session, indoor_markers = load_session_data(indoor_path, config)
    outdoor_session, outdoor_markers = load_session_data(outdoor_path, config)

    return SessionData(
        participant_name=participant_name,
        indoor_session=indoor_session,
        indoor_markers=indoor_markers,
        outdoor_session=outdoor_session,
        outdoor_markers=outdoor_markers
    )


def load_all_sessions(data_dir: Path, config: DataLoadingConfig = None) -> List[SessionData]:
    experiment_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    sessions = []

    for experiment_dir in experiment_dirs:
        try:
            session_data = load_single_session(experiment_dir, config)
            sessions.append(session_data)
            print(f"✓ Session geladen: {session_data.participant_name}")
        except Exception as e:
            print(f"✗ Fehler beim Laden {experiment_dir.name}: {e}")

    return sessions


if __name__ == '__main__':
    # Beispiel-Nutzung
    data_dir = Path("../../data")
    config = DataLoadingConfig(max_channels=8)

    if data_dir.exists():
        sessions = load_all_sessions(data_dir, config)
        print(f"Insgesamt {len(sessions)} Sessions geladen")
    else:
        print("Data-Verzeichnis nicht gefunden")
