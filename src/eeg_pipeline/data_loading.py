from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import mne
import numpy as np
import pyxdf


@dataclass
class SessionData:
    participant_name: str
    indoor_session: Path
    outdoor_session: Path


def load_xdf_safe(path: Path):
    """Robust XDF loader with fallbacks."""
    try:
        streams, header = pyxdf.load_xdf(
            str(path),
            synchronize_clocks=True,
            dejitter_timestamps=True,
            handle_clock_resets=True,
        )
    except Exception as e:
        print("[WARN] Full load failed:", e)
        streams, header = pyxdf.load_xdf(
            str(path),
            synchronize_clocks=False,
            dejitter_timestamps=False,
            handle_clock_resets=False,
        )
    return streams, header


def _safe_get(d, key, default):
    try:
        v = d.get(key, default)
        if isinstance(v, (list, tuple)) and len(v) == 1:
            return v[0]
        return v
    except Exception:
        return default


def pick_streams(streams) -> Tuple[dict, Optional[dict]]:
    """Pick EEG stream + marker stream."""
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
    if eeg_stream is None:
        raise RuntimeError("No EEG stream found in XDF.")
    return eeg_stream, marker_stream


def eeg_stream_to_raw(
        eeg_stream: dict,
        channels_keep: Optional[List[str]] = None,
        montage: Optional[str] = "standard_1020",
) -> mne.io.Raw:
    """Convert XDF EEG to MNE RawArray (auto-scale µV->V), keep subset."""
    info = eeg_stream["info"]
    fs = float(_safe_get(info, "nominal_srate", "0"))
    data = np.array(eeg_stream["time_series"], dtype=float).T  # (n_ch, n_times)

    # Auto-rescale if data looks like microvolts
    med_abs = float(np.nanmedian(np.abs(data)))
    if med_abs > 1e-3:
        print(f"[INFO] Values look like µV (median={med_abs:.1f}); converting to Volts (×1e-6).")
        data *= 1e-6
    else:
        print(f"[INFO] Values look like Volts (median={med_abs:.3e}); no rescale.")

    # Channel names
    ch_names = []
    try:
        desc = info.get("desc", {})
        clist = desc.get("channels", {}).get("channel")
        if isinstance(clist, dict):
            clist = [clist]
        if clist:
            for ch in clist:
                lab = ch.get("label", "")
                if isinstance(lab, list):
                    lab = lab[0] if lab else ""
                ch_names.append(str(lab) if lab else f"EEG{len(ch_names) + 1}")
    except Exception:
        pass
    if not ch_names or len(ch_names) != data.shape[0]:
        ch_names = [f"EEG{i + 1}" for i in range(data.shape[0])]

    raw = mne.io.RawArray(data, mne.create_info(ch_names, fs, ch_types="eeg"))

    # Keep exactly 8 channels (your 8 or first 8 present)
    if channels_keep:
        keep = [ch for ch in channels_keep if ch in raw.ch_names]
    else:
        keep = raw.ch_names[:8]
    raw.pick_channels(keep)
    print("[INFO] Channels kept:", raw.ch_names)

    # Optional montage
    try:
        if montage:
            raw.set_montage(montage, on_missing="ignore")
    except Exception as e:
        print(f"[WARN] Montage set failed ({montage}):", e)

    return raw


def load_session_data(sessions: dict) -> None:
    for session_type, xdf_file_path in sessions.items():
        print(f"Loading session from {xdf_file_path}")
        streams, header = load_xdf_safe(xdf_file_path)
        eeg_stream, marker_stream = pick_streams(streams)
        raw = eeg_stream_to_raw(
            eeg_stream, channels_keep=None, montage="standard_1020"
        )
        sessions[session_type] = raw


def process_session(session_data: SessionData) -> None:
    print(f"Processing session for participant {session_data.participant_name}")
    # Add your session processing logic here
    pass


def convert_xdf_to_mne():
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data"

    experiments_dir = [p for p in data_dir.iterdir() if p.is_dir()]
    if not experiments_dir:
        print(f"No experiments in folder {data_dir} found.")
        return None

    for experiment_dir in experiments_dir:
        participant_name = experiment_dir.name.split("_")[-1]
        experiment_sessions = list(experiment_dir.rglob("*.xdf"))
        assert (len(experiment_sessions) >= 2)

        sessions = {}
        for session in experiment_sessions:
            if session.parent.parent.name == "ses-S001":
                sessions['indoor'] = session
            elif session.parent.parent.name == "ses-S002":
                sessions['outdoor'] = session
            else:
                print(f"Unbekannter Pfad: {session}")

        load_session_data(sessions)

        session_data = SessionData(participant_name, sessions['indoor'], sessions['outdoor'])

        # Process indoor session
        # TODO
        process_session(session_data)

        # print(f"Converting file: {xdf_file_path}")
        #
        # print(f"[INFO] Loading XDF: {xdf_file_path}")
        # streams, header = load_xdf_safe(xdf_file_path)
        # print(f"[INFO] Loaded {len(streams)} stream(s)")
        #
        # eeg_stream, marker_stream = pick_streams(streams)
        # raw = eeg_stream_to_raw(
        #     eeg_stream, channels_keep=None, montage="standard_1020"
        # )
        #
        # outdir = Path(__file__).parent.parent.parent / "results"
        # raw_init_fif = outdir / f"experiment_{f.name}.fif"
        # raw.save(str(raw_init_fif), overwrite=True)


if __name__ == '__main__':
    raw_eeg = convert_xdf_to_mne()
    if raw_eeg:
        print(raw_eeg)
