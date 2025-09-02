#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing-Pipeline für XDF-EEG (MNE + Braindecode).
- Liest .xdf aus einem Ordner
- Wählt EEG-Stream robust aus (MNELAB falls vorhanden, sonst pyxdf-Fallback)
- Setzt Montage & Referenz
- Notch + Bandpass + (optional) PyPREP
- ICA (EOG) für Artefaktkorrektur
- Extrahiert Events/Marker -> MNE Epochs (optional) & Braindecode Windows
- Speichert: *_raw_clean.fif, *_events.tsv, optional *_epo.fif

Quellen:
- MNE XDF Beispiel & IO: https://mne.tools/stable/auto_examples/io/read_xdf.html
- PREP/PyPREP: https://pyprep.readthedocs.io / Frontiers PREP (2015)
- MNE Artefakte/ICA: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
- Braindecode Windows: https://braindecode.org/stable/generated/braindecode.preprocessing.create_windows_from_events.html
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import mne

# Optional imports (graceful fallback)
try:
    from mnelab.io.xdf import read_raw_xdf as mnelab_read_raw_xdf  # type: ignore
except Exception:
    mnelab_read_raw_xdf = None

try:
    import pyxdf  # type: ignore
except Exception:
    pyxdf = None

try:
    from pyprep import PrepPipeline  # type: ignore
except Exception:
    PrepPipeline = None

# Braindecode (only for windows creation)
from braindecode.preprocessing import create_windows_from_events  # type: ignore


def find_xdf_files(in_dir: Path, pattern: str = "*.xdf") -> List[Path]:
    return sorted([p for p in in_dir.rglob(pattern) if p.is_file()])


def _pick_eeg_stream_id_xdf(
    xdf_path: Path, query: Dict[str, str] = {"type": "EEG"}
) -> Optional[int]:
    """Return the first stream_id matching query using pyxdf metadata lookup."""
    if pyxdf is None:
        return None
    streams = pyxdf.resolve_streams(str(xdf_path))
    # Simple matcher: exact match on key/value in stream['info']
    matches = []
    for s in streams:
        info = s.get("info", {})
        if all(
            str(info.get(k, [""])[0]).upper() == v.upper() for k, v in query.items()
        ):
            matches.append(int(info.get("stream_id", [""])[0]))
    if matches:
        return matches[0]
    # Fallback: choose stream with max EEG-looking channels
    best = None
    for s in streams:
        if s.get("info", {}).get("type", [""])[0].upper() in ("EEG", "SIGNAL", "EXG"):
            try:
                chn = int(s["info"]["channel_count"][0])
            except Exception:
                chn = 0
            sid = int(s["info"]["stream_id"][0])
            if best is None or chn > best[0]:
                best = (chn, sid)
    return best[1] if best else None


def read_xdf_to_raw(
    xdf_path: Path, stream_id: Optional[int] = None, preload: bool = True
) -> mne.io.BaseRaw:
    """Prefer MNELAB's read_raw_xdf (annotations & timing), else build Raw via pyxdf."""
    if mnelab_read_raw_xdf is not None:
        kwargs = {}
        if stream_id is not None:
            kwargs["stream_ids"] = [int(stream_id)]
        raw = mnelab_read_raw_xdf(str(xdf_path), **kwargs)
        if preload:
            raw.load_data()
        return raw

    if pyxdf is None:
        raise RuntimeError(
            "Neither MNELAB nor pyxdf available to read XDF. Install 'mnelab' or 'pyxdf'."
        )

    # Manual conversion (EEG stream only)
    streams, header = pyxdf.load_xdf(str(xdf_path))
    if stream_id is None:
        stream_id = _pick_eeg_stream_id_xdf(xdf_path)  # may still be None
    eeg_stream = None
    marker_stream = None
    for s in streams:
        sid = int(s["info"]["stream_id"][0])
        stype = s["info"]["type"][0].upper()
        if sid == stream_id or (
            stream_id is None and stype in ("EEG", "SIGNAL", "EXG")
        ):
            eeg_stream = s
        if stype in ("MARKERS", "STIM", "TRIGGERS"):
            marker_stream = s
    if eeg_stream is None:
        raise RuntimeError(f"No EEG stream found in {xdf_path}")

    data = np.asarray(eeg_stream["time_series"], dtype=float).T
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])

    # channel labels (best effort)
    ch_names = []
    try:
        ch_names = [
            c["label"][0]
            for c in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        ]
    except Exception:
        ch_names = [f"EEG{i+1:02d}" for i in range(data.shape[0])]

    ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    # Add annotations from marker stream (if present)
    if marker_stream is not None:
        ts = np.asarray(marker_stream["time_stamps"])
        vals = [
            str(v[0]) if isinstance(v, (list, tuple, np.ndarray)) else str(v)
            for v in marker_stream["time_series"]
        ]
        # Align absolute stamps to raw time base
        onset = ts - ts[0]  # relative to start
        annotations = mne.Annotations(
            onset=onset, duration=[0.0] * len(onset), description=vals
        )
        raw.set_annotations(annotations, emit_warning=False)

    if preload:
        raw.load_data()
    return raw


def apply_montage_and_reference(
    raw: mne.io.BaseRaw, montage: str = "standard_1020", average_ref: bool = True
):
    # Montage (ignore unknown channels → mobile EEG tolerant)
    montage_obj = mne.channels.make_standard_montage(montage)
    raw.set_montage(montage_obj, match_case=False, on_missing="ignore")
    if average_ref:
        raw.set_eeg_reference("average", projection=False)
    return raw


def run_pyprep_if_available(
    raw: mne.io.BaseRaw, random_state: int = 97
) -> mne.io.BaseRaw:
    """Run PyPREP (robust ref + bads interpolation). Skipped if PyPREP not installed."""
    if PrepPipeline is None:
        logging.info("PyPREP not installed -> skipping PREP stage.")
        return raw
    # PREP expects set_montage; we'll reuse current montage
    montage = raw.get_montage()
    prep_params = dict(ref_chs=raw.info["ch_names"], ransac=True)
    pp = PrepPipeline(raw.copy(), prep_params, montage, random_state=random_state)
    pp.fit()
    logging.info(f"PyPREP bads: {pp.stored_params_.get('bad_channels', [])}")
    return pp.raw  # cleaned Raw from PyPREP


def basic_filters(
    raw: mne.io.BaseRaw,
    notch: Optional[List[float]] = None,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
):
    if notch:
        raw.notch_filter(freqs=np.array(notch), verbose="ERROR")
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
    return raw


def run_ica_eog(
    raw: mne.io.BaseRaw, n_components: float | int = 0.99, random_state: int = 97
) -> mne.io.BaseRaw:
    """ICA for ocular artifacts – auto-detect & exclude EOG-like components (best effort)."""
    from mne.preprocessing import ICA

    ica = ICA(
        n_components=n_components,
        method="fastica",
        random_state=random_state,
        max_iter="auto",
    )
    _raw_hp = raw.copy().load_data()
    ica.fit(_raw_hp)
    eog_inds, eog_scores = [], []
    try:
        eog_inds, eog_scores = ica.find_bads_eog(raw)
    except Exception:
        pass
    if len(eog_inds):
        ica.exclude = eog_inds
        raw = ica.apply(raw)
        logging.info(f"ICA: removed EOG-like components {eog_inds}")
    else:
        logging.info(
            "ICA: no clear EOG-like components auto-detected; keeping full decomposition."
        )
    return raw


def extract_events(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, Dict[str, int]]:
    """Events from MNE annotations (works with XDF Marker stream)."""
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    return events, event_id


def to_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: Dict[str, int],
    tmin: float = -0.2,
    tmax: float = 0.8,
    baseline: Optional[Tuple[float, float]] = (None, 0.0),
    reject_by_annotation: bool = True,
) -> mne.Epochs:
    picks = mne.pick_types(raw.info, eeg=True, eog=True, exclude="bads")
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=picks,
        preload=True,
        reject_by_annotation=reject_by_annotation,
        detrend=1,
        verbose="ERROR",
    )
    return epochs


def to_braindecode_windows(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: Dict[str, int],
    window_size_s: float = 2.0,
    window_stride_s: float = 0.5,
    trial_start_offset_s: float = 0.0,
    trial_stop_offset_s: float = 0.0,
):
    """Create Braindecode WindowsDataset from MNE Raw + events."""
    sfreq = raw.info["sfreq"]
    windows = create_windows_from_events(
        raw=raw,
        trial_start_offset_samples=int(trial_start_offset_s * sfreq),
        trial_stop_offset_samples=int(trial_stop_offset_s * sfreq),
        window_size_samples=int(window_size_s * sfreq),
        window_stride_samples=int(window_stride_s * sfreq),
        drop_last_window=False,
        events=events,
        event_id=event_id,
        preload=True,
    )
    return windows  # BaseConcatDataset (Braindecode)


def save_events_tsv(
    out_dir: Path, stem: str, events: np.ndarray, event_id: Dict[str, int]
) -> None:
    out = out_dir / f"{stem}_events.tsv"
    import pandas as pd

    df = pd.DataFrame(events, columns=["sample", "prev", "event_id"])
    inv = {v: k for k, v in event_id.items()}
    df["event_label"] = df["event_id"].map(inv)
    df.to_csv(out, sep="\t", index=False)


def process_one_file(
    xdf_path: Path,
    out_dir: Path,
    line_freq: int = 50,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    resample_sfreq: Optional[float] = 250.0,
    use_pyprep: bool = False,
    run_ica: bool = True,
    montage: str = "standard_1020",
) -> None:
    logging.info(f"==> {xdf_path.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = xdf_path.stem

    stream_id = _pick_eeg_stream_id_xdf(xdf_path)  # robust EEG stream chooser
    raw = read_xdf_to_raw(xdf_path, stream_id=stream_id, preload=True)

    # Basic channel hygiene
    apply_montage_and_reference(raw, montage=montage, average_ref=True)

    # Early-stage robust cleaning (optional, follows PREP)
    if use_pyprep:
        raw = run_pyprep_if_available(raw)

    # Filters (after PREP or raw)
    harmonics = [
        line_freq * k for k in range(1, int((raw.info["sfreq"] // line_freq) + 1))
    ]
    basic_filters(raw, notch=harmonics, l_freq=l_freq, h_freq=h_freq)

    # Resample for compactness
    if resample_sfreq:
        raw.resample(resample_sfreq)

    # ICA → remove ocular components
    if run_ica:
        raw = run_ica_eog(raw)

    # Persist cleaned Raw
    raw_out = out_dir / f"{stem}_raw_clean.fif"
    raw.save(raw_out, overwrite=True)

    # Export events & (optional) Epochs
    events, event_id = extract_events(raw)
    if len(events):
        save_events_tsv(out_dir, stem, events, event_id)
        # Optional: compact trial epochs around markers (edit tmin/tmax to task)
        # epochs = to_epochs(raw, events, event_id, tmin=0.0, tmax=2.0, baseline=None)
        # epochs.save(out_dir / f"{stem}_epo.fif", overwrite=True)
        # Also return Braindecode Windows if you want them programmatically:
        # windows = to_braindecode_windows(raw, events, event_id, window_size_s=2.0, window_stride_s=0.5)
    else:
        logging.info("Keine Events/Marker gefunden – überspringe Epoch/Window-Export.")


def main():
    ap = argparse.ArgumentParser(description="XDF → MNE/Braindecode Preprocessing")
    ap.add_argument("--in-dir", type=Path, required=True, help="Ordner mit .xdf")
    ap.add_argument("--out-dir", type=Path, required=True, help="Zielordner")
    ap.add_argument(
        "--pattern", type=str, default="*.xdf", help="Dateimuster (rekursiv)"
    )
    ap.add_argument("--line-freq", type=int, default=50, help="Netzbrumm-Frequenz")
    ap.add_argument("--l-freq", type=float, default=1.0, help="Highpass/Untergrenze")
    ap.add_argument("--h-freq", type=float, default=40.0, help="Lowpass/Obergrenze")
    ap.add_argument(
        "--resample", type=float, default=250.0, help="Ziel-Abtastrate (Hz), 0=keine"
    )
    ap.add_argument(
        "--pyprep", action="store_true", help="PyPREP einsetzen (falls installiert)"
    )
    ap.add_argument("--no-ica", action="store_true", help="ICA deaktivieren")
    ap.add_argument("--montage", type=str, default="standard_1020")
    ap.add_argument("--loglevel", type=str, default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s | %(message)s",
    )

    files = find_xdf_files(args.in_dir, args.pattern)
    if not files:
        raise SystemExit(f"Keine Dateien gefunden unter {args.in_dir} ({args.pattern})")

    for f in files:
        try:
            process_one_file(
                f,
                args.out_dir,
                line_freq=args.line_freq,
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                resample_sfreq=args.resample if args.resample > 0 else None,
                use_pyprep=args.pyprep,
                run_ica=not args.no_ica,
                montage=args.montage,
            )
        except Exception as e:
            logging.exception(f"Fehler bei {f.name}: {e}")


if __name__ == "__main__":
    main()
