#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High‑level preprocessing pipeline for XDF EEG recordings (MNE + optional Braindecode).

Ziele / Features (Schritt für Schritt):
    1. Rekursive Suche nach *.xdf Dateien.
    2. Laden des EEG Streams (bevorzugt MNELABs ``read_raw_xdf`` – enthält Timing/Annotations).
         Fallback: direkter pyxdf → manuelles Bauen eines Raw + Marker‑Annotations.
    3. Kanal‑Montage setzen (Standard 10‑20) & Average Referenz.
    4. (Optional) PyPREP Pipeline für robuste Referenz & Bad‑Channel Interpolation.
    5. Notch Filter (Fundamental + Harmonische) & Bandpass (1–40 Hz default).
    6. (Optional) Resampling zur Reduktion von Datenmenge.
    7. (Optional) ICA zur Entfernung okulärer Artefakte (EOG)
    8. Events aus Annotations extrahieren → TSV exportieren.
    9. (Optional – auskommentiert) Epochs + Braindecode Windows vorbereiten.
 10. Saubere FIF pro Aufnahme speichern.

Design Prinzipien:
    * Robuste Fallbacks (funktioniert auch mit unvollständigen Metadaten).
    * Ausführliches Logging (jede Phase ankündigen & Resultate / Kennzahlen loggen).
    * Kleine, gut testbare Hilfsfunktionen mit klaren Inputs/Outputs.

Hinweise:
    * Die Pipeline verändert Daten *in-place* (Raw wird direkt gefiltert). Für
        Re‑Analysen ggf. vorher ``raw.copy()`` verwenden.
    * ICA & PREP sind optional um Laufzeit zu sparen, falls nur schneller Test.
    * Für sehr lange Aufnahmen kann vor ICA zusätzlich ein High‑Pass (>1 Hz)
        notwendig sein – hier bereits l_freq=1.0 standard.

Referenzen:
    * MNE XDF IO      – https://mne.tools/stable/auto_examples/io/read_xdf.html
    * PREP / PyPREP   – https://pyprep.readthedocs.io
    * MNE ICA         – https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
    * Braindecode Win – https://braindecode.org/stable/generated/braindecode.preprocessing.create_windows_from_events.html
"""
from __future__ import annotations
import argparse
import json
import logging
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import mne

# -----------------------------------------------------------------------------
# Logging Setup (module level logger). Script / CLI initialisiert Level & Format.
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)

# Optional imports (graceful fallback)
try:
    from mnelab.io.xdf import read_raw_xdf as mnelab_read_raw_xdf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mnelab_read_raw_xdf = None

try:
    import pyxdf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pyxdf = None

try:
    from pyprep import PrepPipeline  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PrepPipeline = None

# Braindecode (only for windows creation)
from braindecode.preprocessing import create_windows_from_events  # type: ignore


def find_xdf_files(in_dir: Path, pattern: str = "*.xdf") -> List[Path]:
    """Recursively collect XDF files.

    Parameters
    ----------
    in_dir : Path
        Root directory to search.
    pattern : str
        Glob pattern (default: *.xdf)
    """
    files = sorted([p for p in in_dir.rglob(pattern) if p.is_file()])
    LOGGER.debug(
        "Found %d XDF files (pattern=%s) under %s", len(files), pattern, in_dir
    )
    return files


def list_xdf_streams(xdf_path: Path) -> List[Dict[str, str]]:
    """Return lightweight metadata about streams inside an XDF file.

    Uses pyxdf.resolve_streams if available, else a minimal pyxdf.load_xdf fallback.
    """
    meta: List[Dict[str, str]] = []
    if pyxdf is None:  # pragma: no cover
        LOGGER.warning(
            "pyxdf not installed -> cannot list streams for %s", xdf_path.name
        )
        return meta
    try:
        streams = pyxdf.resolve_streams(str(xdf_path))
    except Exception:
        try:
            streams, _ = pyxdf.load_xdf(
                str(xdf_path), synchronize_clocks=False, dejitter_timestamps=False
            )
        except Exception as e:  # pragma: no cover
            LOGGER.error("Could not read %s for stream listing: %s", xdf_path.name, e)
            return meta
    for s in streams:
        info = s.get("info", {})
        try:
            sid = info.get("stream_id", [""])[0]
            stype = info.get("type", [""])[0]
            name = info.get("name", [""])[0]
            chn = info.get("channel_count", [""])[0]
            rate = info.get("nominal_srate", [""])[0]
        except Exception:
            continue
        meta.append(
            {
                "stream_id": str(sid),
                "type": str(stype),
                "name": str(name),
                "channels": str(chn),
                "srate": str(rate),
            }
        )
    return meta


def _pick_eeg_stream_id_xdf(
    xdf_path: Path,
    preferred_types: Sequence[str] = ("EEG", "EXG", "SIGNAL"),
    min_channels: int = 4,
) -> Optional[int]:
    """Heuristically select an EEG stream id from XDF metadata (no full load).

    Parameters
    ----------
    xdf_path : Path
        File to inspect.
    preferred_types : list[str]
        Ordered list of type labels considered EEG-like (case-insensitive).
    min_channels : int
        Minimal channel threshold for generic numeric fallback.
    """
    if pyxdf is None:  # pragma: no cover
        LOGGER.debug(
            "pyxdf not installed -> cannot preselect stream id for %s", xdf_path.name
        )
        return None
    try:
        streams = pyxdf.resolve_streams(str(xdf_path))
    except Exception as e:  # pragma: no cover
        LOGGER.debug("resolve_streams failed for %s: %s", xdf_path.name, e)
        return None

    def _get(field_list, default=""):
        try:
            return field_list[0]
        except Exception:
            return default

    # 1) Exact preferred type match(s) with max channels
    candidates: List[Tuple[int, int]] = []  # (n_channels, stream_id)
    all_stream_infos = []
    for s in streams:
        info = s.get("info", {})
        stype = str(_get(info.get("type", [""]))).strip().upper()
        try:
            sid = int(_get(info.get("stream_id", [""])) or -1)
            chn = int(_get(info.get("channel_count", ["0"])) or 0)
        except Exception:
            continue
        all_stream_infos.append((sid, stype, chn))
        if stype in [t.upper() for t in preferred_types]:
            candidates.append((chn, sid))
    if candidates:
        chn, sid = max(candidates)
        LOGGER.debug(
            "Selected stream id %d by preferred type (channels=%d) for %s",
            sid,
            chn,
            xdf_path.name,
        )
        return sid

    # 2) Fallback: pick largest numeric stream disregarding type if channels >= min_channels
    generic: Optional[Tuple[int, int]] = None
    for sid, stype, chn in all_stream_infos:
        if chn >= min_channels:
            if generic is None or chn > generic[0]:
                generic = (chn, sid)
    if generic:
        LOGGER.debug(
            "Fallback picked stream id %d (channels=%d) for %s (available types=%s)",
            generic[1],
            generic[0],
            xdf_path.name,
            {t for _sid, t, _chn in all_stream_infos},
        )
        return generic[1]
    LOGGER.debug(
        "No suitable stream id found for %s (streams=%s)",
        xdf_path.name,
        all_stream_infos,
    )
    return None


def read_xdf_to_raw(
    xdf_path: Path,
    stream_id: Optional[int] = None,
    preload: bool = True,
    preferred_types: Sequence[str] = ("EEG", "EXG", "SIGNAL"),
) -> mne.io.BaseRaw:
    """Load XDF into an MNE Raw object.

    Priorität:
      1. **MNELAB** (liefert oft bereits korrekt ausgerichtete Annotationen).
      2. **pyxdf** (manuelle Konversion + Heuristik für Marker).
    """
    if mnelab_read_raw_xdf is not None:
        # Ensure we have a concrete stream id; otherwise reading all streams would
        # require specifying fs_new (resampling) in current MNELAB versions.
        if stream_id is None and pyxdf is not None:
            stream_id = _pick_eeg_stream_id_xdf(
                xdf_path, preferred_types=preferred_types
            )
        if stream_id is not None:
            LOGGER.info(
                "Load XDF via MNELAB: %s (stream_id=%s)", xdf_path.name, stream_id
            )
            stream_ids_list = [int(stream_id)]
            try:
                try:
                    sig = inspect.signature(mnelab_read_raw_xdf)
                    param_names = list(sig.parameters.keys())
                except Exception:  # pragma: no cover
                    param_names = []
                raw = None  # type: ignore
                if param_names and param_names[0] in {"stream_ids", "stream_id"}:
                    raw = mnelab_read_raw_xdf(stream_ids_list, str(xdf_path))  # type: ignore[arg-type]
                else:
                    raw = mnelab_read_raw_xdf(str(xdf_path), stream_ids=stream_ids_list)  # type: ignore[arg-type]
            except TypeError:
                # Last resort: try both orderings explicitly
                try:
                    raw = mnelab_read_raw_xdf(stream_ids_list, str(xdf_path))  # type: ignore[arg-type]
                except Exception:
                    raw = mnelab_read_raw_xdf(str(xdf_path), stream_ids=stream_ids_list)  # type: ignore[arg-type]
            if raw is None:
                raise RuntimeError("MNELAB read_raw_xdf returned None unexpectedly")
            if preload:
                raw.load_data()
            return raw
        else:
            LOGGER.info(
                "MNELAB available but no stream_id resolved -> fallback to pyxdf loader"
            )

    if pyxdf is None:
        raise RuntimeError(
            "Neither MNELAB nor pyxdf available to read XDF. Install 'mnelab' or 'pyxdf'."
        )
    LOGGER.info("Load XDF via pyxdf fallback: %s", xdf_path.name)
    streams, _header = pyxdf.load_xdf(str(xdf_path))
    if stream_id is None:
        stream_id = _pick_eeg_stream_id_xdf(xdf_path, preferred_types=preferred_types)
    eeg_stream = None
    marker_stream = None
    for s in streams:
        try:
            sid = int(s["info"]["stream_id"][0])
            stype = s["info"]["type"][0].upper()
        except Exception:
            continue
        if sid == stream_id or (
            stream_id is None
            and stype.strip().upper() in [t.upper() for t in preferred_types]
        ):
            eeg_stream = s
        if stype in ("MARKERS", "STIM", "TRIGGERS"):
            marker_stream = s
    if eeg_stream is None:
        # Fallback heuristic: choose first multichannel numeric stream when metadata missing.
        for s in streams:
            try:
                ts = np.asarray(s["time_series"], dtype=float)
                if (
                    ts.ndim == 2 and ts.shape[1] >= 4
                ):  # shape: (samples, channels) or vice versa
                    # Heuristic: if second dimension >=4 treat as EEG; pyxdf uses time_series[sample][channels]
                    eeg_stream = s
                    LOGGER.warning(
                        "Fallback selected unnamed stream as EEG (shape=%s) in %s",
                        ts.shape,
                        xdf_path.name,
                    )
                    break
            except Exception:
                continue
        if eeg_stream is None:
            # Diagnostic logging of all stream types to help user configure.
            diag = []
            for s in streams:
                try:
                    sid = s["info"]["stream_id"][0]
                    stype = s["info"]["type"][0]
                    chn = s["info"]["channel_count"][0]
                except Exception:
                    continue
                diag.append(f"(id={sid} type={stype} ch={chn})")
            raise RuntimeError(
                f"No EEG-like stream found in {xdf_path}. Available streams: {' '.join(diag)}. "
                f"Adjust --stream-type or --stream-id."
            )

    data = np.asarray(eeg_stream["time_series"], dtype=float).T
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])
    ch_names: List[str]
    try:
        ch_names = [
            c["label"][0]
            for c in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        ]
    except Exception:
        ch_names = [f"EEG{i+1:02d}" for i in range(data.shape[0])]
    info = mne.create_info(
        ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names)
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    LOGGER.debug(
        "Raw created: channels=%d samples=%d sfreq=%.2f",
        len(ch_names),
        data.shape[1],
        sfreq,
    )

    # Attach marker annotations (0 duration pulses) if available
    if marker_stream is not None:
        ts = np.asarray(marker_stream["time_stamps"], dtype=float)
        vals = [
            str(v[0]) if isinstance(v, (list, tuple, np.ndarray)) else str(v)
            for v in marker_stream["time_series"]
        ]
        if len(ts):
            onset_rel = ts - ts[0]
            annotations = mne.Annotations(
                onset=onset_rel, duration=[0.0] * len(onset_rel), description=vals
            )
            raw.set_annotations(annotations, emit_warning=False)
            LOGGER.debug("Added %d annotations from marker stream", len(vals))

    if preload:
        raw.load_data()
    return raw


def apply_montage_and_reference(
    raw: mne.io.BaseRaw, montage: str = "standard_1020", average_ref: bool = True
) -> mne.io.BaseRaw:
    """Set montage + (optional) average reference.

    Unknown channel names werden ignoriert, damit mobile EEG Systeme robust laufen.
    """
    LOGGER.info("Apply montage=%s average_ref=%s", montage, average_ref)
    montage_obj = mne.channels.make_standard_montage(montage)
    raw.set_montage(montage_obj, match_case=False, on_missing="ignore")
    if average_ref:
        raw.set_eeg_reference("average", projection=False)
    return raw


def run_pyprep_if_available(
    raw: mne.io.BaseRaw, random_state: int = 97
) -> mne.io.BaseRaw:
    """Apply PyPREP robust referencing & bad channel interpolation.

    Returns the cleaned Raw; falls back gracefully when PyPREP missing.
    """
    if PrepPipeline is None:
        LOGGER.info("PyPREP not installed -> skip.")
        return raw
    LOGGER.info("Run PyPREP (random_state=%d)", random_state)
    montage = raw.get_montage()
    prep_params = dict(ref_chs=raw.info["ch_names"], ransac=True)
    pp = PrepPipeline(raw.copy(), prep_params, montage, random_state=random_state)
    pp.fit()
    bads = pp.stored_params_.get("bad_channels", [])
    LOGGER.info("PyPREP bad channels interpolated: %s", bads)
    return pp.raw


def basic_filters(
    raw: mne.io.BaseRaw,
    notch: Optional[List[float]] = None,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
) -> mne.io.BaseRaw:
    """Apply notch (if provided) + bandpass.

    Notch verwendet hier Standard MNE IIR/FIR Defaults; für starke Netzbrumm
    Artefakte ggf. line_freq ±1 Hz als Band‑Stop hinzufügen.
    """
    if notch:
        LOGGER.info("Notch filter freqs=%s", notch)
        raw.notch_filter(freqs=np.array(notch), verbose="ERROR")
    LOGGER.info("Bandpass filter l_freq=%.2f h_freq=%.2f", l_freq, h_freq)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
    return raw


def run_ica_eog(
    raw: mne.io.BaseRaw, n_components: float | int = 0.99, random_state: int = 97
) -> mne.io.BaseRaw:
    """Attempt automatic ocular artifact removal via ICA.

    Komponenten mit hoher Korrelation zu EOG‑typischem Muster werden ausgeschlossen.
    """
    from mne.preprocessing import ICA

    LOGGER.info("Run ICA (n_components=%s)", n_components)
    ica = ICA(
        n_components=n_components,
        method="fastica",
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw.copy().load_data())
    eog_inds: List[int] = []
    try:
        eog_inds, _scores = ica.find_bads_eog(raw)
    except Exception:
        LOGGER.debug("ICA EOG detection failed silently (no EOG channel?)")
    if eog_inds:
        ica.exclude = eog_inds
        raw = ica.apply(raw)
        LOGGER.info("ICA removed components: %s", eog_inds)
    else:
        LOGGER.info("ICA: no EOG components identified")
    return raw


def extract_events(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, Dict[str, int]]:
    """Convert annotations to events array & mapping."""
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    LOGGER.info("Extracted %d events (%d unique labels)", len(events), len(event_id))
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
    """Create epochs around events.

    NOTE: Standard baseline = (tmin, 0) oft sinnvoll; hier (None, 0) optional.
    """
    picks = mne.pick_types(raw.info, eeg=True, eog=True, exclude="bads")
    LOGGER.info("Create epochs tmin=%.3f tmax=%.3f baseline=%s", tmin, tmax, baseline)
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
    """Create Braindecode WindowsDataset (optional deep learning input)."""
    sfreq = raw.info["sfreq"]
    LOGGER.info(
        "Create windows size=%.2fs stride=%.2fs (fs=%.1f) offsets=(%.2f,%.2f)",
        window_size_s,
        window_stride_s,
        sfreq,
        trial_start_offset_s,
        trial_stop_offset_s,
    )
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
    return windows


def save_events_tsv(
    out_dir: Path, stem: str, events: np.ndarray, event_id: Dict[str, int]
) -> None:
    """Persist events as TSV (sample indices + label names)."""
    out = out_dir / f"{stem}_events.tsv"
    import pandas as pd  # local import to keep base import light

    df = pd.DataFrame(events, columns=["sample", "prev", "event_id"])
    inv = {v: k for k, v in event_id.items()}
    df["event_label"] = df["event_id"].map(inv)
    df.to_csv(out, sep="\t", index=False)
    LOGGER.info("Saved events TSV: %s (n=%d)", out.name, len(df))


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
    preferred_types: Sequence[str] = ("EEG", "EXG", "SIGNAL"),
    force_stream_id: Optional[int] = None,
) -> None:
    """Full pipeline for a single XDF file.

    Schritte werden geloggt & können einzeln deaktiviert werden (PyPREP/ICA).
    """
    LOGGER.info("==> START %s", xdf_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = xdf_path.stem

    # 1) EEG Stream bestimmen
    stream_id = (
        force_stream_id
        if force_stream_id is not None
        else _pick_eeg_stream_id_xdf(xdf_path, preferred_types=preferred_types)
    )
    # 2) Laden (+Annotations)
    raw = read_xdf_to_raw(
        xdf_path,
        stream_id=stream_id,
        preload=True,
        preferred_types=preferred_types,
    )
    LOGGER.info(
        "Loaded raw: channels=%d duration=%.1fs sfreq=%.1f",
        len(raw.ch_names),
        raw.n_times / raw.info["sfreq"],
        raw.info["sfreq"],
    )
    # 3) Montage & Referenz
    apply_montage_and_reference(raw, montage=montage, average_ref=True)
    # 4) Optional PyPREP
    if use_pyprep:
        raw = run_pyprep_if_available(raw)
    # 5) Filter (Notch + Bandpass)
    if raw.n_times == 0:
        LOGGER.warning("Empty recording (0 samples) -> skip filtering & ICA")
    else:
        nyq = raw.info["sfreq"] / 2.0
        harmonics: List[float] = []
        k = 1
        while True:
            f = line_freq * k
            if f >= nyq - 0.5:  # keep margin below Nyquist
                break
            harmonics.append(f)
            k += 1
        if not harmonics:
            LOGGER.info(
                "No valid notch frequencies below Nyquist (sfreq=%.1f)",
                raw.info["sfreq"],
            )
        basic_filters(
            raw, notch=harmonics if harmonics else None, l_freq=l_freq, h_freq=h_freq
        )
    # 6) Resample
    if resample_sfreq:
        LOGGER.info("Resample to %.1f Hz", resample_sfreq)
        raw.resample(resample_sfreq)
    # 7) ICA (optional)
    if run_ica:
        raw = run_ica_eog(raw)
    else:
        LOGGER.info("Skip ICA (flag)")
    # 8) Speichern (Clean Raw) – nur wenn nicht leer
    if raw.n_times == 0:
        LOGGER.warning("Recording empty -> skip save & event extraction")
    else:
        raw_out = (
            out_dir / f"{stem}_clean_raw.fif"
        )  # ends with _raw.fif (MNE convention)
        raw.save(raw_out, overwrite=True)
        LOGGER.info("Saved cleaned FIF: %s", raw_out)
        # 9) Events extrahieren & TSV
        events, event_id = extract_events(raw)
        if len(events):
            save_events_tsv(out_dir, stem, events, event_id)
            # (Optional) Epochs / Windows – auskommentiert (Performance sparen)
            # epochs = to_epochs(raw, events, event_id, tmin=0.0, tmax=2.0, baseline=None)
            # epochs.save(out_dir / f"{stem}_epo.fif", overwrite=True)
        else:
            LOGGER.info("No events detected -> skip epoch/window export")
    LOGGER.info("<== DONE  %s", xdf_path.name)


def main():
    """CLI entrypoint.

    Beispiel:
        python -m experiments.preprocessing.preprocessing --in-dir data --out-dir preprocessed
    """
    ap = argparse.ArgumentParser(description="XDF → MNE/Braindecode Preprocessing")
    ap.add_argument("--in-dir", type=Path, required=True, help="Ordner mit .xdf")
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="Zielordner für Outputs"
    )
    ap.add_argument(
        "--pattern", type=str, default="*.xdf", help="Glob Muster (rekursiv)"
    )
    ap.add_argument(
        "--line-freq", type=int, default=50, help="Netzbrumm Grundfrequenz (Hz)"
    )
    ap.add_argument(
        "--l-freq", type=float, default=1.0, help="Highpass / untere Grenzfrequenz"
    )
    ap.add_argument(
        "--h-freq", type=float, default=40.0, help="Lowpass / obere Grenzfrequenz"
    )
    ap.add_argument(
        "--resample",
        type=float,
        default=250.0,
        help="Ziel-Abtastrate (0 = unverändert)",
    )
    ap.add_argument(
        "--pyprep", action="store_true", help="PyPREP aktivieren (wenn installiert)"
    )
    ap.add_argument("--no-ica", action="store_true", help="ICA deaktivieren")
    ap.add_argument("--montage", type=str, default="standard_1020", help="Montage Name")
    ap.add_argument(
        "--loglevel", type=str, default="INFO", help="Logging Level (DEBUG/INFO/…)"
    )
    ap.add_argument(
        "--stream-type",
        type=str,
        default="EEG,EXG,SIGNAL",
        help="Comma separated list of acceptable stream type labels (priority order)",
    )
    ap.add_argument(
        "--stream-id",
        type=int,
        default=None,
        help="Force specific stream id (skips automatic selection)",
    )
    ap.add_argument(
        "--list-streams",
        action="store_true",
        help="Only list discovered stream metadata for each file and exit (no processing)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    files = find_xdf_files(args.in_dir, args.pattern)
    if not files:
        raise SystemExit(
            f"Keine Dateien gefunden unter {args.in_dir} (pattern={args.pattern})"
        )
    LOGGER.info("Processing %d file(s)", len(files))
    preferred_types = tuple(
        t.strip() for t in args.stream_type.split(",") if t.strip()
    ) or ("EEG",)
    for f in files:
        if args.list_streams:
            meta = list_xdf_streams(f)
            if meta:
                for m in meta:
                    LOGGER.info(
                        "STREAM file=%s id=%s type=%s name=%s channels=%s srate=%s",
                        f.name,
                        m["stream_id"],
                        m["type"],
                        m["name"],
                        m["channels"],
                        m["srate"],
                    )
            else:
                LOGGER.warning("No stream metadata found for %s", f.name)
            continue
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
                preferred_types=preferred_types,
                force_stream_id=args.stream_id,
            )
        except Exception as e:  # pragma: no cover - robust CLI
            LOGGER.exception("Failure while processing %s: %s", f.name, e)
    LOGGER.info("All done.")


if __name__ == "__main__":
    main()
