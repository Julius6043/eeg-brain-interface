
import mne
import numpy as np
from typing import Dict, Any, Tuple

def default_config() -> Dict[str, Any]:
    """Reasonable defaults for ERP-style work + ICA."""
    return {
        # Filtering
        "notch_freqs": [50, 100],      # line noise + 1st harmonic
        "l_freq": 0.1,                 # high-pass for task viewing
        "h_freq": 40.0,                # low-pass (set None if you want full band)
        "ref": "average",              # re-reference strategy ("average" or list of ch names)
        "resample_sfreq": 250,         # downsample for speed (None to skip)

        # ICA
        "ica_l_freq": 1.0,             # high-pass for ICA fitting (recommended >= 1 Hz)
        "ica_method": "fastica",       # "fastica", "picard", or "infomax"
        "ica_n_components": 0.99,      # keep 99% variance (or int for fixed n)
        "random_state": 97,
        "eog_ch_name": None,           # e.g., "EOG001" if present
        "ecg_ch_name": None,           # e.g., "ECG001" if present
        "ica_auto_exclude": True,      # automatically exclude detected ICs
        "apply_ica": True,             # apply ICA to filtered Raw

        # Saving
        "save_base": None,             # e.g., OUT_BASENAME; if set, saves ICA + cleaned Raw
    }

def _ensure_list(x):
    if x is None:
        return None
    return x if isinstance(x, (list, tuple)) else [x]

def preprocess_raw(raw: mne.io.BaseRaw, cfg: Dict[str, Any]) -> Tuple[mne.io.BaseRaw, mne.preprocessing.ICA, Dict[str, Any]]:
    """Filter → notch → re-ref → (resample) → ICA; detect EOG/ECG components; apply & save.

    Returns
    -------
    raw_clean : Raw
        Cleaned Raw (ICA-applied if cfg["apply_ica"] is True).
    ica : ICA
        Fitted ICA object (with `exclude` set according to detections).
    report : dict
        Small dict with diagnostics (excluded indices, scores, cfg used).
    """
    cfg = {**default_config(), **(cfg or {})}

    raw_pre = raw.copy()

    # 1) Notch
    if cfg["notch_freqs"]:
        raw_pre.notch_filter(freqs=cfg["notch_freqs"], picks="eeg")

    # 2) Band-pass for task view
    raw_pre.filter(l_freq=cfg["l_freq"], h_freq=cfg["h_freq"], picks="eeg",
                   phase="zero-double", fir_window="hamming")

    # 3) Reference
    if cfg["ref"] == "average":
        raw_pre.set_eeg_reference("average")
    elif isinstance(cfg["ref"], (list, tuple)):
        raw_pre.set_eeg_reference(cfg["ref"])

    # 4) Optional resample
    if cfg["resample_sfreq"]:
        raw_pre.resample(cfg["resample_sfreq"], npad="auto")

    # 5) Prepare a copy for ICA fitting (>= 1 Hz HP recommended)
    raw_ica_fit = raw_pre.copy()
    if cfg["ica_l_freq"] and cfg["ica_l_freq"] > 0:
        raw_ica_fit.filter(l_freq=cfg["ica_l_freq"], h_freq=None, picks="eeg")

    # 6) Fit ICA
    ica = mne.preprocessing.ICA(
        n_components=cfg["ica_n_components"],
        method=cfg["ica_method"],
        random_state=cfg["random_state"],
        max_iter="auto",
    )
    ica.fit(raw_ica_fit)

    # 7) Detect artifacts
    eog_inds, eog_scores = [], []
    ecg_inds, ecg_scores = [], []

    # Prefer ICA methods (correct usage: ica.find_bads_eog/ecg)
    try:
        eog_inds, eog_scores = ica.find_bads_eog(raw_ica_fit, ch_name=cfg["eog_ch_name"])
    except Exception as e:
        # Fallback: try without explicit ch_name, or skip
        try:
            eog_inds, eog_scores = ica.find_bads_eog(raw_ica_fit)
        except Exception:
            pass  # no EOG detection

    try:
        ecg_inds, ecg_scores = ica.find_bads_ecg(raw_ica_fit, ch_name=cfg["ecg_ch_name"], method="correlation")
    except Exception:
        try:
            ecg_inds, ecg_scores = ica.find_bads_ecg(raw_ica_fit, method="correlation")
        except Exception:
            pass  # no ECG detection

    exclude = sorted(set((eog_inds or []) + (ecg_inds or [])))
    if cfg["ica_auto_exclude"]:
        ica.exclude = exclude

    # 8) Apply ICA to the task-filtered signal
    raw_clean = raw_pre.copy()
    if cfg["apply_ica"] and exclude:
        raw_clean = ica.apply(raw_clean.copy())

    # 9) Save (optional)
    if cfg["save_base"]:
        base = str(cfg["save_base"])
        ica.save(f"{base}-ica.fif", overwrite=True)
        raw_clean.save(f"{base}_preproc_raw.fif", overwrite=True)

    report = {
        "exclude": exclude,
        "n_excluded": len(exclude),
        "eog_inds": eog_inds, "ecg_inds": ecg_inds,
        "cfg": cfg,
    }
    return raw_clean, ica, report
