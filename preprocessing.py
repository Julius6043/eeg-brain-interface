"""
EEG preprocessing pipeline for BCI/ML on XDF recordings.

Steps:
  1) Load XDF (robust loader with fallback)
  2) Pick EEG stream, convert to MNE Raw (auto µV→V), keep 8 channels
  3) Save presentation plots: raw+PSD (before)
  4) Notch filter at mains (50/60 Hz) + optional harmonic
  5) Band-pass (e.g., 1–40 Hz)
  6) Re-reference to average
  7) Optional ICA for EOG/EMG artifact removal + before/after comparison plots
  8) Save cleaned FIF
  9) Optional ML export: sliding windows (raw) + bandpower features

Requires: mne, numpy, matplotlib, pyxdf, scipy
"""

python c:/Users/janni/Documents/GitHub/eeg-brain-interface/preprocessing.py --xdf c:/Users/janni/Documents/GitHub/eeg-brain-interface/data/sub-P001_ses-S001_task-Default_run-001_eeg.xdf --outdir c:/Users/janni/Documents/GitHub/eeg-brain-interface/results

import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless, safe on any machine
import matplotlib.pyplot as plt
import mne
import pyxdf
from typing import Optional, List, Tuple
from mne.time_frequency import psd_array_welch


# ----------------------------
# Utilities
# ----------------------------
def load_xdf_safe(path: Path):
    """Load XDF with robust fallbacks against truncated clock chunks."""
    try:
        streams, header = pyxdf.load_xdf(
            str(path),
            synchronize_clocks=True,
            dejitter_timestamps=True,
            handle_clock_resets=True,
        )
        return streams, header
    except Exception as e:
        print(f"[WARN] Full XDF load failed: {e}")
        print("[INFO] Retrying without sync/dejitter/reset...")
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
    """Pick EEG stream + (optional) marker stream."""
    eeg_stream = None
    marker_stream = None
    for st in streams:
        info = st.get("info", {})
        stype = str(_safe_get(info, "type", "")).lower()
        sname = str(_safe_get(info, "name", "")).lower()
        ch_count = int(float(_safe_get(info, "channel_count", "0")))
        if ("eeg" in stype or "unicorn" in sname) and ch_count >= 1:
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
    """Convert XDF EEG stream to MNE RawArray (auto-scale µV->V), keep subset."""
    info = eeg_stream["info"]
    fs = float(_safe_get(info, "nominal_srate", "0"))
    data = np.array(eeg_stream["time_series"], dtype=float).T  # (n_ch, n_times)

    # Heuristic: if data looks like microvolts (huge), convert µV -> V
    med_abs = float(np.nanmedian(np.abs(data)))
    if med_abs > 1e-3:
        print(f"[INFO] Data median abs={med_abs:.1f} (looks like µV) -> scaling to Volts (x1e-6)")
        data *= 1e-6
    else:
        print(f"[INFO] Data median abs={med_abs:.3e} (looks like Volts) -> no scaling")

    # Channel names
    ch_count = data.shape[0]
    ch_names = []
    try:
        desc = info.get("desc", {})
        chs = desc.get("channels", {})
        clist = chs.get("channel")
        if isinstance(clist, dict):
            clist = [clist]
        if clist:
            for ch in clist:
                lab = ch.get("label", "")
                if isinstance(lab, list):
                    lab = lab[0] if lab else ""
                ch_names.append(str(lab) if lab else f"EEG{len(ch_names)+1}")
    except Exception:
        pass
    if not ch_names or len(ch_names) != ch_count:
        ch_names = [f"EEG{i+1}" for i in range(ch_count)]

    info_mne = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info_mne)

    # Keep exactly 8 channels
    if channels_keep and len(channels_keep) > 0:
        keep = [ch for ch in channels_keep if ch in raw.ch_names]
    else:
        keep = raw.ch_names[:8]
    raw.pick_channels(keep)
    print("[INFO] Channels kept:", raw.ch_names)

    # Optional montage (won't error if names don't match)
    try:
        if montage:
            raw.set_montage(montage, on_missing="ignore")
    except Exception as e:
        print(f"[WARN] Could not set montage {montage}: {e}")

    return raw


def save_raw_segment(raw: mne.io.Raw, out: Path, seconds: float = 10.0, uV: float = 100.0):
    dur = min(seconds, raw.n_times / raw.info["sfreq"])
    fig = raw.plot(
        start=0,
        duration=dur,
        show=False,
        decim=1,
        show_first_samp=False,
        scalings={"eeg": uV * 1e-6},
        show_options=False,
    )
    fig.suptitle(f"Raw (first {dur:.1f}s)")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_psd(raw: mne.io.Raw, out: Path, fmin=1, fmax=60, ylim=(-40, 20)):
    fig = raw.plot_psd(fmin=fmin, fmax=fmax, show=False)
    try:
        for ax in fig.axes:
            ax.set_ylim(*ylim)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power (dB/Hz)")
    except Exception:
        pass
    fig.suptitle(f"PSD ({fmin}-{fmax} Hz)")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def bandpower_features(X: np.ndarray, fs: float, bands=None) -> np.ndarray:
    """
    X: (n_ch, n_samples) in Volts
    returns: (n_ch * n_bands,) log-bandpower features
    """
    if bands is None:
        bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (13, 30), "lgamma": (30, 40)}
    # Welch
    nperseg = min(256, X.shape[1])
    noverlap = min(128, max(0, nperseg - 1))
    psds, freqs = psd_array_welch(
        X, sfreq=fs, fmin=min(b[0] for b in bands.values()), fmax=max(b[1] for b in bands.values()),
        n_fft=nperseg, n_overlap=noverlap, n_per_seg=nperseg, average="mean"
    )  # (n_ch, n_freqs) V^2/Hz

    feats = []
    for lo, hi in bands.values():
        mask = (freqs >= lo) & (freqs <= hi)
        bp = psds[:, mask].mean(axis=1)  # mean power per channel in band
        feats.append(np.log(np.maximum(bp, 1e-20)))
    feats = np.stack(feats, axis=1)  # (n_ch, n_bands)
    return feats.reshape(-1)


def export_ml(raw: mne.io.Raw, out_npz: Path, win_s=2.0, hop_s=1.0):
    """Make sliding windows & bandpower features (no labels)."""
    fs = float(raw.info["sfreq"])
    data = raw.get_data()  # (n_ch, n_times) in Volts
    win = int(win_s * fs)
    hop = int(hop_s * fs)
    starts = np.arange(0, data.shape[1] - win + 1, hop, dtype=int)

    X_raw = []
    X_feat = []
    for s in starts:
        seg = data[:, s:s+win]
        X_raw.append(seg.astype(np.float32))
        X_feat.append(bandpower_features(seg, fs).astype(np.float32))

    X_raw = np.stack(X_raw, axis=0) if X_raw else np.empty((0, data.shape[0], win), dtype=np.float32)
    X_feat = np.stack(X_feat, axis=0) if X_feat else np.empty((0, 1), dtype=np.float32)
    np.savez(out_npz, X_raw=X_raw, X_feat=X_feat, fs=np.array(fs, dtype=np.float32),
             win_samples=np.array(win, dtype=np.int32), hop_samples=np.array(hop, dtype=np.int32),
             ch_names=np.array(raw.ch_names, dtype=object))
    print(f"[OK] Saved ML tensors/features -> {out_npz} | windows={len(starts)} shape_raw={X_raw.shape} shape_feat={X_feat.shape}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="EEG preprocessing for BCI/ML (XDF → plots/FIF/features).")
    ap.add_argument("--xdf", required=True, help="Path to .xdf")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--channels", nargs="+", default=None, help="Channel names to keep (e.g., EEG1 EEG2 ...). Defaults to first 8.")
    ap.add_argument("--mains", type=int, default=50, help="Mains frequency (50 EU / 60 US).")
    ap.add_argument("--no-notch", action="store_true", help="Skip notch filtering.")
    ap.add_argument("--bp", nargs=2, type=float, default=[1.0, 40.0], metavar=("L_HZ", "H_HZ"), help="Band-pass [low high] Hz.")
    ap.add_argument("--no-bp", action="store_true", help="Skip band-pass filtering.")
    ap.add_argument("--reref", action="store_true", help="Average re-reference after filters.")
    ap.add_argument("--ica", action="store_true", help="Run ICA and remove EOG components (best-effort).")
    ap.add_argument("--win", type=float, default=2.0, help="Sliding window length (s) for ML export.")
    ap.add_argument("--hop", type=float, default=1.0, help="Sliding window hop (s) for ML export.")
    ap.add_argument("--save-ml", action="store_true", help="Export ML windows and bandpower features (npz).")
    ap.add_argument("--uV", type=float, default=100.0, help="Plot vertical scale per channel in microvolts.")
    args = ap.parse_args()

    xdf_path = Path(args.xdf)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading XDF: {xdf_path}")
    streams, header = load_xdf_safe(xdf_path)
    print(f"[INFO] Loaded {len(streams)} stream(s)")

    eeg_stream, marker_stream = pick_streams(streams)
    raw = eeg_stream_to_raw(eeg_stream, channels_keep=args.channels, montage="standard_1020")
    print(raw)
    print(f"[INFO] Duration ≈ {raw.n_times / raw.info['sfreq']:.1f} s; sfreq={raw.info['sfreq']} Hz")

    # Save initial FIF and initial plots
    raw_init_fif = outdir / "raw_initial_eeg.fif"
    raw.save(str(raw_init_fif), overwrite=True)
    print(f"[OK] Saved: {raw_init_fif}")

    save_raw_segment(raw, outdir / "raw_before.png", seconds=10.0, uV=args.uV)
    save_psd(raw, outdir / "psd_before.png", fmin=1, fmax=120, ylim=(-40, 20))

    # Notch
    raw_proc = raw.copy()
    if not args.no_notch:
        freqs = [args.mains, 2 * args.mains]
        print(f"[INFO] Notch filter @ {freqs} Hz")
        raw_proc.notch_filter(freqs=freqs, picks="eeg", verbose="WARNING")
        save_raw_segment(raw_proc, outdir / "raw_after_notch.png", seconds=10.0, uV=args.uV)
        save_psd(raw_proc, outdir / "psd_after_notch.png", fmin=1, fmax=120, ylim=(-40, 20))
    else:
        print("[INFO] Skipping notch filter")

    # Band-pass
    if not args.no_bp:
        l_freq, h_freq = args.bp
        print(f"[INFO] Band-pass filter {l_freq}-{h_freq} Hz (zero-phase FIR, Hamming)")
        raw_proc.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg",
                        method="fir", phase="zero", fir_window="hamming", verbose="WARNING")
        save_raw_segment(raw_proc, outdir / "raw_after_bandpass.png", seconds=10.0, uV=args.uV)
        save_psd(raw_proc, outdir / "psd_after_bandpass.png", fmin=1, fmax=120, ylim=(-40, 20))
    else:
        print("[INFO] Skipping band-pass filter")

    # Re-reference
    if args.reref:
        print("[INFO] Setting average reference over kept channels")
        raw_proc.set_eeg_reference("average")
        save_raw_segment(raw_proc, outdir / "raw_after_reref.png", seconds=10.0, uV=args.uV)
        save_psd(raw_proc, outdir / "psd_after_reref.png", fmin=1, fmax=120, ylim=(-40, 20))
        print("[INFO] custom_ref_applied =", raw_proc.info.get("custom_ref_applied", False))
    else:
        print("[INFO] Skipping re-reference")

    # Optional ICA
    if args.ica:
        try:
            from mne.preprocessing import ICA, create_eog_epochs
            print("[INFO] Running ICA (high-pass 1 Hz for fit)")
            raw_for_ica = raw_proc.copy().filter(l_freq=1.0, h_freq=None, picks="eeg",
                                                 method="fir", phase="zero", fir_window="hamming", verbose="WARNING")
            ica = ICA(n_components=None, random_state=97, method="fastica", max_iter="auto")
            ica.fit(raw_for_ica)

            # Try automatic EOG detection
            try:
                eog_epochs = create_eog_epochs(raw_proc, ch_name=None, reject_by_annotation=True)
                eog_inds, scores = ica.find_bads_eog(eog_epochs)
                ica.exclude = eog_inds
                print("[INFO] Excluding EOG-like components:", ica.exclude)
            except Exception as ee:
                print("[WARN] Could not auto-detect EOG components:", ee)

            raw_ica = ica.apply(raw_proc.copy())

            # Comparison plots
            # Time-domain stacked 10 s
            fs = raw_proc.info["sfreq"]
            nsec = 10
            n_samp = int(nsec * fs)
            t = np.arange(n_samp) / fs
            seg_b = raw_proc.get_data(start=0, stop=n_samp) * 1e6  # µV
            seg_a = raw_ica.get_data(start=0, stop=n_samp) * 1e6

            spacing = 120.0
            offsets = np.arange(seg_b.shape[0])[::-1] * spacing
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            for i, ch in enumerate(raw_proc.ch_names):
                axes[0].plot(t, seg_b[i] + offsets[i], lw=1)
                axes[0].text(0, offsets[i], ch, va="center", ha="right", fontsize=8)
                axes[1].plot(t, seg_a[i] + offsets[i], lw=1)
            axes[0].set_title("Before ICA (10 s)"); axes[1].set_title("After ICA (10 s)")
            for ax in axes:
                ax.set_xlim(0, nsec); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.2)
            axes[0].set_ylabel("Amplitude (µV, stacked)")
            axes[0].set_ylim(-spacing, offsets[0] + spacing)
            fig.tight_layout(); fig.savefig(outdir / "ica_compare_timeseries.png", dpi=150); plt.close(fig)

            # PSD overlay (avg across channels)
            def avg_psd_db(raw_obj, fmin=1.0, fmax=40.0):
                X = raw_obj.get_data()
                fs_ = float(raw_obj.info["sfreq"])
                nperseg = min(256, X.shape[1]); noverlap = min(128, max(0, nperseg - 1))
                psds, freqs = psd_array_welch(
                    X, sfreq=fs_, fmin=fmin, fmax=fmax,
                    n_fft=nperseg, n_overlap=noverlap, n_per_seg=nperseg, average="mean"
                )
                psds_db = 10 * np.log10(np.maximum(psds, 1e-20))
                return freqs, psds_db.mean(axis=0)

            fmin, fmax = 1.0, 40.0
            fb, Pb = avg_psd_db(raw_proc, fmin, fmax)
            fa, Pa = avg_psd_db(raw_ica,  fmin, fmax)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(fb, Pb, label="Before ICA", lw=2)
            ax.plot(fa, Pa, label="After ICA",  lw=2)
            ax.set_xlim(fmin, fmax); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (dB/Hz)")
            ax.set_title("PSD (mean across channels)"); ax.grid(True, alpha=0.3); ax.legend()
            fig.tight_layout(); fig.savefig(outdir / "ica_compare_psd.png", dpi=150); plt.close(fig)

            # Continue with ICA-cleaned data
            raw_proc = raw_ica
        except Exception as e:
            print("[WARN] ICA step failed:", e)
    else:
        print("[INFO] Skipping ICA")

    # Save cleaned raw
    raw_clean_fif = outdir / "raw_clean.fif"
    raw_proc.save(str(raw_clean_fif), overwrite=True)
    print(f"[OK] Saved cleaned FIF: {raw_clean_fif}")

    # Optional ML export
    if args.save_ml:
        export_ml(raw_proc, outdir / "ml_windows_bandpower.npz", win_s=args.win, hop_s=args.hop)

    print("[DONE] All outputs in:", outdir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", e)
        sys.exit(1)
