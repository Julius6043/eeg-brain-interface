"""FIF Analysis / Debug Utility

Dieses Skript lädt eine FIF-Datei (Raw), extrahiert strukturierte
Diagnoseinformationen und gibt sie menschenlesbar auf der Konsole aus
ODER schreibt bei Bedarf eine Markdown / JSON Zusammenfassung.

Nutzung (Beispiele):
    python scripts/fif_analysis.py --fif path/to/file.fif
    python scripts/fif_analysis.py --fif results/processed/jannik/indoor_processed_raw.fif --markdown report.md
    python scripts/fif_analysis.py --fif results/processed/jannik/indoor_processed_raw.fif --json report.json --head 10

Fokus:
    * Meta / Header (Samplingrate, Kanäle, Dauer)
    * Kanalstatistiken (min/max/mean/std, Missing / NaNs)
    * Basis-Spektren (optionale schnelle PSD Schätzung)
    * Annotations / Events (falls vorhanden)
    * Plausibilitätschecks (z.B. konstante Kanäle, extreme Amplituden)

Performance-Hinweis:
    Für große Aufnahmen können Statistiken auf ein Subsegment begrenzt werden (--max-seconds).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mne
import numpy as np

# ------------------------- Data Structures ------------------------- #


@dataclass
class ChannelStats:
    name: str
    ch_type: str
    n_samples: int
    mean: float
    std: float
    var: float
    min: float
    max: float
    ptp: float  # peak-to-peak
    median: float
    mad: float  # median absolute deviation
    n_zeros: int
    n_nans: int
    pct_zeros: float
    flat: bool
    high_amp: bool


@dataclass
class GlobalStats:
    file: str
    size_mb: float
    sfreq: float
    n_channels: int
    ch_types: Dict[str, int]
    duration_sec: float
    n_samples: int
    notch_applied: bool
    highpass: Optional[float]
    lowpass: Optional[float]


@dataclass
class AnnotationInfo:
    description: str
    onset: float
    duration: float


@dataclass
class AnalysisResult:
    global_stats: GlobalStats
    channel_stats: List[ChannelStats]
    annotations: List[AnnotationInfo]
    suspicious_channels: List[str]
    psd_summary: Optional[Dict[str, Any]] = None


# ------------------------- Helper Functions ------------------------- #


def human_readable_bytes(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 2)


def compute_channel_stats(raw: mne.io.BaseRaw, data: np.ndarray) -> List[ChannelStats]:
    stats: List[ChannelStats] = []
    ch_types = raw.get_channel_types()
    for idx, ch_name in enumerate(raw.ch_names):
        samples = data[idx]
        finite = np.isfinite(samples)
        finite_samples = samples[finite] if finite.any() else np.array([])
        if finite_samples.size == 0:
            # Degenerate channel
            stats.append(
                ChannelStats(
                    name=ch_name,
                    ch_type=ch_types[idx],
                    n_samples=int(samples.size),
                    mean=float("nan"),
                    std=float("nan"),
                    var=float("nan"),
                    min=float("nan"),
                    max=float("nan"),
                    ptp=float("nan"),
                    median=float("nan"),
                    mad=float("nan"),
                    n_zeros=int((samples == 0).sum()),
                    n_nans=int((~finite).sum()),
                    pct_zeros=100.0 if samples.size else 0.0,
                    flat=True,
                    high_amp=False,
                )
            )
            continue
        mean = float(finite_samples.mean())
        std = float(finite_samples.std(ddof=0))
        var = float(std**2)
        min_v = float(finite_samples.min())
        max_v = float(finite_samples.max())
        ptp_v = float(max_v - min_v)
        median_v = float(np.median(finite_samples))
        mad_v = float(np.median(np.abs(finite_samples - median_v)))
        n_zeros = int((finite_samples == 0).sum())
        n_nans = int((~finite).sum())
        flat = std < 1e-9 or ptp_v < 1e-6
        high_amp = max(abs(min_v), abs(max_v)) > 1e-3  # >1mV heuristisch
        stats.append(
            ChannelStats(
                name=ch_name,
                ch_type=ch_types[idx],
                n_samples=int(samples.size),
                mean=mean,
                std=std,
                var=var,
                min=min_v,
                max=max_v,
                ptp=ptp_v,
                median=median_v,
                mad=mad_v,
                n_zeros=n_zeros,
                n_nans=n_nans,
                pct_zeros=(n_zeros / samples.size * 100.0) if samples.size else 0.0,
                flat=flat,
                high_amp=high_amp,
            )
        )
    return stats


def summarize_psd(
    raw: mne.io.BaseRaw, picks: Optional[Sequence[str]] = None, fmax: float = 45.0
) -> Dict[str, Any]:
    """Compute a lightweight PSD summary (average over channels).

    Aktuelle MNE-Version liefert ein Spectrum-Objekt zurück (nicht Tuple).
    """
    if picks is None:
        picks = mne.pick_types(raw.info, eeg=True, meg=False, seeg=False, eog=False)
    spectrum = raw.compute_psd(fmax=fmax, picks=picks, verbose="ERROR")
    data = spectrum.get_data()  # shape (n_channels, n_freqs)
    freqs = spectrum.freqs
    psd_db = 10 * np.log10(data)
    mean_db = psd_db.mean(axis=0)

    def band_power(f_lo, f_hi):
        idx = (freqs >= f_lo) & (freqs < f_hi)
        return float(mean_db[idx].mean()) if idx.any() else float("nan")

    return {
        "freqs": freqs.tolist(),
        "mean_psd_db": mean_db.tolist(),
        "bands_db": {
            "delta(1-4)": band_power(1, 4),
            "theta(4-8)": band_power(4, 8),
            "alpha(8-13)": band_power(8, 13),
            "beta(13-30)": band_power(13, 30),
        },
    }


def detect_suspicious_channels(ch_stats: List[ChannelStats]) -> List[str]:
    bads: List[str] = []
    for st in ch_stats:
        if st.flat or st.high_amp or st.n_nans > 0 or st.pct_zeros > 50:
            bads.append(st.name)
    return bads


def collect_global_stats(raw: mne.io.BaseRaw, fif_path: Path) -> GlobalStats:
    info = raw.info
    size_mb = (
        human_readable_bytes(fif_path.stat().st_size) if fif_path.exists() else 0.0
    )
    ch_types = {}
    for t in raw.get_channel_types():
        ch_types[t] = ch_types.get(t, 0) + 1
    filt = info.get("highpass", None), info.get("lowpass", None)
    notch_applied = (
        any("Notch" in str(step) for step in info.get("proc_history", []))
        if "proc_history" in info
        else False
    )
    return GlobalStats(
        file=str(fif_path),
        size_mb=size_mb,
        sfreq=float(info["sfreq"]),
        n_channels=int(len(raw.ch_names)),
        ch_types=ch_types,
        duration_sec=float(raw.times[-1]) if raw.n_times else 0.0,
        n_samples=int(raw.n_times),
        notch_applied=notch_applied,
        highpass=info.get("highpass", None),
        lowpass=info.get("lowpass", None),
    )


def extract_annotations(raw: mne.io.BaseRaw) -> List[AnnotationInfo]:
    if not raw.annotations:
        return []
    ann_list: List[AnnotationInfo] = []
    for ann in raw.annotations:
        ann_list.append(
            AnnotationInfo(
                description=str(ann["description"]),
                onset=float(ann["onset"]),
                duration=float(ann["duration"]),
            )
        )
    return ann_list


def analyze_fif(
    fif_path: Path,
    head: Optional[int] = None,
    max_seconds: Optional[float] = None,
    compute_psd_flag: bool = False,
) -> AnalysisResult:
    if not fif_path.exists():
        raise FileNotFoundError(f"FIF nicht gefunden: {fif_path}")

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")

    # Optionale Kürzung der Datenmenge für schnellere Statistiken
    if max_seconds is not None and raw.times[-1] > max_seconds:
        raw = raw.copy().crop(tmax=max_seconds)

    data = raw.get_data()  # shape (n_channels, n_samples)
    if head is not None:
        data = data[:, :head]

    channel_stats = compute_channel_stats(raw, data)
    suspicious = detect_suspicious_channels(channel_stats)
    global_stats = collect_global_stats(raw, fif_path)
    annotations = extract_annotations(raw)

    psd_summary = summarize_psd(raw) if compute_psd_flag else None

    return AnalysisResult(
        global_stats=global_stats,
        channel_stats=channel_stats,
        annotations=annotations,
        suspicious_channels=suspicious,
        psd_summary=psd_summary,
    )


# ------------------------- Output Formatting ------------------------- #


def format_result_text(res: AnalysisResult, limit_channels: Optional[int] = 25) -> str:
    gs = res.global_stats
    lines: List[str] = []
    lines.append("# Global")
    lines.append(f"File: {gs.file}")
    lines.append(
        f"Size: {gs.size_mb} MB | Duration: {gs.duration_sec:.1f}s | sfreq={gs.sfreq}"
    )
    lines.append(f"Channels: {gs.n_channels} ({gs.ch_types})")
    lines.append(
        f"Highpass={gs.highpass} Lowpass={gs.lowpass} NotchApplied={gs.notch_applied}"
    )
    if res.suspicious_channels:
        lines.append(f"Suspicious channels: {', '.join(res.suspicious_channels)}")
    lines.append("")

    lines.append("# Channels (stats)")
    header = "name type mean std min max ptp zeros% flat hiAmp"
    lines.append(header)
    for idx, st in enumerate(res.channel_stats):
        if limit_channels and idx >= limit_channels:
            lines.append(f"... ({len(res.channel_stats)-limit_channels} more)")
            break
        lines.append(
            f"{st.name} {st.ch_type} {st.mean:.2e} {st.std:.2e} {st.min:.2e} {st.max:.2e} "
            f"{st.ptp:.2e} {st.pct_zeros:.1f} {int(st.flat)} {int(st.high_amp)}"
        )
    lines.append("")

    if res.annotations:
        lines.append(f"# Annotations (n={len(res.annotations)})")
        for a in res.annotations[:50]:
            lines.append(f"{a.onset:8.2f}s dur={a.duration:6.2f}s desc={a.description}")
        if len(res.annotations) > 50:
            lines.append(f"... ({len(res.annotations)-50} more)")
        lines.append("")
    else:
        lines.append("# Annotations: none\n")

    if res.psd_summary:
        lines.append("# PSD (mean dB) bands")
        for k, v in res.psd_summary["bands_db"].items():
            lines.append(f"{k}: {v:.2f} dB")
        lines.append("")

    return "\n".join(lines)


def write_markdown(res: AnalysisResult, path: Path) -> None:
    text = format_result_text(res, limit_channels=None)
    path.write_text(text, encoding="utf-8")


def write_json(res: AnalysisResult, path: Path) -> None:
    obj = {
        "global": asdict(res.global_stats),
        "channels": [asdict(cs) for cs in res.channel_stats],
        "annotations": [asdict(a) for a in res.annotations],
        "suspicious_channels": res.suspicious_channels,
        "psd_summary": res.psd_summary,
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ------------------------- CLI ------------------------- #


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Analysiere eine FIF-Datei und zeige menschenlesbare Debug-Infos."
    )
    ap.add_argument("--fif", type=Path, required=True, help="Pfad zur Raw FIF Datei")
    ap.add_argument(
        "--head",
        type=int,
        default=None,
        help="Nur erste N Samples pro Kanal analysieren",
    )
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Aufnahme auf diese Länge kürzen (Performance)",
    )
    ap.add_argument("--psd", action="store_true", help="Zusätzlich PSD zusammenfassen")
    ap.add_argument("--markdown", type=Path, help="Markdown-Report schreiben")
    ap.add_argument("--json", type=Path, help="JSON-Report schreiben")
    ap.add_argument(
        "--no-console", action="store_true", help="Keine Ausgabe auf STDOUT"
    )
    ap.add_argument(
        "--limit-channels",
        type=int,
        default=25,
        help="Kanalzeilen in Text-Ausgabe beschränken",
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        res = analyze_fif(
            args.fif,
            head=args.head,
            max_seconds=args.max_seconds,
            compute_psd_flag=args.psd,
        )
    except Exception as e:  # pragma: no cover - CLI robustness
        print(f"[ERROR] Analyse fehlgeschlagen: {e}")
        return 1

    if not args.no_console:
        print(format_result_text(res, limit_channels=args.limit_channels))

    if args.markdown:
        write_markdown(res, args.markdown)
        print(f"[INFO] Markdown geschrieben: {args.markdown}")
    if args.json:
        write_json(res, args.json)
        print(f"[INFO] JSON geschrieben: {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
