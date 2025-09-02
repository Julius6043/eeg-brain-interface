"""Convenience entry point to run the EEG training pipeline on local sample data.

Dieses Skript bietet eine einfache Funktion, um die bestehende Pipeline
(`eeg_pipeline.src.train`) ohne lange Kommandozeilen direkt auszuführen.

Verwendung (interaktiv / Notebook / Konsole):

        from main import run_default_pipeline
        run_default_pipeline(model="logreg", deep=False)

Oder einfach:

        python main.py

Die Funktion versucht automatisch passende Dateien zu finden:
  * Rohdaten: bevorzugt `raw_initial_eeg.fif`, sonst erste *.xdf Datei
  * Marker:  `markers_all.json` oder erste *.json mit "marker" im Namen

Parameter
---------
model : str
        'logreg' oder 'svm'.
deep : bool
        Wenn True wird zusätzlich EEGNet (PyTorch) trainiert.
data_root : str | Path
        Wurzelverzeichnis, unter dem nach Dateien gesucht wird (Standard: Projektwurzel).
raise_on_missing : bool
        Wenn True -> Exception bei fehlenden Dateien, sonst freundliche Meldung & Rückkehr.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import sys

try:
    from eeg_pipeline.src import train as pipeline_train
except ImportError as e:  # pragma: no cover
    pipeline_train = None  # type: ignore
    _import_error = e  # type: ignore
else:
    _import_error = None


def _find_first(patterns: Sequence[str], base: Path) -> Optional[Path]:
    for pat in patterns:
        for p in base.rglob(pat):
            if p.is_file():
                return p
    return None


def run_default_pipeline(
    model: str = "logreg",
    deep: bool = False,
    data_root: str | Path = ".",
    raise_on_missing: bool = False,
) -> Optional[int]:
    """Run the training pipeline with auto‑discovered data files.

    Returns the exit code from the underlying pipeline `main` or None if aborted.
    """
    if pipeline_train is None:
        msg = f"Pipeline-Modul konnte nicht importiert werden: {_import_error}"
        if raise_on_missing:
            raise RuntimeError(msg)
        print(msg)
        return None

    base = Path(data_root).resolve()
    project_root = Path(__file__).resolve().parent
    # Falls data_root relativ ist, auf Projektwurzel beziehen
    if not base.is_absolute():
        base = (project_root / base).resolve()

    if not base.exists():
        msg = f"Daten-Verzeichnis existiert nicht: {base}"
        if raise_on_missing:
            raise FileNotFoundError(msg)
        print(msg)
        return None

    # Kandidaten für Rohdaten
    raw = _find_first(
        [
            "raw_initial_eeg.fif",
            "**/raw_initial_eeg.fif",
            "*.xdf",
            "**/*.xdf",
        ],
        base,
    )

    # Marker Datei
    markers = _find_first(
        [
            "markers_all.json",
            "**/markers_all.json",
            "*marker*.json",
            "**/*marker*.json",
        ],
        base,
    )

    if raw is None or markers is None:
        msg = (
            "Konnte benötigte Dateien nicht finden. Gefunden wurde:\n"
            f"  raw: {raw}\n  markers: {markers}\n  Suchbasis: {base}"
        )
        if raise_on_missing:
            raise FileNotFoundError(msg)
        print(msg)
        return None

    print("Gefundene Dateien:")
    print(f"  RAW     : {raw}")
    print(f"  MARKERS : {markers}")
    print(f"Starte Pipeline (model={model}, deep={deep}) ...")

    argv = [
        "--raw",
        str(raw),
        "--markers",
        str(markers),
        "--model",
        model,
    ]
    if deep:
        argv.append("--deep")

    # Aufruf der bestehenden CLI-Funktion
    try:
        return pipeline_train.main(argv)
    except SystemExit as e:  # falls argparse sys.exit nutzt
        return int(e.code)


def main():  # CLI Wrapper
    # Sehr einfache Argument-Weitergabe (optional: könnte auf argparse erweitert werden)
    model = "logreg"
    deep = False
    if "--svm" in sys.argv:
        model = "svm"
    if "--deep" in sys.argv:
        deep = True
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Usage: python main.py [--svm] [--deep] [--data PATH]\n"
            "  --svm        : Verwende SVM statt logreg\n"
            "  --deep       : EEGNet zusätzlich trainieren\n"
            "  --data PATH  : Alternatives Datenwurzel-Verzeichnis (Default Projektwurzel)\n"
        )
        return 0
    # Optionaler Datenpfad
    if "--data" in sys.argv:
        idx = sys.argv.index("--data")
        try:
            data_root = sys.argv[idx + 1]
        except IndexError:
            print("--data benötigt einen Pfad")
            return 2
    else:
        data_root = "."
    return run_default_pipeline(model=model, deep=deep, data_root=data_root) or 0


if __name__ == "__main__":
    sys.exit(main())
