"""Minimaler End-to-End Lauf.

Beispielaufrufe:
  python run_example.py --mode classical --limit-epochs 150
  python run_example.py --mode deep --limit-epochs 80 --dl-epochs 3

"""

from __future__ import annotations
import argparse
from utils.logging_utils import setup_logging
from preprocessing import pipeline
from config import settings
from models import train as train_mod
import mne


def load_sample_raw() -> mne.io.BaseRaw:
    from mne.datasets import sample
    from pathlib import Path

    data_path = Path(sample.data_path(verbose=False))
    fif_path = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["classical", "deep"], default="classical")
    parser.add_argument(
        "--limit-epochs",
        type=int,
        default=120,
        help="Reduziert Anzahl Epochen zur Beschleunigung",
    )
    parser.add_argument("--dl-epochs", type=int, default=3)
    args = parser.parse_args()
    log = setup_logging()

    log.info("Lade Rohdaten...")
    raw = load_sample_raw()
    log.info("Preprocessing...")
    raw = pipeline.basic_preprocess(raw)

    # Beispiel-Event IDs (Audio links / rechts wie im Sample Dataset)
    event_id = {"auditory/left": 1, "auditory/right": 2}
    epochs = pipeline.create_epochs(raw, event_id)
    if args.limit_epochs:
        epochs = epochs[: args.limit_epochs]
    log.info(
        f"Epochen: {len(epochs)} | Kanäle: {len(epochs.ch_names)} | Samples: {epochs.get_data().shape[-1]}"
    )

    if args.mode == "classical":
        results = train_mod.train_classical(epochs, use_csp=True, use_bandpower=True)
        for k, v in results.items():
            log.info(f"{k}: {v}")
    else:
        res = train_mod.train_deep(epochs, dl_epochs=args.dl_epochs)
        log.info(f"Deep Learning Loss Historie: {[round(x, 4) for x in res.history]}")
    log.info("Fertig.")


if __name__ == "__main__":
    main()
