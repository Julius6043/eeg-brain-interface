"""Preprocessing-Pipeline (vereinfachtes Beispiel).

Hinweis: Für produktive Nutzung können zusätzliche Schritte wie:
 - Bad-Channel Detection (pyprep / RANSAC)
 - ICA Artefaktentfernung
 - Re-Referenzierung (robuste Durchschnittsreferenz)
ergänzt werden.
"""

from __future__ import annotations
import mne
from mne.io import BaseRaw
from typing import Tuple, Iterable
from . import filters
from config import settings


def basic_preprocess(
    raw: BaseRaw,
    resample: bool = True,
    bandpass: Tuple[float, float] | None = settings.BANDPASS,
    notch: Iterable[float] | None = settings.NOTCH_FREQS,
    picks: str = "eeg",
) -> BaseRaw:
    """Führt einfache Preprocessing-Schritte auf Rohdaten aus.

    Parameter
    ---------
    raw : mne.io.BaseRaw
        Vorab geladene Rohdaten (preload=True empfohlen).
    resample : bool
        Downsampling zur Reduktion von Rechenaufwand.
    bandpass : (low, high) | None
        Grenzen des Bandpass-Filters.
    notch : Iterable[float] | None
        Frequenzen für Notch-Filter (z. B. Netzbrummen 50 Hz).
    picks : str
        Kanaltyp (standard 'eeg').
    """
    raw.load_data()
    # Stim-Kanäle behalten für Event-Extraktion; später könnte man sie droppen
    if picks == "eeg":
        raw.pick_types(eeg=True, stim=True, meg=False)
    else:
        raw.pick(picks)
    if resample:
        raw.resample(settings.RESAMPLE_SFREQ)
    if bandpass:
        filters.apply_bandpass(raw, *bandpass)
    if notch:
        filters.apply_notch(raw, notch)
    return raw


def create_epochs(
    raw: BaseRaw,
    event_id: dict[str, int],
    tmin: float = settings.TMIN,
    tmax: float = settings.TMAX,
) -> mne.Epochs:
    """Extrahiert Events und erzeugt Epochen.

    Versucht zuerst Standard-Event-Kanäle. Falls keine Stim-Kanäle
    vorhanden sind (ValueError), wird auf Annotationen ausgewichen.

    Baseline wird hier bewusst nicht angewendet (baseline=None).
    """
    try:
        events = mne.find_events(raw, verbose=False)
    except ValueError:
        # Fallback: Events aus Annotationen
        events, _ = mne.events_from_annotations(raw, verbose=False)
        # Filter event_id auf vorhandene Codes
        available = set(events[:, 2])
        event_id = {k: v for k, v in event_id.items() if v in available}
        if not event_id:
            raise RuntimeError("Keine passenden Events in Annotationen gefunden.")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs
