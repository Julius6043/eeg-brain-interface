"""Filter-Helferfunktionen."""

from __future__ import annotations
import mne
from mne.io import BaseRaw
from typing import Iterable


def apply_bandpass(raw: BaseRaw, l_freq: float, h_freq: float) -> None:
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)


def apply_notch(raw: BaseRaw, freqs: Iterable[float]) -> None:
    raw.notch_filter(freqs=freqs, verbose=False)
