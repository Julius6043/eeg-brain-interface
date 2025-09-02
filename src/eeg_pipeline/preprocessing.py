"""
EEG Preprocessing Module

Dieses Modul enthält alle Funktionen für das Preprocessing von EEG-Daten.
"""

from dataclasses import dataclass
from typing import Optional
from mne.io import Raw
from mne.preprocessing import ICA


@dataclass
class PreprocessingConfig:
    notch_freq: float = 50.0
    l_freq: float = 1.0
    h_freq: float = 40.0
    reference: str = 'average'
    run_ica: bool = False
    ica_n_components: float = 0.95
    ica_random_state: int = 42


def apply_notch_filter(raw: Raw, notch_freq: float = 50.0) -> Raw:
    raw.notch_filter(
        freqs=[notch_freq, 2 * notch_freq],
        picks="eeg",
        verbose="WARNING"
    )
    return raw


def apply_bandpass_filter(raw: Raw, l_freq: float = 1.0, h_freq: float = 40.0) -> Raw:
    """Wende Bandpass-Filter an."""
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks="eeg",
        method="fir",
        phase="zero",
        fir_window="hamming",
        verbose="WARNING"
    )
    return raw


def apply_rereferencing(raw: Raw, reference: str = 'average') -> Raw:
    """Wende Re-Referenzierung an."""
    raw.set_eeg_reference(reference)
    return raw


def apply_ica(raw: Raw, n_components: float = 0.95, random_state: int = 42) -> Raw:
    """Wende ICA zur Artefakt-Entfernung an."""
    ica = ICA(n_components=n_components, random_state=random_state)
    ica.fit(raw)

    # Automatische Artefakt-Erkennung
    try:
        eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=3.0)
        ica.exclude = eog_indices
        ica.apply(raw)
        print(f"ICA applied, excluded {len(eog_indices)} components")
    except Exception as e:
        print(f"ICA failed: {e}")

    return raw


def preprocess_raw(raw: Raw, config: PreprocessingConfig) -> Raw:
    """Vollständiges Preprocessing einer Raw-Datei."""
    # Notch filter
    raw = apply_notch_filter(raw, config.notch_freq)

    # Bandpass filter
    raw = apply_bandpass_filter(raw, config.l_freq, config.h_freq)

    # Re-reference
    raw = apply_rereferencing(raw, config.reference)

    # Optional ICA
    if config.run_ica:
        raw = apply_ica(raw, config.ica_n_components, config.ica_random_state)

    return raw
