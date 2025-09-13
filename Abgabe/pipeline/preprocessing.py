"""Core EEG Preprocessing Steps.

Dieses Modul implementiert elementare Signalvorverarbeitung:
    * Netzbrumm-Unterdrückung via Notch-Filter (Grundfrequenz + erste Harmonische)
    * Bandpass (FIR Zero-Phase)
    * Re-Referenzierung (Standard: Durchschnittsreferenz)
    * (Optional) ICA zur Artefaktentfernung (einfacher EOG-Heuristik Workflow)

Bewusste Einfachheit:
    * Kein Resampling – könnte vorgeschaltet werden.
    * ICA automatisiert nur EOG-Komponenten (keine ECG/Muskel Separierung).
    * Keine Parameter-Validierung (z.B. l_freq < h_freq) – TODO markiert.

Erweiterungsideen / TODO:
    - Nyquist-Schutz für Notch-Harmoniken (falls Sampling-Rate < 2*notch_freq*2).
    - Adaptive Auswahl 50/60 Hz via Spektrumabschätzung.
    - Persistenz der ICA-Matrix für Wiederverwendung / Auditing.
    - Artefaktannotation (statt manipulativer Projektion) für reversible Pipelines.
"""

from dataclasses import dataclass
from typing import Optional
from mne.io import Raw
from mne.preprocessing import ICA


@dataclass
class PreprocessingConfig:
    """Parametercontainer für die Vorverarbeitung.

    Attribute
    ---------
    notch_freq:
        Grundfrequenz (Netzbrumm) für Notch-Filter (Harmonische: 2*f eingeschlossen).
    l_freq / h_freq:
        Grenzfrequenzen des Bandpass-Filters (FIR). TODO: Validierung einbauen.
    reference:
        Referenzschema ("average" oder Kanalname); wird direkt an MNE durchgereicht.
    run_ica:
        Steuerflag zur Aktivierung der ICA.
    ica_n_components:
        Anzahl / Varianzschwelle der ICA (float -> explained variance).
    ica_random_state:
        Seed für deterministisches Verhalten.
    """

    notch_freq: float = 50.0
    l_freq: float = 1.0
    h_freq: float = 40.0
    reference: str = "average"
    run_ica: bool = False
    ica_n_components: float = 0.95
    ica_random_state: int = 42


def apply_notch_filter(raw: Raw, notch_freq: float = 50.0) -> Raw:
    """Unterdrücke Netzbrumm (Grundfrequenz + erste Harmonische).

    Hinweise:
        * Keine Prüfung auf Nyquist (kann bei niedriger sfreq zu Warnungen führen).
        * Erweiterbar: Mehr Harmonische nur falls sfreq ausreichend.
    """
    raw.notch_filter(
        freqs=[notch_freq, 2 * notch_freq],
        picks="eeg",
        verbose="WARNING",
    )
    return raw


def apply_bandpass_filter(raw: Raw, l_freq: float = 1.0, h_freq: float = 40.0) -> Raw:
    """Bandpass-FIR-Filter (Zero-Phase) anwenden.

    Parameterwahl: Hamming-Fenster (Standard-kompromiss), Zero-Phase für phasenneutrale
    Zeitreihen (Artefaktrisiko minimal erhöht bei Rand).
    """
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks="eeg",
        method="fir",
        phase="zero",
        fir_window="hamming",
        verbose="WARNING",
    )
    return raw


def apply_rereferencing(raw: Raw, reference: str = "average") -> Raw:
    """Setze EEG-Referenz (Durchschnitt oder spezifischer Kanal)."""
    raw.set_eeg_reference(reference)
    return raw


def apply_ica(raw: Raw, n_components: float = 0.95, random_state: int = 42) -> Raw:
    """Führe eine einfache ICA zur Artefaktentfernung durch.

    Workflow:
        1. Fit ICA auf gefilterten Daten.
        2. EOG-Artefakte identifizieren (Schwelle=3.0 std – heuristisch).
        3. Markierte Komponenten exkludieren und anwenden.

    Fehlerbehandlung: ICA-Fehlschläge (z.B. zu kurze Aufnahme) werden abgefangen,
    ohne die restliche Pipeline zu stoppen.
    """
    ica = ICA(n_components=n_components, random_state=random_state)
    ica.fit(raw)
    try:
        eog_indices, _eog_scores = ica.find_bads_eog(raw, threshold=3.0)
        ica.exclude = eog_indices
        ica.apply(raw)
        print(f"ICA applied, excluded {len(eog_indices)} components")
    except Exception as e:  # pragma: no cover - heuristische Pfade
        print(f"ICA failed: {e}")
    return raw


def preprocess_raw(raw: Raw, config: PreprocessingConfig) -> Raw:
    """Orchestriere Standard-Vorverarbeitung (in-place Modifikation).

    Reihenfolge:
        1. Notch (Netzbrumm & 1. Harmonische)
        2. Bandpass (1–40 Hz per Default)
        3. Re-Referenzierung
        4. (Optional) ICA
    """
    raw = apply_notch_filter(raw, config.notch_freq)
    raw = apply_bandpass_filter(raw, config.l_freq, config.h_freq)
    raw = apply_rereferencing(raw, config.reference)
    if config.run_ica:
        raw = apply_ica(raw, config.ica_n_components, config.ica_random_state)
    return raw
