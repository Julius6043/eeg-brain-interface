"""Epoching Module for EEG Pipeline.

Dieses Modul konvertiert annotierte Raw EEG-Daten in Epochen basierend auf
den Marker-Annotationen. Jede Epoche entspricht einem experimentellen Block
oder einer Baseline-Periode.

Funktionalität:
    * Extraktion von Epochen aus Raw-Daten basierend auf Annotationen
    * Labeling mit Block-Namen und n-back Schwierigkeitsgraden
    * Erstellung von MNE Epochs-Objekten mit Metadaten
    * 3D-Datenstruktur: (Epochen x Kanäle x Zeit)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import mne
from mne.io import Raw


@dataclass
class EpochingConfig:
    """Konfiguration für Epoching-Parameter.

    Attribute
    ---------
    tmin : float
        Start-Zeit relativ zum Event-Onset (in Sekunden, typisch 0.0)
    tmax : float
        End-Zeit relativ zum Event-Onset (in Sekunden, None = gesamte Annotation)
    baseline : tuple or None
        Baseline-Korrektur Zeitfenster (tmin, tmax) in Sekunden
    picks : list or None
        Kanäle zum Epoching (None = alle EEG-Kanäle)
    reject : dict or None
        Artifact rejection criteria (z.B. {'eeg': 100e-6} für 100µV)
    """

    tmin: float = 0.0
    tmax: Optional[float] = None  # None = use full annotation duration
    baseline: Optional[Tuple[float, float]] = None  # (None, 0) für pre-stimulus baseline
    picks: Optional[List[str]] = None
    reject: Optional[Dict[str, float]] = None


def create_epochs_from_raw(
        raw: Raw, config: EpochingConfig = None
) -> Optional[mne.Epochs]:
    """Erstellt Epochen aus annotiertem Raw-Objekt mit 4s Segmenten und 2s Overlap.

    Parameter
    ---------
    raw : mne.io.Raw
        Raw-Objekt mit n-back Annotationen
    config : EpochingConfig, optional
        Epoching-Konfiguration

    Rückgabe
    --------
    mne.Epochs or None
        Standard MNE Epochs-Objekt mit Metadaten für automatische Konvertierung
    """
    if config is None:
        config = EpochingConfig(tmin=0.0, tmax=4.0)

    if not raw.annotations or len(raw.annotations) == 0:
        print("[WARN] Keine Annotationen in Raw-Objekt gefunden")
        return None

    segment_length = 4.0
    overlap = 2.0

    all_events = []
    event_id = {
        'baseline': 0,
        '0-back': 1,
        '1-back': 2,
        '2-back': 3,
        '3-back': 4
    }

    for block_idx, annot in enumerate(raw.annotations):
        description = annot['description']

        start_time = annot['onset']
        duration = annot['duration']

        block_events = mne.make_fixed_length_events(
            raw,
            id=event_id[description],
            start=start_time,
            stop=start_time + duration,
            duration=segment_length,
            overlap=overlap
        )

        if len(block_events) == 0:
            print(f"    Keine Segmente für Block {block_idx} (zu kurz)")
            continue

        print(f"    → {len(block_events)} Segmente à {segment_length}s")
        all_events.append(block_events)

    if not all_events:
        print("[WARN] Keine gültigen Events erstellt")
        return None

    events = np.vstack(all_events)
    events = events[events[:, 0].argsort()]

    print(f"\n[INFO] Erstelle {len(events)} Epochen...")
    print(f"  - Event IDs: {event_id}")

    try:
        # Standard MNE Epochs erstellen
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=config.tmin,
            tmax=config.tmax or segment_length,
            baseline=config.baseline,
            picks=config.picks,
            reject=config.reject,
            preload=True,
            verbose=False
        )

        data_shape = epochs.get_data().shape
        print(f"[SUCCESS] {len(epochs)} Epochen erstellt")
        print(f"  - Form: {data_shape}")

        assert epochs is not None, "Epochs-Objekt ist None"
        assert len(epochs) > 0, "Keine Epochen erstellt"

        assert len(data_shape) == 3, f"Erwarte 3D-Daten, erhalten: {len(data_shape)}D"
        assert data_shape[0] > 0, "Keine Epochen in Datenmatrix"
        assert data_shape[1] > 0, "Keine Kanäle in Datenmatrix"
        assert data_shape[2] > 0, "Keine Zeitpunkte in Datenmatrix"

        assert hasattr(epochs, 'event_id'), "Event-ID Dictionary fehlt"
        assert len(epochs.event_id) > 0, "Event-ID Dictionary ist leer"
        assert epochs.info['sfreq'] > 0, "Ungültige Sampling-Rate"

        onset_times = epochs.events[:, 0] / epochs.info['sfreq']
        assert np.all(onset_times[:-1] <= onset_times[1:]), "Events nicht chronologisch sortiert"

        return epochs

    except Exception as e:
        print(f"[ERROR] Fehler beim Epoching: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_epochs_summary(epochs: mne.Epochs) -> pd.DataFrame:
    """Erstellt eine Zusammenfassung der Epochen-Verteilung.

    Parameter
    ---------
    epochs : mne.Epochs
        MNE Epochs-Objekt

    Rückgabe
    --------
    pd.DataFrame
        Zusammenfassung mit Event-Namen und Anzahl
    """
    summary_data = []

    for event_name, event_id in epochs.event_id.items():
        count = sum(epochs.events[:, 2] == event_id)
        summary_data.append({
            'event_name': event_name,
            'event_id': event_id,
            'count': count,
            'percentage': (count / len(epochs)) * 100
        })

    return pd.DataFrame(summary_data)


def epochs_to_dataframe(epochs: mne.Epochs) -> pd.DataFrame:
    """Konvertiert Epochen zu DataFrame für pandas/scikit-learn.

    Parameter
    ---------
    epochs : mne.Epochs
        MNE Epochs-Objekt

    Rückgabe
    --------
    pd.DataFrame
        DataFrame mit Features als Spalten und Labels
    """
    # Daten extrahieren und zu 2D umformen
    data = epochs.get_data()  # (n_epochs, n_channels, n_timepoints)
    n_epochs, n_channels, n_timepoints = data.shape

    # Zu 2D umformen: (n_epochs, n_features)
    data_2d = data.reshape(n_epochs, n_channels * n_timepoints)

    # Feature-Namen erstellen
    feature_names = []
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        for time_idx in range(n_timepoints):
            feature_names.append(f"{ch_name}_t{time_idx}")

    # DataFrame erstellen
    df = pd.DataFrame(data_2d, columns=feature_names)

    # Labels hinzufügen (reverse lookup von event_id)
    id_to_name = {v: k for k, v in epochs.event_id.items()}
    df['label'] = [id_to_name[event_id] for event_id in epochs.events[:, 2]]

    return df


def validate_epochs(epochs: mne.Epochs) -> None:
    """Validiert das Epoching-Ergebnis mit Asserts.
    
    Parameter
    ---------
    epochs : mne.Epochs
        Zu validierende Epochen
        
    Raises
    ------
    AssertionError
        Bei Validierungsfehlern
    """
    # Basis-Validierung
    assert epochs is not None, "Epochs-Objekt ist None"
    assert len(epochs) > 0, "Keine Epochen erstellt"
    
    data_shape = epochs.get_data().shape
    assert len(data_shape) == 3, f"Erwarte 3D-Daten, erhalten: {len(data_shape)}D"
    assert data_shape[0] > 0, "Keine Epochen in Datenmatrix"
    assert data_shape[1] > 0, "Keine Kanäle in Datenmatrix"
    assert data_shape[2] > 0, "Keine Zeitpunkte in Datenmatrix"
    
    assert hasattr(epochs, 'event_id'), "Event-ID Dictionary fehlt"
    assert len(epochs.event_id) > 0, "Event-ID Dictionary ist leer"
    assert epochs.info['sfreq'] > 0, "Ungültige Sampling-Rate"
    
    # Zeitliche Konsistenz
    onset_times = epochs.events[:, 0] / epochs.info['sfreq']
    assert np.all(onset_times[:-1] <= onset_times[1:]), "Events nicht chronologisch sortiert"
    
    # Abstände zwischen Events prüfen - flexiblere Validierung
    if len(onset_times) > 1:
        time_diffs = np.diff(onset_times)
        
        # Keine negativen Abstände
        assert np.all(time_diffs >= 0), "Negative Zeitabstände gefunden"
        
        # Prüfe Abstände innerhalb von Blöcken (sollten ~2s sein wegen Overlap)
        # und zwischen Blöcken (können größer sein)
        small_diffs = time_diffs[time_diffs <= 5.0]  # Innerhalb von Blöcken
        large_diffs = time_diffs[time_diffs > 5.0]   # Zwischen Blöcken
        
        if len(small_diffs) > 0:
            median_small = np.median(small_diffs)
            assert 1.0 <= median_small <= 3.0, f"Unerwartete Intra-Block-Abstände: {median_small:.2f}s (erwartet: ~2s)"
        
        # Überprüfe, dass große Abstände nicht zu extrem sind (max 5 Minuten zwischen Blöcken)
        if len(large_diffs) > 0:
            max_large = np.max(large_diffs)
            assert max_large <= 300, f"Zu große Inter-Block-Abstände: {max_large:.1f}s (max: 300s)"
            print(f"  Info: {len(large_diffs)} Inter-Block-Übergänge gefunden (max: {max_large:.1f}s)")
    
    print(f"✓ Epochen-Validierung erfolgreich: {len(epochs)} Epochen, Form: {data_shape}")
