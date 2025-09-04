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
from typing import List, Dict, Optional, Tuple
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
    baseline: Optional[Tuple[float, float]] = (
        None  # (None, 0) für pre-stimulus baseline
    )
    picks: Optional[List[str]] = None
    reject: Optional[Dict[str, float]] = None


def parse_annotation_info(description: str) -> Dict[str, str]:
    """Extrahiert Informationen aus Annotation-Beschreibungen.

    Parameter
    ---------
    description : str
        Annotation-Beschreibung (z.B. "block_04_nback_3_onset_1091.2s_dur_216.1s")

    Rückgabe
    --------
    dict
        Dictionary mit extrahierten Informationen:
        - 'type': 'baseline' oder 'block'
        - 'block_name': 'Baseline' oder 'Block0', 'Block1', etc.
        - 'difficulty': 'base' oder 'n-back 0', 'n-back 1', etc.
        - 'block_number': Numerischer Block-Index (None für Baseline)
        - 'nback_level': N-back Level (None für Baseline)
    """
    info = {
        "type": None,
        "block_name": None,
        "difficulty": None,
        "block_number": None,
        "nback_level": None,
    }

    if description.startswith("baseline"):
        info["type"] = "baseline"
        info["block_name"] = "Baseline"
        info["difficulty"] = "base"

        # Extrahiere Baseline-Nummer falls vorhanden
        parts = description.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            info["block_number"] = int(parts[1])
            info["block_name"] = f"Baseline{parts[1]}"

    elif description.startswith("block"):
        info["type"] = "block"

        # Parse "block_04_nback_3_onset_1091.2s_dur_216.1s"
        parts = description.split("_")

        # Block number
        if len(parts) > 1:
            try:
                block_num = int(parts[1])
                info["block_number"] = block_num
                info["block_name"] = f"Block{block_num}"
            except ValueError:
                info["block_name"] = "BlockUnknown"

        # N-back level
        if "nback" in parts:
            nback_idx = parts.index("nback")
            if nback_idx + 1 < len(parts):
                try:
                    nback_level = int(parts[nback_idx + 1])
                    info["nback_level"] = nback_level
                    info["difficulty"] = f"n-back {nback_level}"
                except ValueError:
                    info["difficulty"] = "n-back unknown"

    return info


def create_epochs_from_raw(
    raw: Raw, config: EpochingConfig = None
) -> Optional[mne.Epochs]:
    """Erstellt Epochen aus annotiertem Raw-Objekt.

    Parameter
    ---------
    raw : mne.io.Raw
        Raw-Objekt mit Annotationen
    config : EpochingConfig, optional
        Epoching-Konfiguration

    Rückgabe
    --------
    mne.Epochs or None
        Epochs-Objekt mit allen Blöcken und Baselines als separate Epochen
    """
    if config is None:
        config = EpochingConfig()

    if not raw.annotations or len(raw.annotations) == 0:
        print("[WARN] Keine Annotationen in Raw-Objekt gefunden")
        return None

    # Erstelle Events aus Annotationen
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    if len(events) == 0:
        print("[WARN] Keine Events aus Annotationen erstellt")
        return None

    # Bestimme tmax basierend auf Konfiguration oder Annotation-Dauer
    if config.tmax is None:
        # Verwende die Dauer der Annotationen als tmax
        durations = raw.annotations.duration
        max_duration = np.max(durations)
        tmax = max_duration
        print(f"[INFO] Verwende maximale Annotation-Dauer als tmax: {tmax:.1f}s")
    else:
        tmax = config.tmax

    # Erstelle Metadaten VOR dem Epoching
    metadata = create_epochs_metadata(raw.annotations)

    # Erstelle Epochen
    try:
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=config.tmin,
            tmax=tmax,
            baseline=config.baseline,
            picks=config.picks,
            reject=config.reject,
            preload=True,
            verbose=False,
        )

        # Prüfe ob die Anzahl der Epochen mit den Metadaten übereinstimmt
        if len(epochs) != len(metadata):
            print(
                f"[WARN] Anzahl Epochen ({len(epochs)}) != Anzahl Metadaten ({len(metadata)})"
            )
            # Schneide Metadaten auf Epochen-Anzahl zu
            metadata = metadata.iloc[: len(epochs)].copy()

        # Füge Metadaten hinzu
        epochs.metadata = metadata

        print(f"[INFO] {len(epochs)} Epochen erstellt")
        print(f"  - Epochen-Form: {epochs.get_data().shape}")
        print(f"  - Baselines: {sum(metadata['type'] == 'baseline')}")
        print(f"  - Experimentelle Blöcke: {sum(metadata['type'] == 'block')}")

        return epochs

    except Exception as e:
        print(f"[ERROR] Fehler beim Epoching: {e}")
        import traceback

        traceback.print_exc()
        return None


def create_epochs_metadata(annotations: mne.Annotations) -> pd.DataFrame:
    """Erstellt Metadaten-DataFrame für Epochen.

    Parameter
    ---------
    annotations : mne.Annotations
        Annotationen mit Block- und Baseline-Informationen

    Rückgabe
    --------
    pd.DataFrame
        Metadaten mit Spalten: type, block_name, difficulty, block_number, nback_level
    """
    metadata_list = []

    for i, description in enumerate(annotations.description):
        info = parse_annotation_info(description)

        # Erweitere um zusätzliche Epochen-Informationen
        info.update(
            {
                "epoch_id": i,
                "description": description,
                "onset": annotations.onset[i],
                "duration": annotations.duration[i],
            }
        )

        metadata_list.append(info)

    metadata = pd.DataFrame(metadata_list)

    # Stelle sicher, dass alle erwarteten Spalten existieren
    expected_columns = [
        "type",
        "block_name",
        "difficulty",
        "block_number",
        "nback_level",
    ]
    for col in expected_columns:
        if col not in metadata.columns:
            metadata[col] = None

    return metadata


def extract_epochs_by_type(epochs: mne.Epochs, epoch_type: str) -> mne.Epochs:
    """Extrahiert Epochen nach Typ (baseline oder block).

    Parameter
    ---------
    epochs : mne.Epochs
        Epochen-Objekt mit Metadaten
    epoch_type : str
        Typ der zu extrahierenden Epochen ('baseline' oder 'block')

    Rückgabe
    --------
    mne.Epochs
        Gefilterte Epochen
    """
    if epochs.metadata is None:
        raise ValueError("Epochen haben keine Metadaten")

    mask = epochs.metadata["type"] == epoch_type
    return epochs[mask]


def extract_epochs_by_nback(epochs: mne.Epochs, nback_level: int) -> mne.Epochs:
    """Extrahiert Epochen nach n-back Level.

    Parameter
    ---------
    epochs : mne.Epochs
        Epochen-Objekt mit Metadaten
    nback_level : int
        N-back Level (0, 1, 2, 3)

    Rückgabe
    --------
    mne.Epochs
        Gefilterte Epochen
    """
    if epochs.metadata is None:
        raise ValueError("Epochen haben keine Metadaten")

    mask = epochs.metadata["nback_level"] == nback_level
    return epochs[mask]


def extract_epochs_by_block_name(epochs: mne.Epochs, block_name: str) -> mne.Epochs:
    """Extrahiert Epochen nach Block-Namen.

    Parameter
    ---------
    epochs : mne.Epochs
        Epochen-Objekt mit Metadaten
    block_name : str
        Block-Name (z.B. 'Baseline', 'Block0', 'Block1')

    Rückgabe
    --------
    mne.Epochs
        Gefilterte Epochen
    """
    if epochs.metadata is None:
        raise ValueError("Epochen haben keine Metadaten")

    mask = epochs.metadata["block_name"] == block_name
    return epochs[mask]


def get_epochs_summary(epochs: mne.Epochs) -> pd.DataFrame:
    """Erstellt eine Zusammenfassung der Epochen.

    Parameter
    ---------
    epochs : mne.Epochs
        Epochen-Objekt mit Metadaten

    Rückgabe
    --------
    pd.DataFrame
        Zusammenfassung mit Anzahl Epochen pro Kategorie
    """
    if epochs.metadata is None:
        return pd.DataFrame()

    # Gruppiere nach relevanten Kategorien
    summary = (
        epochs.metadata.groupby(["type", "difficulty"]).size().reset_index(name="count")
    )
    summary = summary.sort_values(["type", "difficulty"])

    return summary


if __name__ == "__main__":
    # Test-Beispiel
    print("Epoching-Modul Test")

    # Teste Annotation-Parsing
    test_descriptions = [
        "baseline_1_onset_0.0s_dur_121.0s",
        "block_00_nback_0_onset_129.5s_dur_310.7s",
        "block_04_nback_3_onset_1091.2s_dur_216.1s",
    ]

    print("\nTest Annotation-Parsing:")
    for desc in test_descriptions:
        info = parse_annotation_info(desc)
        print(f"{desc}")
        print(f"  -> {info}")
