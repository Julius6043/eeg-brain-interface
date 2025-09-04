"""EEG Annotation Module for N-Back Block Annotation.

Dieses Modul erweitert verarbeitete EEG-Daten um Block-Annotationen basierend auf
Marker-CSV-Dateien. Es identifiziert N-Back-Blöcke und deren Schwierigkeitsgrad
und fügt diese als MNE-Annotationen zu den Raw-Objekten hinzu.

Hauptfunktionen:
    * Parsing von Marker-CSV-Dateien zur Block-Identifikation
    * Schwierigkeitsbestimmung über Block_difficulty_extractor
    * Automatische Annotation von FIF-Dateien mit Block-Informationen

Designziele:
    * Robuste CSV-Verarbeitung mit Fallback-Strategien
    * Integration in bestehende Pipeline-Struktur
    * Kompatibilität mit MNE-Annotations-Format
"""

import re
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Callable

import numpy as np
import pandas as pd
import mne
from mne.io import Raw


@dataclass
class AnnotationConfig:
    """Konfiguration für EEG-Annotation.

    Attribute
    ---------
    use_difficulty_mapping:
        Aktiviert die Schwierigkeitsbestimmung über Block_difficulty_extractor.py
    difficulty_extractor_path:
        Pfad zum Block_difficulty_extractor.py Skript
    annotation_prefix:
        Prefix für Annotation-Beschreibungen (default: "nback")
    time_offset_s:
        Globaler Zeitoffset in Sekunden für alle Annotationen
    """

    use_difficulty_mapping: bool = True
    difficulty_extractor_path: Optional[Path] = None
    annotation_prefix: str = "nback"
    time_offset_s: float = 0.0


def load_difficulty_extractor(
    extractor_path: Optional[Path],
) -> Optional[Callable[[pd.DataFrame], List[int]]]:
    """Lade calculate_nvals Funktion aus Block_difficulty_extractor.py.

    Parameter
    ---------
    extractor_path:
        Pfad zur Extractor-Datei oder None für automatische Suche

    Rückgabe
    --------
    calculate_nvals:
        Funktion zur Schwierigkeitsberechnung oder None bei Fehlern
    """
    if extractor_path is None:
        # Automatische Suche im scripts-Verzeichnis
        potential_paths = [
            Path("scripts/Block_difficulty_extractor.py"),
            Path("../scripts/Block_difficulty_extractor.py"),
            Path("../../scripts/Block_difficulty_extractor.py"),
        ]
        extractor_path = next((p for p in potential_paths if p.exists()), None)

    if extractor_path is None or not extractor_path.exists():
        print(f"[WARN] Block_difficulty_extractor.py nicht gefunden: {extractor_path}")
        return None

    try:
        spec = importlib.util.spec_from_file_location(
            "difficulty_extractor", str(extractor_path)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        calculate_nvals = getattr(module, "calculate_nvals", None)
        if calculate_nvals is None or not callable(calculate_nvals):
            print("[WARN] calculate_nvals Funktion nicht gefunden oder nicht aufrufbar")
            return None

        print(f"✓ Difficulty extractor geladen: {extractor_path}")
        return calculate_nvals

    except Exception as e:
        print(f"[WARN] Fehler beim Laden des Difficulty Extractors: {e}")
        return None


def parse_blocks_from_marker_csv(df_markers: pd.DataFrame) -> pd.DataFrame:
    """Extrahiere Block-Intervalle aus Marker-CSV.

    Sucht nach 'main_block_<i>_start' Markern und bestimmt Block-Enden
    über den letzten 'trial_*_end' Marker vor dem nächsten Block.

    Parameter
    ---------
    df_markers:
        DataFrame mit 'marker' und optional 'timestamp' Spalten

    Rückgabe
    --------
    blocks_df:
        DataFrame mit Spalten: block_num, onset_s, duration_s
    """
    if "marker" not in df_markers.columns:
        raise ValueError("Marker CSV benötigt eine 'marker' Spalte")

    # Zeitbasis bestimmen (Sekunden relativ zum ersten Timestamp)
    if "timestamp" in df_markers.columns and df_markers["timestamp"].notnull().all():
        t0 = df_markers["timestamp"].iloc[0]
        times_sec = df_markers["timestamp"] - t0
        use_timestamps = True
    else:
        use_timestamps = False
        times_sec = None
        print("[WARN] Keine gültigen Timestamps gefunden, verwende Fallback-Zeitschema")

    # Pattern für Block-Start und Trial-Ende
    pat_block_start = re.compile(r"^main_block_(\d+)_start$")
    pat_trial_end = re.compile(r"^trial_(\d+)_end$")

    # Block-Starts sammeln
    block_starts = []
    for idx, marker in df_markers["marker"].items():
        match = pat_block_start.match(str(marker))
        if match:
            block_starts.append((idx, int(match.group(1))))

    if not block_starts:
        raise RuntimeError("Keine 'main_block_<i>_start' Marker gefunden")

    print(f"✓ {len(block_starts)} Blöcke gefunden")

    # Block-Bereiche bestimmen
    blocks: List[Dict] = []
    for i, (start_row_idx, block_num) in enumerate(block_starts):
        # Ende-Limit für diesen Block
        end_limit = (
            block_starts[i + 1][0] - 1
            if (i + 1) < len(block_starts)
            else len(df_markers) - 1
        )

        # Letzten trial_end vor dem nächsten Block finden
        last_trial_end_idx = None
        for j in range(start_row_idx, end_limit + 1):
            if pat_trial_end.match(str(df_markers["marker"].iloc[j])):
                last_trial_end_idx = j

        end_row_idx = (
            last_trial_end_idx if last_trial_end_idx is not None else end_limit
        )

        # Zeiten berechnen
        if use_timestamps:
            onset_s = float(times_sec.iloc[start_row_idx])
            end_s = float(times_sec.iloc[end_row_idx])
        else:
            # Fallback: Annahme von 60s pro Block
            onset_s = float(len(blocks)) * 60.0
            end_s = onset_s + 60.0

        duration_s = max(0.0, end_s - onset_s)
        blocks.append(
            {"block_num": block_num, "onset_s": onset_s, "duration_s": duration_s}
        )

    return pd.DataFrame(blocks).sort_values("onset_s").reset_index(drop=True)


def attach_difficulty_mapping(
    blocks_df: pd.DataFrame,
    calculate_nvals: Callable[[pd.DataFrame], List[int]],
    marker_df: pd.DataFrame,
) -> pd.DataFrame:
    """Füge Schwierigkeitsgrade zu Block-DataFrame hinzu.

    Parameter
    ---------
    blocks_df:
        DataFrame mit Block-Informationen
    calculate_nvals:
        Funktion zur Schwierigkeitsberechnung
    marker_df:
        Original Marker-DataFrame für die Berechnung

    Rückgabe
    --------
    blocks_df:
        Erweitert um 'difficulty' Spalte
    """
    try:
        # DataFrame für calculate_nvals vorbereiten
        df_for_calc = pd.DataFrame({"marker": marker_df["marker"].astype(str)})
        nvals = calculate_nvals(df_for_calc)

        if len(nvals) != len(blocks_df):
            print(
                f"[WARN] Anzahl n-values ({len(nvals)}) != Anzahl Blöcke ({len(blocks_df)})"
            )

        # Schwierigkeitsgrade zuweisen
        blocks_df = blocks_df.copy()
        blocks_df["difficulty"] = [
            int(nvals[i]) if i < len(nvals) else None for i in range(len(blocks_df))
        ]

        print(f"✓ Schwierigkeitsgrade zugewiesen: {blocks_df['difficulty'].tolist()}")
        return blocks_df

    except Exception as e:
        print(f"[WARN] Fehler bei Schwierigkeitsberechnung: {e}")
        blocks_df = blocks_df.copy()
        blocks_df["difficulty"] = None
        return blocks_df


def create_mne_annotations(
    blocks_df: pd.DataFrame, config: AnnotationConfig
) -> mne.Annotations:
    """Erstelle MNE-Annotations aus Block-DataFrame.

    Parameter
    ---------
    blocks_df:
        DataFrame mit Block-Informationen und optional Schwierigkeitsgraden
    config:
        Annotation-Konfiguration

    Rückgabe
    --------
    annotations:
        MNE Annotations-Objekt
    """
    descriptions = []
    for _, row in blocks_df.iterrows():
        # Schwierigkeitsgrad bestimmen
        difficulty = "unknown"
        if "difficulty" in row and not pd.isna(row["difficulty"]):
            difficulty = str(int(row["difficulty"]))

        # Block-Nummer formatieren
        block_num = int(row["block_num"]) if not pd.isna(row["block_num"]) else -1

        # Beschreibung erstellen
        desc = f"{config.annotation_prefix}/{difficulty}|block:B{block_num:02d}"
        descriptions.append(desc)

    # Zeitoffset anwenden
    onsets = blocks_df["onset_s"] + config.time_offset_s

    annotations = mne.Annotations(
        onset=onsets.tolist(),
        duration=blocks_df["duration_s"].tolist(),
        description=descriptions,
    )

    return annotations


def annotate_raw_with_blocks(
    raw: Raw, markers_df: Optional[pd.DataFrame], config: AnnotationConfig
) -> Raw:
    """Annotiere ein Raw-Objekt mit N-Back-Block-Informationen.

    Parameter
    ---------
    raw:
        MNE Raw-Objekt zum Annotieren
    markers_df:
        DataFrame mit Marker-Informationen oder None
    config:
        Annotation-Konfiguration

    Rückgabe
    --------
    raw:
        Annotiertes Raw-Objekt (in-place Modifikation)
    """
    if markers_df is None:
        print("[WARN] Keine Marker verfügbar für Annotation")
        return raw

    try:
        # 1. Block-Bereiche aus Marker-CSV extrahieren
        blocks_df = parse_blocks_from_marker_csv(markers_df)

        # 2. Schwierigkeitsgrade hinzufügen (falls aktiviert)
        if config.use_difficulty_mapping:
            calculate_nvals = load_difficulty_extractor(
                config.difficulty_extractor_path
            )
            if calculate_nvals is not None:
                blocks_df = attach_difficulty_mapping(
                    blocks_df, calculate_nvals, markers_df
                )

        # 3. Validierung der Block-Zeiten gegen Raw-Dauer
        recording_duration = raw.n_times / raw.info["sfreq"]
        block_ends = blocks_df["onset_s"] + blocks_df["duration_s"]
        overflow_mask = block_ends > recording_duration

        if overflow_mask.any():
            n_overflow = overflow_mask.sum()
            print(
                f"[WARN] {n_overflow} Block(s) überschreiten die Aufnahmedauer und werden angepasst"
            )
            # Clip Blöcke auf Aufnahmedauer
            blocks_df.loc[overflow_mask, "duration_s"] = (
                recording_duration - blocks_df.loc[overflow_mask, "onset_s"]
            )

        # 4. MNE-Annotations erstellen
        new_annotations = create_mne_annotations(blocks_df, config)

        # 5. Zu Raw hinzufügen
        if raw.annotations is not None and len(raw.annotations) > 0:
            raw.set_annotations(raw.annotations + new_annotations)
        else:
            raw.set_annotations(new_annotations)

        print(f"✓ {len(new_annotations)} Block-Annotationen hinzugefügt")

        # 6. Event-IDs ableiten (zur Information)
        try:
            events, event_id = mne.events_from_annotations(raw)
            n_back_events = {k: v for k, v in event_id.items() if "nback" in k}
            if n_back_events:
                print(f"✓ Event-IDs erstellt: {list(n_back_events.keys())}")
        except Exception as e:
            print(f"[WARN] Event-Ableitung fehlgeschlagen: {e}")

    except Exception as e:
        print(f"[ERROR] Annotation fehlgeschlagen: {e}")

    return raw


def auto_find_difficulty_extractor() -> Optional[Path]:
    """Automatische Suche nach Block_difficulty_extractor.py.

    Sucht in gängigen Verzeichnissen relativ zum aktuellen Arbeitsverzeichnis.
    """
    search_paths = [
        Path("scripts/Block_difficulty_extractor.py"),
        Path("../scripts/Block_difficulty_extractor.py"),
        Path("../../scripts/Block_difficulty_extractor.py"),
        Path("./Block_difficulty_extractor.py"),
    ]

    for path in search_paths:
        if path.exists():
            return path.resolve()

    return None
