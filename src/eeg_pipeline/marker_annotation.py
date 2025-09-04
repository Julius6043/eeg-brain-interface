"""Marker-based Annotation for EEG Data.

Dieses Modul extrahiert Block-Informationen aus Marker-Streams und konvertiert sie
in MNE-Annotationen, die in Raw-Objekte integriert werden können.

Funktionalität:
    * Identifikation von experiment blocks anhand von Marker-Patterns
    * Berechnung der n-back Schwierigkeit mittels Block_difficulty_extractor
    * Erstellung von MNE Annotations mit korrekten Zeitstempeln
    * Integration der Annotationen in Raw-Objekte

Zeitkonversion:
    * Marker-Timestamps sind in Sekunden (absolute Zeit)
    * EEG-Daten haben eine Sampling-Rate (Standard: 250 Hz)
    * Annotationen verwenden relative Zeit ab EEG-Start
"""

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import mne
from mne.io import Raw


def extract_nblock(sequence: List[str], targets: List[int], zero_flag: bool) -> int:
    """Extrahiert den n-back Grad für einen einzelnen Block.

    Adapted from Block_difficulty_extractor.py
    """
    for name, arg in {"sequence": sequence, "targets": targets}.items():
        if not isinstance(arg, list):
            raise TypeError(f"'{name}' is not a list (got: {type(arg).__name__})")

    # Special Handling of Block 0...
    if zero_flag:
        return 0

    n_vals = np.zeros(4)
    for t in targets:
        if t >= len(sequence):
            continue
        target_letter = sequence[t]
        if t >= 1 and target_letter == sequence[t - 1]:
            n_vals[1] += 1
        if t >= 2 and target_letter == sequence[t - 2]:
            n_vals[2] += 1
        if t >= 3 and target_letter == sequence[t - 3]:
            n_vals[3] += 1
    return int(np.argmax(n_vals))


def calculate_nvals(df: pd.DataFrame) -> List[int]:
    """Berechnet n-back Werte für alle Blöcke.

    Adapted from Block_difficulty_extractor.py
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas.DataFrame, but got: {type(df).__name__}")

    df = df.copy()
    df["prev_marker"] = df["marker"].shift(1)
    df["prev_prev_marker"] = df["marker"].shift(2)

    # Sequences - suche nach sequence_ Markern die nach einem main_block_X_start kommen
    mask_seq = df["marker"].str.startswith("sequence") & df["prev_marker"].str.contains(
        "main_block.*start", na=False
    )

    seq_df = (
        df.loc[mask_seq, "marker"]
        .str.removeprefix("sequence_")
        .str.split(",")
        .to_frame(name="sequence")
        .reset_index(drop=True)
    )

    # Targets - suche nach targets_ Markern die nach einem sequence_ Marker kommen
    mask_trg = df["marker"].str.startswith("targets") & df[
        "prev_marker"
    ].str.startswith("sequence")

    trg_df = (
        df.loc[mask_trg, "marker"]
        .str.removeprefix("targets_")
        .str.split(",")
        .apply(
            lambda x: [int(i) for i in x] if len(x) > 0 and x[0] != "" else []
        )  # Handle empty targets
        .to_frame(name="targets")
        .reset_index(drop=True)
    )

    n_vals = []
    min_len = min(len(seq_df), len(trg_df))

    for idx in range(min_len):
        seq = seq_df.at[idx, "sequence"]
        trg = trg_df.at[idx, "targets"]
        if idx == 0:
            n_vals.append(extract_nblock(seq, trg, True))
        else:
            n_vals.append(extract_nblock(seq, trg, False))

    return n_vals


def extract_baseline_info(markers_df: pd.DataFrame) -> List[Tuple[float, float, str]]:
    """Extrahiert Baseline-Informationen aus Marker-DataFrame.

    Parameter
    ---------
    markers_df : pd.DataFrame
        DataFrame mit Spalten 'Timestamp' und 'Marker1' (Marker-Text)

    Rückgabe
    --------
    List[Tuple[float, float, str]]
        Liste von (start_time, end_time, description) Tupeln für Baseline-Perioden
    """
    if markers_df is None or markers_df.empty:
        return []

    # Arbeite mit Kopie der Daten
    markers_work = markers_df.copy()
    if "Marker1" in markers_work.columns:
        markers_work = markers_work.rename(columns={"Marker1": "marker"})
    elif "marker" not in markers_work.columns:
        return []

    baseline_info = []

    # Finde baseline_start und baseline_end Paare
    baseline_starts = markers_work[markers_work["marker"] == "baseline_start"]
    baseline_ends = markers_work[markers_work["marker"] == "baseline_end"]

    # Paare die starts und ends
    for i, (_, start_row) in enumerate(baseline_starts.iterrows()):
        start_time = start_row["Timestamp"]

        # Finde das nächste baseline_end nach diesem start
        subsequent_ends = baseline_ends[baseline_ends["Timestamp"] > start_time]
        if not subsequent_ends.empty:
            end_time = subsequent_ends.iloc[0]["Timestamp"]
            baseline_info.append((start_time, end_time, f"baseline_{i+1}"))
        else:
            # Falls kein End gefunden wird, nimm 120s als Standard-Baseline-Dauer
            end_time = start_time + 120.0
            baseline_info.append((start_time, end_time, f"baseline_{i+1}"))

    return baseline_info


def extract_block_info(markers_df: pd.DataFrame) -> List[Tuple[float, float, int, int]]:
    """Extrahiert Block-Informationen aus Marker-DataFrame.

    Parameter
    ---------
    markers_df : pd.DataFrame
        DataFrame mit Spalten 'Timestamp' und 'Marker1' (Marker-Text)

    Rückgabe
    --------
    List[Tuple[float, float, int, int]]
        Liste von (start_time, end_time, block_number, n_back_level) Tupeln
    """
    if markers_df is None or markers_df.empty:
        return []

    # Rename für Kompatibilität mit Block_difficulty_extractor
    markers_work = markers_df.copy()
    if "Marker1" in markers_work.columns:
        markers_work = markers_work.rename(columns={"Marker1": "marker"})
    elif "marker" not in markers_work.columns:
        print("[WARN] Keine 'marker' oder 'Marker1' Spalte gefunden")
        return []

    # Finde main_block_X_start Marker
    block_starts = markers_work[
        markers_work["marker"].str.contains("main_block.*start", na=False)
    ]

    if block_starts.empty:
        print("[WARN] Keine main_block_X_start Marker gefunden")
        return []

    # Berechne n-back Werte
    try:
        n_vals = calculate_nvals(markers_work)
    except Exception as e:
        print(f"[WARN] Fehler bei n-back Berechnung: {e}")
        import traceback

        traceback.print_exc()
        n_vals = [0] * len(block_starts)

    block_info = []

    for idx, (_, row) in enumerate(block_starts.iterrows()):
        start_time = row["Timestamp"]

        # Finde Ende des Blocks (nächster main_block start oder Ende der Daten)
        if idx + 1 < len(block_starts):
            next_start = block_starts.iloc[idx + 1]["Timestamp"]
            end_time = next_start
        else:
            # Letzter Block - verwende letzten Timestamp oder schätze
            end_time = markers_work["Timestamp"].max()
            if end_time == start_time:
                end_time = start_time + 60.0  # 60s default duration

        # Block-Nummer aus Marker extrahieren
        marker_text = row["marker"]
        try:
            # Extract block number from "main_block_X_start"
            import re

            match = re.search(r"main_block_(\d+)_start", marker_text)
            if match:
                block_num = int(match.group(1))
            else:
                block_num = idx
        except (ValueError, IndexError):
            block_num = idx

        # n-back Level zuweisen
        n_back = n_vals[idx] if idx < len(n_vals) else 0

        block_info.append((start_time, end_time, block_num, n_back))

    return block_info


def create_annotations_from_blocks_and_baseline(
    block_info: List[Tuple[float, float, int, int]],
    baseline_info: List[Tuple[float, float, str]],
    eeg_start_time: float,
    sampling_rate: float = None,
) -> mne.Annotations:
    """Erstellt MNE Annotations aus Block- und Baseline-Informationen.

    Parameter
    ---------
    block_info : List[Tuple[float, float, int, int]]
        Liste von (start_time, end_time, block_number, n_back_level) Tupeln
    baseline_info : List[Tuple[float, float, str]]
        Liste von (start_time, end_time, description) Tupeln für Baselines
    eeg_start_time : float
        Start-Zeitstempel der EEG-Aufnahme (für relative Zeitberechnung)
    sampling_rate : float, optional
        Sampling-Rate der EEG-Daten in Hz (nicht mehr verwendet)

    Rückgabe
    --------
    mne.Annotations
        Annotations-Objekt für MNE Raw
    """
    if not block_info and not baseline_info:
        return mne.Annotations(onset=[], duration=[], description=[])

    onsets = []
    durations = []
    descriptions = []

    # Füge Baseline-Annotationen hinzu
    for start_time, end_time, description in baseline_info:
        # Konvertiere zu relativer Zeit (EEG-Start = 0)
        onset = start_time - eeg_start_time
        duration = end_time - start_time

        # Einfache Beschreibung nur mit Zeitinformationen
        desc = f"{description}_onset_{onset:.1f}s_dur_{duration:.1f}s"

        onsets.append(onset)
        durations.append(duration)
        descriptions.append(desc)

    # Füge Block-Annotationen hinzu
    for start_time, end_time, block_num, n_back in block_info:
        # Konvertiere zu relativer Zeit (EEG-Start = 0)
        onset = start_time - eeg_start_time
        duration = end_time - start_time

        # Einfache Beschreibung nur mit Zeitinformationen
        desc = f"block_{block_num:02d}_nback_{n_back}_onset_{onset:.1f}s_dur_{duration:.1f}s"

        onsets.append(onset)
        durations.append(duration)
        descriptions.append(desc)

    return mne.Annotations(
        onset=onsets, duration=durations, description=descriptions, orig_time=None
    )


def annotate_raw_with_markers(raw: Raw, markers_df: Optional[pd.DataFrame]) -> Raw:
    """Fügt Marker-basierte Annotationen zu einem Raw-Objekt hinzu.

    Parameter
    ---------
    raw : mne.io.Raw
        EEG Raw-Objekt
    markers_df : pd.DataFrame or None
        DataFrame mit Marker-Informationen

    Rückgabe
    --------
    mne.io.Raw
        Raw-Objekt mit hinzugefügten Annotationen
    """
    if markers_df is None or markers_df.empty:
        print("[INFO] Keine Marker-Daten verfügbar")
        return raw

    # Extrahiere Block-Informationen
    block_info = extract_block_info(markers_df)

    # Extrahiere Baseline-Informationen
    baseline_info = extract_baseline_info(markers_df)

    if not block_info and not baseline_info:
        print("[INFO] Keine Block- oder Baseline-Informationen extrahiert")
        return raw

    # Schätze EEG-Start-Zeit aus erstem Marker-Timestamp
    # EEG startet vor dem ersten Marker
    eeg_start_time = raw.times[0]

    # Erstelle Annotationen (ohne Sampling-Rate)
    annotations = create_annotations_from_blocks_and_baseline(
        block_info, baseline_info, eeg_start_time
    )

    # Füge zu Raw hinzu
    raw.set_annotations(annotations)

    print(f"[INFO] {len(annotations)} Annotationen hinzugefügt:")
    print(f"  - {len(baseline_info)} Baseline-Perioden")
    print(f"  - {len(block_info)} Experimentelle Blöcke")

    # Zeige erste paar Annotationen als Beispiel
    for i in range(min(5, len(annotations))):
        desc = annotations.description[i]
        onset = annotations.onset[i]
        duration = annotations.duration[i]
        print(f"  {desc}")

    if len(annotations) > 5:
        print(f"  ... und {len(annotations) - 5} weitere")

    return raw
