"""Marker-based Annotation for EEG Data.

This module extracts block information from marker streams and converts it
into MNE annotations that can be integrated into Raw objects.

Functionality:
    * Identification of experiment blocks based on marker patterns
    * Calculation of n-back difficulty using Block_difficulty_extractor
    * Creation of MNE Annotations with correct timestamps
    * Integration of annotations into Raw objects

Time Conversion:
    * Marker timestamps are in seconds (absolute time)
    * EEG data has a sampling rate (default: 250 Hz)
    * Annotations use relative time from the start of the EEG
"""

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import mne
from mne.io import Raw


def extract_nblock(sequence: List[str], targets: List[int], zero_flag: bool) -> int:
    """Extracts the n-back degree for a single block.
    """
    for name, arg in {"sequence": sequence, "targets": targets}.items():
        if not isinstance(arg, list):
            raise TypeError(f"'{name}' is not a list (got: {type(arg).__name__})")

    # Special handling of Block 0...
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
    """Calculates n-back values for all blocks.
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas.DataFrame, but got: {type(df).__name__}")

    df = df.copy()
    df["prev_marker"] = df["marker"].shift(1)
    df["prev_prev_marker"] = df["marker"].shift(2)

    # Sequences - search for sequence_ markers that come after a main_block_X_start
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

    # Targets - search for targets_ markers that come after a sequence_ marker
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
    """Extracts baseline information from a marker DataFrame.

    Parameters
    ----------
    markers_df : pd.DataFrame
        DataFrame with columns 'Timestamp' and 'Marker1' (marker text)

    Returns
    -------
    List[Tuple[float, float, str]]
        List of (start_time, end_time, description) tuples for baseline periods
    """
    if markers_df is None or markers_df.empty:
        return []

    # Work with a copy of the data
    markers_work = markers_df.copy()
    if "Marker1" in markers_work.columns:
        markers_work = markers_work.rename(columns={"Marker1": "marker"})
    elif "marker" not in markers_work.columns:
        return []

    baseline_info = []

    # Find baseline_start and baseline_end pairs
    baseline_starts = markers_work[markers_work["marker"] == "baseline_start"]
    baseline_ends = markers_work[markers_work["marker"] == "baseline_end"]

    # Pair the starts and ends
    for i, (_, start_row) in enumerate(baseline_starts.iterrows()):
        start_time = start_row["Timestamp"]

        # Find the next baseline_end after this start
        subsequent_ends = baseline_ends[baseline_ends["Timestamp"] > start_time]
        if not subsequent_ends.empty:
            end_time = subsequent_ends.iloc[0]["Timestamp"]
            baseline_info.append((start_time, end_time, f"baseline_{i+1}"))
        else:
            # If no end is found, use 120s as the default baseline duration
            end_time = start_time + 120.0
            baseline_info.append((start_time, end_time, f"baseline_{i+1}"))

    return baseline_info


def extract_block_info(markers_df: pd.DataFrame) -> List[Tuple[float, float, int, int]]:
    """Extracts block information from a marker DataFrame.
    """
    if markers_df is None or markers_df.empty:
        return []

    # Rename for compatibility with Block_difficulty_extractor
    markers_work = markers_df.copy()
    if "Marker1" in markers_work.columns:
        markers_work = markers_work.rename(columns={"Marker1": "marker"})
    elif "marker" not in markers_work.columns:
        print("[WARN] No 'marker' or 'Marker1' column found")
        return []

    # Find main_block_X_start markers
    block_starts = markers_work[
        markers_work["marker"].str.contains("main_block.*start", na=False)
    ]

    if block_starts.empty:
        print("[WARN] No main_block_X_start markers found")
        return []

    # Calculate n-back values
    try:
        n_vals = calculate_nvals(markers_work)
    except Exception as e:
        print(f"[WARN] Error during n-back calculation: {e}")
        import traceback

        traceback.print_exc()
        n_vals = [0] * len(block_starts)

    block_info = []

    for idx, (_, row) in enumerate(block_starts.iterrows()):
        start_time = row["Timestamp"]

        # Find the end of the block (next main_block start or end of data)
        if idx + 1 < len(block_starts):
            next_start = block_starts.iloc[idx + 1]["Timestamp"]
            end_time = next_start
        else:
            # Last block - use the last timestamp or estimate
            end_time = markers_work["Timestamp"].max()
            if end_time == start_time:
                end_time = start_time + 60.0  # 60s default duration

        # Extract block number from marker
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

        # Assign n-back level
        n_back = n_vals[idx] if idx < len(n_vals) else 0

        block_info.append((start_time, end_time, block_num, n_back))

    return block_info


def create_annotations_from_blocks_and_baseline(
    block_info: List[Tuple[float, float, int, int]],
    baseline_info: List[Tuple[float, float, str]],
    eeg_start_time: float,
    sampling_rate: float = None,
) -> mne.Annotations:
    """Creates MNE Annotations from block and baseline information.
    """
    if not block_info and not baseline_info:
        return mne.Annotations(onset=[], duration=[], description=[])

    onsets = []
    durations = []
    descriptions = []

    # Add baseline annotations
    for start_time, end_time, description in baseline_info:
        # Convert to relative time (EEG start = 0)
        onset = start_time - eeg_start_time
        duration = end_time - start_time

        # Simple description
        desc = f"baseline"

        onsets.append(onset)
        durations.append(duration)
        descriptions.append(desc)

    # Add block annotations
    for start_time, end_time, block_num, n_back in block_info:
        # Convert to relative time (EEG start = 0)
        onset = start_time - eeg_start_time
        duration = end_time - start_time

        # Simple description
        desc = f"{n_back}-back"

        onsets.append(onset)
        durations.append(duration)
        descriptions.append(desc)

    return mne.Annotations(
        onset=onsets, duration=durations, description=descriptions, orig_time=None
    )


def annotate_raw_with_markers(raw: Raw, markers_df: Optional[pd.DataFrame]) -> Raw:
    """Adds marker-based annotations to a Raw object.
    """
    if markers_df is None or markers_df.empty:
        print("[INFO] No marker data available")
        return raw

    # Extract block information
    block_info = extract_block_info(markers_df)

    # Extract baseline information
    baseline_info = extract_baseline_info(markers_df)

    if not block_info and not baseline_info:
        print("[INFO] No block or baseline information extracted")
        return raw

    # Estimate EEG start time from the first marker timestamp
    # EEG starts before the first marker

    # Create annotations (without sampling rate)
    annotations = create_annotations_from_blocks_and_baseline(
        block_info, baseline_info, 0.0
    )

    # Add to Raw object
    raw.set_annotations(annotations)

    print(f"[INFO] {len(annotations)} annotations added:")
    print(f"   - {len(baseline_info)} baseline periods")
    print(f"   - {len(block_info)} experimental blocks")

    # Show the first few annotations as an example
    for i in range(min(5, len(annotations))):
        desc = annotations.description[i]
        onset = annotations.onset[i]
        duration = annotations.duration[i]
        print(f"   {desc}")

    if len(annotations) > 5:
        print(f"   ... and {len(annotations) - 5} more")

    return raw