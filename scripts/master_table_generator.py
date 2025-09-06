"""
This script processes EEG experiment data to build a master table containing spectral
features for each subject/session/block/segment combination.  It expects that
every subject participates in two sessions: ``session 1`` recorded under silent
conditions and ``session 2`` recorded under noisy conditions.  For each session
there is a corresponding ``*.fif`` file with the raw EEG data and a marker
``*.csv`` file containing event markers.


python master_table_generator.py --eeg_dir ../results/processed/Jannik --marker_dir ../results/processed/Jannik --subject_name Jannik --output_csv ../results/processed/Jannik_master.csv --debug



The workflow implemented here performs the following steps:

1.  **Load the marker log and determine N‑back difficulties** –
    The provided ``Block_difficulty_extractor.py`` file defines two helper
    functions: ``extract_nblock`` and ``calculate_nvals``.  We import these
    functions to determine the difficulty (0‑, 1‑, 2‑ or 3‑back) for each
    ``main_block`` of the experiment.  The marker CSV must contain
    ``sequence_*`` and ``targets_*`` rows preceding each ``main_block``
    to allow the difficulty to be inferred.

2.  **Identify block boundaries** –
    Each ``main_block_n_start`` and ``main_block_n_end`` marker delimits a
    consecutive block of trials.  The ``timestamp`` values stored in the
    marker CSV encode absolute times (e.g. seconds since some reference).  We
    first convert these times into **relative times** by subtracting the
    timestamp of the first marker (or the recording start time if available).
    Then, by multiplying with the sampling frequency, we obtain sample
    positions for the start and end of each block.  This approach yields
    realistic block durations—on the order of minutes—because the difference
    between the start and end markers represents the actual length of the
    block in seconds.

3.  **Segment the EEG signal** –
    Within each block we create overlapping segments of fixed length (5 s) with
    a 50 % overlap (i.e. a hop size of 2.5 s).  Segments extending beyond the
    end of the block are discarded.  Each segment is defined by its starting
    sample index and contains data from all eight electrodes.

4.  **Compute features** –
    Für jedes Segment und jede Elektrode berechnen wir die Leistungsverteilung
    (PSD) mittels der Welch‑Methode aus :mod:`scipy.signal`.  Aus den PSDs
    werden die mittleren Leistungen in den Theta‑ (4–7 Hz), Alpha‑
    (8–12 Hz) und Beta‑Bändern (13–30 Hz) extrahiert.  Diese Features sind
    besser geeignet, um Unterschiede im mentalen Workload zwischen den
    N‑Back‑Schwierigkeitsstufen zu erfassen, da mit steigender Belastung
    insbesondere die frontale Theta‑Aktivität zunimmt und parietale Alpha‑
    und Beta‑Aktivität abnimmt【431945578351821†L169-L182】.  Sie können den
    Feature‑Bereich an Ihre Bedürfnisse anpassen (z. B. weitere Bänder oder
    Entropie‑Maße).

5.  **Build the master table** –
    A pandas ``DataFrame`` is constructed where each row corresponds to a
    segment.  The columns are:

    * ``Subject`` – identifier derived from the filename (e.g. ``p1``).
    * ``Session`` – session number derived from the filename (``s1`` or
      ``s2``).
    * ``Location`` – ``silent`` for session 1 and ``noisy`` for session 2.
    * ``Block`` – index of the main block (0‑basiert).  There are typically
      sieben Blöcke (0–6).  This value indicates from which block the
      corresponding segment was extracted.
    * ``Segment`` – sequential segment number within the session (starting
      bei 0) across all blocks.
    * ``NBackDifficulty`` – Schwierigkeitswert, der durch ``calculate_nvals``
      (0, 1, 2 oder 3) für den Block bestimmt wird, aus dem das Segment
      stammt.
    * ``ThetaPowerElectrode1`` … ``ThetaPowerElectrode8`` – mittlere
      Leistung im Theta‑Band (4–7 Hz) für jede Elektrode.
    * ``AlphaPowerElectrode1`` … ``AlphaPowerElectrode8`` – mittlere
      Leistung im Alpha‑Band (8–12 Hz) pro Elektrode.
    * ``BetaPowerElectrode1`` … ``BetaPowerElectrode8`` – mittlere
      Leistung im Beta‑Band (13–30 Hz) pro Elektrode.

The resulting table can be saved to CSV or used directly for downstream
analysis or machine‑learning tasks.  See the ``main`` block at the end of
this file for example usage.

Note that this script requires the ``mne`` package to load ``.fif`` files.
If ``mne`` is not available in your environment you can install it via
``pip install mne``.  Also ensure that the SciPy and pandas packages are
available.

Author: ChatGPT, OpenAI
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple
import sys

import numpy as np
import pandas as pd
from scipy.signal import welch

try:
    import mne  # type: ignore
except ImportError as exc:
    raise ImportError(
        "The mne package is required to run this script. Please install it via\n"
        "\n    pip install mne\n\n"
        "and ensure that SciPy and pandas are also installed."
    ) from exc

# Import helper functions from the provided script to determine block difficulties.

sys.path.append(r"C:\Users\morit\OneDrive\Desktop\Uni\Master Neuroscience\Semester 2\BPR\eeg-brain-interface")

from Block_difficulty_extractor import calculate_nvals


@dataclass
class SessionInfo:
    """Container holding information about a single recording session."""

    subject: str
    session: str
    location: str
    eeg_path: str
    marker_path: str


def parse_filename_for_info(marker_filename: str) -> Tuple[str, str]:
    """Extract subject and session identifiers from a marker filename.

    The expected filename pattern is ``marker_log_pXsY_<name>.csv`` where ``pX``
    encodes the participant number and ``sY`` encodes the session number.  For
    example, ``marker_log_p1s2_Jannik.csv`` corresponds to subject ``p1``
    (participant 1) and session ``s2``.

    Parameters
    ----------
    marker_filename : str
        Filename of the marker CSV (without directory path).

    Returns
    -------
    Tuple[str, str]
        A tuple ``(subject, session)`` derived from the filename.
    """
    base = os.path.basename(marker_filename)
    # Example filename: marker_log_p1s1_Jannik.csv or marker_log_P1S2.csv
    # First, attempt to locate an ID pattern using a regular expression.  The pattern
    # looks for a 'p' or 'P' followed by one or more digits, followed by 's' or 'S'
    # and one or more digits (e.g. p1s2, P10S3).
    import re

    # Remove extension for easier matching
    name_no_ext = os.path.splitext(base)[0]
    match = re.search(r"[pP]\d+[sS]\d+", name_no_ext)
    if not match:
        raise ValueError(
            f"Cannot parse subject and session from filename: {marker_filename}."
            " Expected pattern like 'p1s2' or 'P1S2'."
        )
    id_part = match.group(0).lower()  # e.g. 'p1s2'
    # Split into subject and session by 's'
    if "s" not in id_part:
        raise ValueError(
            f"Identifier part '{id_part}' does not contain session information."
        )
    subject = id_part.split("s")[0]  # e.g. 'p1'
    session = "s" + id_part.split("s")[1]  # e.g. 's2'
    return subject, session


def read_markers(marker_path: str) -> pd.DataFrame:
    """Read the marker CSV file into a DataFrame.

    The marker file is expected to have two columns: ``marker`` and
    ``timestamp``.  The timestamp column should be convertible to a float.

    Parameters
    ----------
    marker_path : str
        Path to the marker CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing marker names and timestamps.
    """
    df = pd.read_csv(marker_path)
    # Ensure correct column names
    df.columns = [c.strip() for c in df.columns]
    # Drop any completely empty rows
    df = df.dropna(how="all")
    return df


def extract_block_times(
    marker_df: pd.DataFrame,
    sfreq: float,
    meas_start: float | None = None,
) -> List[Tuple[int, int]]:
    """Extract the sample indices for the start and end of each block.

    Parameters
    ----------
    marker_df : pandas.DataFrame
        Marker DataFrame with columns ``marker`` and ``timestamp``.
    sfreq : float
        Sampling frequency of the EEG recording in Hertz.

    Returns
    -------
    List[Tuple[int, int]]
        A list of ``(start_sample, end_sample)`` tuples for each block.  Blocks
        with missing start or end markers are skipped.
    """
    # Convert absolute timestamps to relative times (subtract the first timestamp)
    timestamps = marker_df["timestamp"].astype(float)
    # Determine zero‑reference time.  If `meas_start` (measurement start in
    # seconds) is provided, subtract it from the absolute timestamps.  This
    # aligns marker times with the start of the EEG recording.  Otherwise,
    # fall back to subtracting the first timestamp.
    if meas_start is not None:
        rel_times = timestamps - meas_start
    else:
        time0 = timestamps.iloc[0]
        rel_times = timestamps - time0

    # Build mapping of block indices to start/end relative times
    block_start_times = {}
    block_end_times = {}
    for marker, rel_time in zip(marker_df["marker"], rel_times):
        if marker.startswith("main_block_"):
            # Format example: main_block_0_start or main_block_0_end
            parts = marker.split("_")
            # parts[0] = 'main', parts[1] = 'block', parts[2] = index, parts[3] = 'start'/'end'/'accuracy' etc.
            if len(parts) < 4:
                continue
            block_idx = parts[2]
            event_type = parts[3]
            # Only handle start and end events
            if event_type == "start":
                block_start_times[block_idx] = rel_time
            elif event_type == "end":
                block_end_times[block_idx] = rel_time

    # Match blocks with both start and end
    block_bounds: List[Tuple[int, int]] = []
    for block_idx in sorted(block_start_times.keys(), key=lambda x: int(x)):
        if block_idx in block_end_times:
            start_time = block_start_times[block_idx]
            end_time = block_end_times[block_idx]
            # Convert to sample indices from seconds; ensure non-negative
            start_samp = max(int(np.floor(start_time * sfreq)), 0)
            end_samp = max(int(np.floor(end_time * sfreq)), 0)
            if end_samp > start_samp:
                block_bounds.append((start_samp, end_samp))
    return block_bounds


def segment_blocks(block_bounds: List[Tuple[int, int]], win_length_sec: float, hop_length_sec: float, sfreq: float) -> List[Tuple[int, int, int]]:
    """Segment each block into overlapping windows.

    Parameters
    ----------
    block_bounds : List[Tuple[int, int]]
        List of ``(start_sample, end_sample)`` tuples defining block boundaries.
    win_length_sec : float
        Length of each segment in seconds (e.g. 5.0).
    hop_length_sec : float
        Hop (stride) between segment starts in seconds (e.g. 2.5 for 50 % overlap).
    sfreq : float
        Sampling frequency in Hertz.

    Returns
    -------
    List[Tuple[int, int, int]]
        A list of ``(block_index, seg_start_sample, seg_end_sample)`` tuples.
    """
    win_length = int(np.round(win_length_sec * sfreq))
    hop_length = int(np.round(hop_length_sec * sfreq))
    segments: List[Tuple[int, int, int]] = []
    for block_idx, (start_samp, end_samp) in enumerate(block_bounds):
        segment_start = start_samp
        while True:
            segment_end = segment_start + win_length
            # Stop if segment exceeds block boundaries
            if segment_end > end_samp:
                break
            segments.append((block_idx, segment_start, segment_end))
            segment_start += hop_length
    return segments


def compute_bandpower(segment_data: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    """Compute bandpower features (theta, alpha, beta) for each channel.

    For each electrode in the segment, this function computes the average power
    within predefined frequency bands using Welch’s PSD estimate.  The bands
    used are:

    * Theta: 4–7 Hz
    * Alpha: 8–12 Hz
    * Beta: 13–30 Hz

    Parameters
    ----------
    segment_data : numpy.ndarray
        Array of shape ``(n_channels, n_samples)`` containing EEG data for a
        single segment.
    sfreq : float
        Sampling frequency in Hertz.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dictionary mapping band names (``'theta'``, ``'alpha'``, ``'beta'``) to
        arrays of length ``n_channels`` containing the mean power in that band
        for each channel.
    """
    # Define frequency band edges
    bands = {
        'theta': (4.0, 7.0),
        'alpha': (8.0, 12.0),
        'beta': (13.0, 30.0),
    }
    # Compute PSD once per channel
    band_powers = {band: [] for band in bands.keys()}
    for ch_data in segment_data:
        freqs, psd = welch(ch_data, fs=sfreq, nperseg=min(len(ch_data), 256))
        for band, (f_low, f_high) in bands.items():
            # Select frequency indices within band
            idx_band = np.logical_and(freqs >= f_low, freqs <= f_high)
            # Mean power in the band (integral approximated by mean * bandwidth)
            if np.any(idx_band):
                band_power = psd[idx_band].mean()
            else:
                band_power = 0.0
            band_powers[band].append(band_power)
    # Convert lists to numpy arrays
    for band in bands.keys():
        band_powers[band] = np.array(band_powers[band])
    return band_powers


def process_session(session_info: SessionInfo) -> pd.DataFrame:
    """Process a single session and return a DataFrame of extracted features.

    Parameters
    ----------
    session_info : SessionInfo
        Dataclass containing file paths and metadata for the session.

    Returns
    -------
    pandas.DataFrame
        DataFrame where each row corresponds to a segment with extracted
        features and associated metadata.
    """
    # Load raw EEG data
    print(f"Loading EEG from {session_info.eeg_path}…")
    raw = mne.io.read_raw_fif(session_info.eeg_path, preload=True, verbose='error')
    # Select the first eight channels.  If your FIF contains exactly eight
    # electrodes these will correspond to them; adjust selection as needed.
    data = raw.get_data(picks=range(8))  # shape: (n_channels, n_samples)
    sfreq = raw.info['sfreq']

    # Load markers and determine block difficulties
    marker_df = read_markers(session_info.marker_path)
    # Compute N‑back difficulties for each block using provided helper
    nback_difficulties = calculate_nvals(marker_df.copy())

    # Determine measurement start time for aligning absolute marker timestamps.
    #
    # Marker CSV files store timestamps as absolute times (e.g. seconds since a
    # global reference).  To convert these to sample indices we need to know
    # when the EEG recording started.  The ``meas_date`` field in the FIF
    # metadata encodes the measurement start.  If available, we convert it to
    # seconds and use it directly.  If ``meas_date`` is missing, we attempt
    # to approximate the start time by assuming the last marker occurs near
    # the end of the EEG recording and subtracting the recording duration from
    # the last marker timestamp.  This yields an estimate of the absolute
    # timestamp corresponding to sample 0.  Without such an approximation the
    # code would fallback to subtracting the first marker timestamp, which
    # incorrectly discards any baseline period before the first block.
    meas_date = raw.info.get('meas_date')  # can be None, tuple, or datetime
    meas_start: float | None = None
    if meas_date is not None:
        try:
            # ``meas_date`` can be a (seconds, microseconds) tuple or a datetime
            if isinstance(meas_date, tuple) and len(meas_date) == 2:
                secs, usecs = meas_date
                meas_start = secs + usecs * 1e-6
            else:
                # treat as datetime; convert to Unix timestamp in seconds
                meas_start = meas_date.timestamp()
        except Exception:
            meas_start = None
    if meas_start is None:
        # Fall back: approximate measurement start using the last marker and total
        # recording duration.  We assume the last marker occurs near the end of
        # the recording.  This prevents baseline data from being erroneously
        # discarded by aligning to the first marker.  If the marker CSV
        # contains no valid timestamps this approximation will still return
        # None and ``extract_block_times`` will use the first marker as zero.
        if not marker_df.empty and np.isfinite(marker_df["timestamp"].astype(float)).any():
            try:
                end_marker = marker_df["timestamp"].astype(float).max()
                total_duration = len(raw.times) / sfreq
                meas_start = end_marker - total_duration
            except Exception:
                meas_start = None

    # Extract start and end sample indices for each block
    block_bounds = extract_block_times(
        marker_df.copy(),
        sfreq,
        meas_start=meas_start,
    )

    # Sanity check: ensure the number of blocks matches difficulties length
    if len(block_bounds) != len(nback_difficulties):
        print(
            f"Warning: number of blocks ({len(block_bounds)}) does not match"
            f" number of difficulty values ({len(nback_difficulties)})."
        )
        # Truncate to the shorter length to avoid mismatches
        min_len = min(len(block_bounds), len(nback_difficulties))
        block_bounds = block_bounds[:min_len]
        nback_difficulties = nback_difficulties[:min_len]

    # Segment blocks
    segments = segment_blocks(
        block_bounds,
        win_length_sec=5.0,
        hop_length_sec=2.5,
        sfreq=sfreq,
    )

    # If debug mode is enabled (set via a global flag), print out
    # information about block boundaries and segment counts.  The debug flag is
    # attached to the main function as an attribute (see __main__).
    debug = getattr(process_session, 'debug', False)
    if debug:
        print(f"\nSession {session_info.subject} {session_info.session} ({session_info.location})")
        print(f"Sampling frequency: {sfreq} Hz")
        for i, (start_samp, end_samp) in enumerate(block_bounds):
            start_time = start_samp / sfreq
            end_time = end_samp / sfreq
            n_segments = sum(1 for seg in segments if seg[0] == i)
            diff = nback_difficulties[i] if i < len(nback_difficulties) else 'NA'
            print(
                f"Block {i}: start={start_time:.2f}s, end={end_time:.2f}s,"
                f" samples=({start_samp},{end_samp}), NBackDifficulty={diff},"
                f" segments={n_segments}"
            )

    # Prepare lists for DataFrame columns
    records = []
    for seg_idx, (block_idx, start_samp, end_samp) in enumerate(segments):
        # Extract data for this segment
        seg_data = data[:, start_samp:end_samp]
        # Compute band power for each electrode and each band
        band_powers = compute_bandpower(seg_data, sfreq)
        # Build record dictionary
        record = {
            "Subject": session_info.subject,
            "Session": session_info.session,
            "Location": session_info.location,
            # Block index (0-based) indicates which main block this segment belongs to
            "Block": block_idx,
            # Segment index enumerates segments across all blocks
            "Segment": seg_idx,
            "NBackDifficulty": nback_difficulties[block_idx],
        }
        # Insert band power features per electrode
        for band_name, powers in band_powers.items():
            for idx, power in enumerate(powers, start=1):
                feature_name = f"{band_name.capitalize()}PowerElectrode{idx}"
                record[feature_name] = power
        records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame.from_records(records)
    return df


def find_sessions(eeg_dir: str, marker_dir: str, subject_name: str | None = None) -> List[SessionInfo]:
    """Search for session files and return a list of SessionInfo objects.

    This function pairs ``*.fif`` files with ``marker_log*.csv`` files by
    subject and session identifiers parsed from their filenames.  It expects
    that the EEG and marker directories contain matching pairs for each
    session.  The location field is set based on the session number
    (``s1`` → ``silent``, ``s2`` → ``noisy``).

    Parameters
    ----------
    eeg_dir : str
        Directory containing ``.fif`` files.
    marker_dir : str
        Directory containing marker CSV files.

    Parameters
    ----------
    eeg_dir : str
        Directory containing ``.fif`` files.
    marker_dir : str
        Directory containing marker CSV files.
    subject_name : str | None, optional
        If provided, this string overrides the subject identifier parsed from
        the marker filenames.  This is useful when your filenames encode the
        session number (e.g. ``p1s1``) but you want the subject column in the
        output to reflect a more descriptive name (e.g. ``Jannik``).

    Returns
    -------
    List[SessionInfo]
        List of session information for all discovered pairs.
    """
    # Gather all EEG and marker files
    eeg_files = [f for f in os.listdir(eeg_dir) if f.lower().endswith('.fif')]
    marker_files = [f for f in os.listdir(marker_dir) if f.lower().endswith('.csv')]

    sessions: List[SessionInfo] = []
    for marker_file in marker_files:
        subject, session = parse_filename_for_info(marker_file)
        # Override subject if explicit name is given
        if subject_name:
            subject = subject_name
        # Determine location by session
        location = 'silent' if session.lower() == 's1' else 'noisy'
        # Find matching EEG file (assume contains subject and session in name)
        candidate_eeg = [ef for ef in eeg_files if subject in ef and session in ef]
        eeg_path: str | None = None
        if candidate_eeg:
            eeg_path = os.path.join(eeg_dir, candidate_eeg[0])
        else:
            # Fallback: use 'indoor' for session s1 and 'outdoor' for session s2
            if session.lower() == 's1':
                fallback = [ef for ef in eeg_files if 'indoor' in ef.lower()]
                if fallback:
                    eeg_path = os.path.join(eeg_dir, fallback[0])
            elif session.lower() == 's2':
                fallback = [ef for ef in eeg_files if 'outdoor' in ef.lower()]
                if fallback:
                    eeg_path = os.path.join(eeg_dir, fallback[0])
            if eeg_path is None:
                print(f"Warning: no EEG file found for {subject} {session}")
                continue
        # Compose full path for marker file
        marker_path = os.path.join(marker_dir, marker_file)
        sessions.append(
            SessionInfo(
                subject=subject,
                session=session,
                location=location,
                eeg_path=eeg_path,
                marker_path=marker_path,
            )
        )
    return sessions


def main(eeg_dir: str, marker_dir: str, output_csv: str) -> None:
    """Entry point to process all sessions and save the master table as CSV.

    Parameters
    ----------
    eeg_dir : str
        Path to directory containing ``*.fif`` files.
    marker_dir : str
        Path to directory containing marker CSV files.
    output_csv : str
        Path where the aggregated master table should be saved.
    """
    # Discover sessions
    # The subject name override is passed via the closure below (see __main__)
    sessions = find_sessions(eeg_dir, marker_dir, subject_name=getattr(main, 'subject_name_override', None))
    if not sessions:
        raise RuntimeError(
            f"No matching session files found in {eeg_dir} and {marker_dir}."
        )
    print(f"Found {len(sessions)} session(s) to process.")

    # Process each session and concatenate results
    all_tables = []
    for sess in sessions:
        df = process_session(sess)
        all_tables.append(df)

    master_table = pd.concat(all_tables, ignore_index=True)
    # Save to CSV
    master_table.to_csv(output_csv, index=False)
    print(f"Master table saved to {output_csv}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate a master table from EEG sessions.")
    # The following arguments are optional.  If none are provided the script will
    # search for files in the current working directory and save ``master_table.csv``
    # into that directory.  This allows you to simply run
    # ``python master_table_generator.py`` from a directory containing both the
    # ``.fif`` and marker ``.csv`` files without specifying any command‑line
    # parameters.
    parser.add_argument(
        '--eeg_dir', default='.', help='Directory containing FIF files (default: current directory)'
    )
    parser.add_argument(
        '--marker_dir', default='.', help='Directory containing marker CSV files (default: current directory)'
    )
    parser.add_argument(
        '--output_csv', default=None, help='Optional path to write the resulting master table CSV.  If not provided, the file will be saved into the directory containing the EEG data.'
    )
    parser.add_argument(
        '--subject_name', default=None, help='Optional override for the subject identifier (e.g. Jannik)'
    )
    parser.add_argument(
        '--debug', action='store_true', help='Print detailed information about blocks and segments for sanity checks'
    )
    args = parser.parse_args()

    # Attach the subject name override to the main function (hacky but simple)
    if args.subject_name:
        setattr(main, 'subject_name_override', args.subject_name)
    # Attach debug flag for process_session
    if args.debug:
        setattr(process_session, 'debug', True)
    # Determine output path.  If none provided, save into the EEG directory
    # using a default filename.  This ensures the master table resides
    # alongside the raw EEG data by default.
    if args.output_csv is None:
        default_filename = 'master_table.csv'
        output_path = os.path.join(args.eeg_dir, default_filename)
    else:
        output_path = args.output_csv

    main(args.eeg_dir, args.marker_dir, output_path)

    