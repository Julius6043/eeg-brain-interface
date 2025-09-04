"""Beispiel-Analyse von epochierten EEG-Daten.

Dieses Skript zeigt, wie die epochierten EEG-Daten aus der Pipeline
analysiert werden können. Die Epochen enthalten die 3D-Datenstruktur
(Epochen x Kanäle x Zeit) mit entsprechenden Labels.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_welch

from src.eeg_pipeline import EEGPipeline, create_default_config
from src.eeg_pipeline.epoching import (
    extract_epochs_by_type,
    extract_epochs_by_nback,
    extract_epochs_by_block_name,
    get_epochs_summary,
)


def analyze_epochs_example():
    """Beispiel-Analyse der epochierten Daten."""

    # 1. Pipeline ausführen mit Epoching
    print("=== EEG-Pipeline mit Epoching ===")
    config = create_default_config()
    config.output_dir = Path("results/epoched_analysis")

    pipeline = EEGPipeline(config)
    sessions = pipeline.run(Path("data"))

    # 2. Analysiere erste Session mit Epochen
    for session in sessions:
        if session.indoor_epochs is not None:
            print(f"\n=== Analyse für {session.participant_name} ===")
            epochs = session.indoor_epochs

            # Überblick über Epochen
            print(f"Epochen-Form: {epochs.get_data().shape}")
            print(f"  - {epochs.get_data().shape[0]} Epochen")
            print(f"  - {epochs.get_data().shape[1]} Kanäle")
            print(f"  - {epochs.get_data().shape[2]} Zeitpunkte")
            print(f"  - Sampling-Rate: {epochs.info['sfreq']} Hz")
            print(f"  - Epochen-Dauer: {epochs.times[-1]:.1f}s")

            # Epochen-Zusammenfassung
            summary = get_epochs_summary(epochs)
            print(f"\nEpochen-Zusammenfassung:")
            print(summary)

            # 3. Extrahiere verschiedene Epochen-Typen
            print(f"\n=== Epochen-Extraktion ===")

            # Baseline vs. Experimentelle Blöcke
            baselines = extract_epochs_by_type(epochs, "baseline")
            blocks = extract_epochs_by_type(epochs, "block")

            print(f"Baselines: {len(baselines)} Epochen")
            print(f"Experimentelle Blöcke: {len(blocks)} Epochen")

            # N-back Level
            for nback_level in [0, 1, 2, 3]:
                nback_epochs = extract_epochs_by_nback(epochs, nback_level)
                if len(nback_epochs) > 0:
                    print(f"N-back {nback_level}: {len(nback_epochs)} Epochen")

            # 4. Spektralanalyse
            print(f"\n=== Spektralanalyse ===")

            # Baseline Power
            if len(baselines) > 0:
                baseline_data = baselines.get_data()  # (n_epochs, n_channels, n_times)

                # Power Spectral Density für Baseline
                psd_baseline, freqs = psd_welch(baselines, fmin=1, fmax=40, n_fft=2048)

                # Alpha-Power (8-12 Hz)
                alpha_idx = (freqs >= 8) & (freqs <= 12)
                alpha_power_baseline = np.mean(psd_baseline[:, :, alpha_idx], axis=2)

                print(
                    f"Baseline Alpha-Power: {np.mean(alpha_power_baseline):.2e} V²/Hz"
                )

            # N-back 3 Power (höchste kognitive Last)
            nback3 = extract_epochs_by_nback(epochs, 3)
            if len(nback3) > 0:
                psd_nback3, freqs = psd_welch(nback3, fmin=1, fmax=40, n_fft=2048)

                # Alpha-Power für N-back 3
                alpha_idx = (freqs >= 8) & (freqs <= 12)
                alpha_power_nback3 = np.mean(psd_nback3[:, :, alpha_idx], axis=2)

                print(f"N-back 3 Alpha-Power: {np.mean(alpha_power_nback3):.2e} V²/Hz")

                # Alpha-Suppression (Baseline vs. N-back 3)
                if len(baselines) > 0:
                    alpha_suppression = (
                        np.mean(alpha_power_baseline) - np.mean(alpha_power_nback3)
                    ) / np.mean(alpha_power_baseline)
                    print(
                        f"Alpha-Suppression (Baseline → N-back 3): {alpha_suppression:.1%}"
                    )

            # 5. Einzelne Epochen-Analyse
            print(f"\n=== Einzelne Epochen ===")
            for i, desc in enumerate(
                epochs.metadata["description"][:3]
            ):  # Erste 3 Epochen
                epoch_data = epochs.get_data()[i]  # (n_channels, n_times)
                block_name = epochs.metadata.iloc[i]["block_name"]
                difficulty = epochs.metadata.iloc[i]["difficulty"]

                print(f"Epoche {i+1}: {block_name} ({difficulty})")
                print(f"  - Daten-Form: {epoch_data.shape}")
                print(f"  - RMS-Amplitude: {np.sqrt(np.mean(epoch_data**2)):.2e} V")

            # 6. Daten-Export Beispiel
            print(f"\n=== Daten-Export Beispiel ===")

            # Extrahiere alle N-back 2 Epochen für weitere Analyse
            nback2 = extract_epochs_by_nback(epochs, 2)
            if len(nback2) > 0:
                nback2_data = nback2.get_data()  # (n_epochs, n_channels, n_times)
                nback2_metadata = nback2.metadata

                print(f"N-back 2 Daten exportierbar:")
                print(f"  - Form: {nback2_data.shape}")
                print(f"  - Block-Namen: {list(nback2_metadata['block_name'])}")

                # Beispiel: Speichere als NumPy Array
                # np.save(f"nback2_{session.participant_name}.npy", nback2_data)
                # nback2_metadata.to_csv(f"nback2_{session.participant_name}_metadata.csv")

            break  # Analysiere nur ersten Teilnehmer für Demo


def load_and_analyze_epochs(participant_dir: Path):
    """Lade gespeicherte Epochen und analysiere sie."""

    print(f"\n=== Lade Epochen für {participant_dir.name} ===")

    # Lade Epochen-Datei
    epochs_file = participant_dir / "indoor_epochs.fif"
    if epochs_file.exists():
        epochs = mne.read_epochs(str(epochs_file))

        print(f"Epochen geladen: {epochs.get_data().shape}")
        print(f"Metadaten verfügbar: {epochs.metadata is not None}")

        if epochs.metadata is not None:
            # Zeige verfügbare Labels
            print(f"\nVerfügbare Block-Namen: {epochs.metadata['block_name'].unique()}")
            print(
                f"Verfügbare Schwierigkeitsgrade: {epochs.metadata['difficulty'].unique()}"
            )

            # Schnelle Analyse
            for difficulty in epochs.metadata["difficulty"].unique():
                if pd.notna(difficulty):
                    mask = epochs.metadata["difficulty"] == difficulty
                    count = mask.sum()
                    print(f"{difficulty}: {count} Epochen")

        return epochs
    else:
        print(f"Keine Epochen-Datei gefunden: {epochs_file}")
        return None


if __name__ == "__main__":
    # Führe Beispiel-Analyse durch
    analyze_epochs_example()

    # Alternative: Lade bereits gespeicherte Epochen
    results_dir = Path("results/epoched_analysis")
    if results_dir.exists():
        for participant_dir in results_dir.iterdir():
            if participant_dir.is_dir():
                epochs = load_and_analyze_epochs(participant_dir)
                if epochs:
                    break  # Analysiere nur ersten gefundenen Teilnehmer
