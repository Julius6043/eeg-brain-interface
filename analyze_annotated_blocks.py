#!/usr/bin/env python3
"""Beispiel für die Nutzung der annotierten EEG-Daten.

Dieses Skript zeigt, wie man die von der Pipeline erstellten FIF-Dateien lädt
und spezifische Blöcke basierend auf n-back Level extrahiert.
"""

from pathlib import Path
import mne
import numpy as np
import matplotlib.pyplot as plt


def load_annotated_data(fif_path: Path):
    """Lädt eine annotierte FIF-Datei."""
    raw = mne.io.read_raw_fif(str(fif_path), preload=True)
    return raw


def extract_blocks_by_nback(raw, n_back_level: int):
    """Extrahiert alle Blöcke eines bestimmten n-back Levels.

    Parameter
    ---------
    raw : mne.io.Raw
        Raw-Objekt mit Annotationen
    n_back_level : int
        Gewünschtes n-back Level (0, 1, 2, 3)

    Rückgabe
    --------
    List[mne.io.Raw]
        Liste von Raw-Objekten für jeden Block des gewünschten Levels
    """
    blocks = []

    for annot_idx, description in enumerate(raw.annotations.description):
        if f"nback_{n_back_level}" in description and "block_" in description:
            onset = raw.annotations.onset[annot_idx]
            duration = raw.annotations.duration[annot_idx]

            # Extrahiere den Block
            tmin = onset
            tmax = onset + duration

            # Crop erstellt eine Kopie
            block_raw = raw.copy().crop(tmin=tmin, tmax=tmax)

            # Füge Metadaten hinzu
            block_raw.info["description"] = description

            blocks.append(block_raw)

    return blocks


def extract_baselines(raw):
    """Extrahiert alle Baseline-Perioden.

    Parameter
    ---------
    raw : mne.io.Raw
        Raw-Objekt mit Annotationen

    Rückgabe
    --------
    List[mne.io.Raw]
        Liste von Raw-Objekten für jede Baseline-Periode
    """
    baselines = []

    for annot_idx, description in enumerate(raw.annotations.description):
        if "baseline" in description:
            onset = raw.annotations.onset[annot_idx]
            duration = raw.annotations.duration[annot_idx]

            # Extrahiere die Baseline
            tmin = onset
            tmax = onset + duration

            # Crop erstellt eine Kopie
            baseline_raw = raw.copy().crop(tmin=tmin, tmax=tmax)

            # Füge Metadaten hinzu
            baseline_raw.info["description"] = description

            baselines.append(baseline_raw)

    return baselines


def analyze_blocks_example():
    """Beispiel-Analyse der annotierten Blöcke."""

    # Pfad zu den verarbeiteten Daten
    data_dir = Path("results/processed_with_annotations")

    if not data_dir.exists():
        print(f"Verzeichnis {data_dir} nicht gefunden!")
        print("Führen Sie zuerst die Pipeline aus: python test_pipeline_annotations.py")
        return

    # Suche nach FIF-Dateien
    fif_files = list(data_dir.rglob("*.fif"))

    if not fif_files:
        print("Keine FIF-Dateien gefunden!")
        return

    print(f"Gefundene FIF-Dateien: {len(fif_files)}")

    # Lade erste Datei als Beispiel
    fif_path = fif_files[0]
    print(f"\nAnalysiere: {fif_path}")

    raw = load_annotated_data(fif_path)

    print(f"\nDateninfo:")
    print(f"  Sampling Rate: {raw.info['sfreq']} Hz")
    print(f"  Kanäle: {raw.info['nchan']}")
    print(f"  Dauer: {raw.times[-1]:.1f} Sekunden")
    print(f"  Annotationen: {len(raw.annotations)}")

    # Zeige alle verfügbaren Annotationen
    print(f"\nVerfügbare Annotationen:")
    nback_counts = {}
    baseline_count = 0

    for desc in raw.annotations.description:
        if "baseline" in desc:
            baseline_count += 1
        elif "nback_" in desc:
            # Extrahiere n-back Level aus erweiterten Beschreibungen
            nback_part = desc.split("nback_")[1].split("_")[0]
            nback_counts[nback_part] = nback_counts.get(nback_part, 0) + 1

    print(f"  Baseline-Perioden: {baseline_count}")
    for nback, count in sorted(nback_counts.items()):
        print(f"  n-back {nback}: {count} Blöcke")

    # Analysiere auch Baseline-Perioden
    baselines = extract_baselines(raw)
    if baselines:
        print(f"\n=== BASELINE ANALYSE ===")
        print(f"Anzahl Baseline-Perioden: {len(baselines)}")

        # Berechne durchschnittliche Baseline-Dauer
        durations = [baseline.times[-1] for baseline in baselines]
        avg_duration = np.mean(durations)
        print(f"Durchschnittliche Baseline-Dauer: {avg_duration:.1f}s")

        # Beispiel: Power-Spektrum für erste Baseline
        if len(baselines) > 0:
            first_baseline = baselines[0]
            print(f"Erste Baseline: {first_baseline.info['description']}")

            # Sample-Informationen aus Beschreibung extrahieren
            desc = first_baseline.info["description"]
            if "_onset_" in desc and "_dur_" in desc:
                parts = desc.split("_")
                onset_s = float([p for p in parts if p.endswith("s")][0][:-1])
                onset_smp = int([p for p in parts if p.endswith("smp")][0][:-3])
                duration_s = float([p for p in parts if p.endswith("s")][1][:-1])
                duration_smp = int([p for p in parts if p.endswith("smp")][1][:-3])

                print(f"  Onset: {onset_s:.1f}s ({onset_smp} samples)")
                print(f"  Dauer: {duration_s:.1f}s ({duration_smp} samples)")

            # Einfache Spektralanalyse
            try:
                psd, freqs = mne.time_frequency.psd_welch(
                    first_baseline, fmin=1, fmax=40, n_fft=1024
                )

                # Durchschnittliche Power in verschiedenen Bändern
                alpha_power = np.mean(psd[:, (freqs >= 8) & (freqs <= 12)])
                beta_power = np.mean(psd[:, (freqs >= 13) & (freqs <= 30)])

                print(f"  Alpha Power (8-12 Hz): {alpha_power:.2e}")
                print(f"  Beta Power (13-30 Hz): {beta_power:.2e}")

            except Exception as e:
                print(f"  Spektralanalyse fehlgeschlagen: {e}")

    # Extrahiere Blöcke für verschiedene n-back Level
    for nback_level in [0, 1, 2, 3]:
        blocks = extract_blocks_by_nback(raw, nback_level)
        if blocks:
            print(f"\n=== N-BACK LEVEL {nback_level} ===")
            print(f"Anzahl Blöcke: {len(blocks)}")

            # Berechne durchschnittliche Blockdauer
            durations = [block.times[-1] for block in blocks]
            avg_duration = np.mean(durations)
            print(f"Durchschnittliche Blockdauer: {avg_duration:.1f}s")

            # Beispiel: Power-Spektrum für ersten Block
            if len(blocks) > 0:
                first_block = blocks[0]
                print(f"Erster Block: {first_block.info['description']}")

                # Sample-Informationen aus Beschreibung extrahieren
                desc = first_block.info["description"]
                if "_onset_" in desc and "_dur_" in desc:
                    parts = desc.split("_")
                    onset_s = float([p for p in parts if p.endswith("s")][0][:-1])
                    onset_smp = int([p for p in parts if p.endswith("smp")][0][:-3])
                    duration_s = float([p for p in parts if p.endswith("s")][1][:-1])
                    duration_smp = int([p for p in parts if p.endswith("smp")][1][:-3])

                    print(f"  Onset: {onset_s:.1f}s ({onset_smp} samples)")
                    print(f"  Dauer: {duration_s:.1f}s ({duration_smp} samples)")
                else:
                    print(f"  Dauer: {first_block.times[-1]:.1f}s")
                    print(f"  Samples: {len(first_block.times)}")

                # Einfache Spektralanalyse
                try:
                    psd, freqs = mne.time_frequency.psd_welch(
                        first_block, fmin=1, fmax=40, n_fft=1024
                    )

                    # Durchschnittliche Power in verschiedenen Bändern
                    alpha_power = np.mean(psd[:, (freqs >= 8) & (freqs <= 12)])
                    beta_power = np.mean(psd[:, (freqs >= 13) & (freqs <= 30)])

                    print(f"  Alpha Power (8-12 Hz): {alpha_power:.2e}")
                    print(f"  Beta Power (13-30 Hz): {beta_power:.2e}")

                except Exception as e:
                    print(f"  Spektralanalyse fehlgeschlagen: {e}")


def create_block_comparison():
    """Erstellt einen Vergleich zwischen verschiedenen n-back Leveln."""

    data_dir = Path("results/processed_with_annotations")
    fif_files = list(data_dir.rglob("*.fif"))

    if not fif_files:
        print("Keine FIF-Dateien für Vergleich gefunden!")
        return

    # Sammle Daten für alle Teilnehmer
    all_blocks = {0: [], 1: [], 2: [], 3: []}

    for fif_path in fif_files:
        try:
            raw = load_annotated_data(fif_path)
            participant = fif_path.parent.name

            for nback_level in [0, 1, 2, 3]:
                blocks = extract_blocks_by_nback(raw, nback_level)
                for block in blocks:
                    all_blocks[nback_level].append(
                        {
                            "participant": participant,
                            "raw": block,
                            "duration": block.times[-1],
                        }
                    )

        except Exception as e:
            print(f"Fehler beim Laden {fif_path}: {e}")

    # Statistiken erstellen
    print("\n=== BLOCK-VERGLEICH ===")
    for nback_level, blocks in all_blocks.items():
        if blocks:
            durations = [b["duration"] for b in blocks]
            participants = set(b["participant"] for b in blocks)

            print(f"\nN-back Level {nback_level}:")
            print(f"  Anzahl Blöcke: {len(blocks)}")
            print(f"  Teilnehmer: {len(participants)}")
            print(
                f"  Durchschn. Dauer: {np.mean(durations):.1f}s ± {np.std(durations):.1f}s"
            )
            print(
                f"  Min/Max Dauer: {np.min(durations):.1f}s / {np.max(durations):.1f}s"
            )


if __name__ == "__main__":
    print("=== EEG Block Analysis ===")

    analyze_blocks_example()
    print("\n" + "=" * 50)
    create_block_comparison()

    print("\n=== USAGE BEISPIELE ===")
    print("Um spezifische Blöcke zu extrahieren:")
    print("  blocks_nback2 = extract_blocks_by_nback(raw, 2)")
    print("  # Alle n-back 2 Blöcke")
    print("\nUm Baseline-Perioden zu extrahieren:")
    print("  baselines = extract_baselines(raw)")
    print("  # Alle Baseline-Perioden")
    print("\nUm einzelne Blöcke zu analysieren:")
    print("  block = blocks_nback2[0]")
    print("  psd, freqs = mne.time_frequency.psd_welch(block)")
    print("  # Spektralanalyse für ersten Block")
    print("\nAnnotations enthalten Zeit- und Sample-Informationen:")
    print("  Beispiel: block_00_nback_0_onset_129.5s_32363smp_dur_310.7s_77682smp")
    print("  - Onset: 129.5s (32363 samples)")
    print("  - Duration: 310.7s (77682 samples)")
