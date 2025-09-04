"""
FIF Epochs Data Loader Script

Lädt verarbeitete Epochen-Dateien (.fif) und erstellt Verteilungsplots
für die n-back Daten aus der EEG-Pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne.io import read_raw_fif


@dataclass
class EpochExperiment:
    """Container für Epochen-Daten eines Teilnehmers."""
    participant_name: str
    indoor_epochs: Optional[mne.Epochs] = None
    outdoor_epochs: Optional[mne.Epochs] = None


def load_epochs_from_fif(root_dir: Path) -> Dict[str, EpochExperiment]:
    """
    Lädt alle Epochen-Dateien aus dem verarbeiteten Datenverzeichnis.

    Parameter
    ---------
    root_dir : Path
        Pfad zum results/processed Verzeichnis

    Rückgabe
    --------
    Dict[str, EpochExperiment]
        Dictionary mit Teilnehmername als Key und EpochExperiment als Value
    """
    experiments = {}
    experiment_dirs = [x for x in root_dir.iterdir() if x.is_dir()]

    print(f"Gefundene Teilnehmer-Verzeichnisse: {len(experiment_dirs)}")

    for experiment_dir in experiment_dirs:
        participant_name = experiment_dir.name
        print(f"\nLade Daten für {participant_name}...")

        # Pfade zu Epochen-Dateien
        indoor_epochs_path = experiment_dir / "indoor_processed-epo.fif"
        outdoor_epochs_path = experiment_dir / "outdoor_processed-epo.fif"

        indoor_epochs, outdoor_epochs = None, None

        # Indoor Epochen laden
        if indoor_epochs_path.exists():
            try:
                indoor_epochs = mne.read_epochs(str(indoor_epochs_path), verbose=False)
                print(f"  ✓ Indoor Epochen: {len(indoor_epochs)} Epochen")
            except Exception as e:
                print(f"  ✗ Fehler beim Laden der Indoor Epochen: {e}")
        else:
            print(f"  ⚠ Keine Indoor Epochen-Datei gefunden")

        # Outdoor Epochen laden
        if outdoor_epochs_path.exists():
            try:
                outdoor_epochs = mne.read_epochs(str(outdoor_epochs_path), verbose=False)
                print(f"  ✓ Outdoor Epochen: {len(outdoor_epochs)} Epochen")
            except Exception as e:
                print(f"  ✗ Fehler beim Laden der Outdoor Epochen: {e}")
        else:
            print(f"  ⚠ Keine Outdoor Epochen-Datei gefunden")

        # Experiment erstellen
        experiment = EpochExperiment(
            participant_name=participant_name,
            indoor_epochs=indoor_epochs,
            outdoor_epochs=outdoor_epochs
        )
        experiments[participant_name] = experiment

    return experiments


def create_label_distribution_summary(experiments: Dict[str, EpochExperiment]) -> pd.DataFrame:
    """
    Erstellt eine Zusammenfassung der Label-Verteilungen für alle Teilnehmer.

    Parameter
    ---------
    experiments : Dict[str, EpochExperiment]
        Dictionary mit geladenen Experimenten

    Rückgabe
    --------
    pd.DataFrame
        DataFrame mit Verteilungsdaten
    """
    summary_data = []

    for participant_name, experiment in experiments.items():
        # Indoor Daten
        if experiment.indoor_epochs:
            for event_name, event_id in experiment.indoor_epochs.event_id.items():
                count = sum(experiment.indoor_epochs.events[:, 2] == event_id)
                summary_data.append({
                    'participant': participant_name,
                    'condition': 'indoor',
                    'label': event_name,
                    'count': count,
                    'total_epochs': len(experiment.indoor_epochs),
                    'percentage': (count / len(experiment.indoor_epochs)) * 100
                })

        # Outdoor Daten
        if experiment.outdoor_epochs:
            for event_name, event_id in experiment.outdoor_epochs.event_id.items():
                count = sum(experiment.outdoor_epochs.events[:, 2] == event_id)
                summary_data.append({
                    'participant': participant_name,
                    'condition': 'outdoor',
                    'label': event_name,
                    'count': count,
                    'total_epochs': len(experiment.outdoor_epochs),
                    'percentage': (count / len(experiment.outdoor_epochs)) * 100
                })

    return pd.DataFrame(summary_data)


def plot_epoch_distributions(summary_df: pd.DataFrame, save_dir: Optional[Path] = None):
    """
    Erstellt verschiedene Plots zur Visualisierung der Epochen-Verteilungen.

    Parameter
    ---------
    summary_df : pd.DataFrame
        Zusammenfassung der Verteilungsdaten
    save_dir : Path, optional
        Verzeichnis zum Speichern der Plots
    """
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Gesamtverteilung aller Labels
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EEG Epochen Verteilungen - N-Back Experiment', fontsize=16, fontweight='bold')

    # Plot 1: Absolute Anzahl pro Label und Condition
    ax1 = axes[0, 0]
    pivot_counts = summary_df.pivot_table(
        index='label',
        columns='condition',
        values='count',
        aggfunc='sum',
        fill_value=0
    )
    pivot_counts.plot(kind='bar', ax=ax1, width=0.7)
    ax1.set_title('Absolute Epochen-Anzahl pro Label')
    ax1.set_xlabel('N-Back Level')
    ax1.set_ylabel('Anzahl Epochen')
    ax1.legend(title='Bedingung')
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Verteilung pro Teilnehmer (Heatmap)
    ax2 = axes[0, 1]
    participant_label_counts = summary_df.pivot_table(
        index='participant',
        columns='label',
        values='count',
        aggfunc='sum',
        fill_value=0
    )
    sns.heatmap(participant_label_counts, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Epochen pro Teilnehmer und Label')
    ax2.set_xlabel('N-Back Level')
    ax2.set_ylabel('Teilnehmer')

    # Plot 3: Durchschnittliche Prozentverteilung
    ax3 = axes[1, 0]
    avg_percentages = summary_df.groupby(['label', 'condition'])['percentage'].mean().reset_index()
    for condition in avg_percentages['condition'].unique():
        condition_data = avg_percentages[avg_percentages['condition'] == condition]
        ax3.bar(condition_data['label'], condition_data['percentage'],
                alpha=0.7, label=condition, width=0.6)
    ax3.set_title('Durchschnittliche Prozentverteilung')
    ax3.set_xlabel('N-Back Level')
    ax3.set_ylabel('Durchschnittlicher Anteil (%)')
    ax3.legend(title='Bedingung')
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Gesamtanzahl Epochen pro Teilnehmer
    ax4 = axes[1, 1]
    total_epochs = summary_df.groupby(['participant', 'condition'])['total_epochs'].first().reset_index()
    for condition in total_epochs['condition'].unique():
        condition_data = total_epochs[total_epochs['condition'] == condition]
        ax4.bar(condition_data['participant'], condition_data['total_epochs'],
                alpha=0.7, label=condition, width=0.6)
    ax4.set_title('Gesamtanzahl Epochen pro Teilnehmer')
    ax4.set_xlabel('Teilnehmer')
    ax4.set_ylabel('Anzahl Epochen')
    ax4.legend(title='Bedingung')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / 'epoch_distributions_overview.png', dpi=300, bbox_inches='tight')
        print(f"Plot gespeichert: {save_dir / 'epoch_distributions_overview.png'}")

    plt.show()


def plot_detailed_participant_distributions(summary_df: pd.DataFrame, save_dir: Optional[Path] = None):
    """
    Erstellt detaillierte Plots für jeden Teilnehmer einzeln.

    Parameter
    ---------
    summary_df : pd.DataFrame
        Zusammenfassung der Verteilungsdaten
    save_dir : Path, optional
        Verzeichnis zum Speichern der Plots
    """
    participants = summary_df['participant'].unique()
    n_participants = len(participants)

    # Berechne Grid-Dimensionen
    cols = 3
    rows = (n_participants + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle('Detaillierte Epochen-Verteilungen pro Teilnehmer', fontsize=16, fontweight='bold')

    # Flache axes für einfachere Iteration
    if n_participants == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for i, participant in enumerate(participants):
        ax = axes_flat[i]

        # Daten für diesen Teilnehmer
        participant_data = summary_df[summary_df['participant'] == participant]

        # Erstelle gestapeltes Balkendiagramm
        indoor_data = participant_data[participant_data['condition'] == 'indoor']
        outdoor_data = participant_data[participant_data['condition'] == 'outdoor']

        labels = sorted(participant_data['label'].unique())
        indoor_counts = [indoor_data[indoor_data['label'] == label]['count'].sum() for label in labels]
        outdoor_counts = [outdoor_data[outdoor_data['label'] == label]['count'].sum() for label in labels]

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, indoor_counts, width, label='Indoor', alpha=0.8)
        ax.bar(x + width / 2, outdoor_counts, width, label='Outdoor', alpha=0.8)

        ax.set_title(f'{participant}')
        ax.set_xlabel('N-Back Level')
        ax.set_ylabel('Anzahl Epochen')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()

        # Füge Zahlen über den Balken hinzu
        for j, (indoor_count, outdoor_count) in enumerate(zip(indoor_counts, outdoor_counts)):
            if indoor_count > 0:
                ax.text(j - width / 2, indoor_count + 1, str(indoor_count),
                        ha='center', va='bottom', fontsize=8)
            if outdoor_count > 0:
                ax.text(j + width / 2, outdoor_count + 1, str(outdoor_count),
                        ha='center', va='bottom', fontsize=8)

    # Verstecke leere Subplots
    for i in range(n_participants, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / 'detailed_participant_distributions.png', dpi=300, bbox_inches='tight')
        print(f"Plot gespeichert: {save_dir / 'detailed_participant_distributions.png'}")

    plt.show()


def print_summary_statistics(summary_df: pd.DataFrame):
    """
    Druckt zusammenfassende Statistiken der Epochen-Verteilungen.

    Parameter
    ---------
    summary_df : pd.DataFrame
        Zusammenfassung der Verteilungsdaten
    """
    print("\n" + "=" * 60)
    print("ZUSAMMENFASSENDE STATISTIKEN")
    print("=" * 60)

    # Gesamtstatistiken
    total_epochs = summary_df['count'].sum()
    total_participants = summary_df['participant'].nunique()
    conditions = summary_df['condition'].unique()
    labels = sorted(summary_df['label'].unique())

    print(f"Gesamtanzahl Epochen: {total_epochs:,}")
    print(f"Anzahl Teilnehmer: {total_participants}")
    print(f"Bedingungen: {', '.join(conditions)}")
    print(f"N-Back Level: {', '.join(labels)}")

    # Verteilung pro Bedingung
    print(f"\nEpochen pro Bedingung:")
    condition_totals = summary_df.groupby('condition')['count'].sum()
    for condition, count in condition_totals.items():
        percentage = (count / total_epochs) * 100
        print(f"  {condition}: {count:,} ({percentage:.1f}%)")

    # Verteilung pro Label
    print(f"\nEpochen pro N-Back Level:")
    label_totals = summary_df.groupby('label')['count'].sum().sort_index()
    for label, count in label_totals.items():
        percentage = (count / total_epochs) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")

    # Teilnehmer-Statistiken
    print(f"\nTeilnehmer-Statistiken:")
    participant_totals = summary_df.groupby('participant')['count'].sum()
    print(f"  Durchschnittliche Epochen pro Teilnehmer: {participant_totals.mean():.1f}")
    print(f"  Min Epochen: {participant_totals.min()}")
    print(f"  Max Epochen: {participant_totals.max()}")

    # Balance-Check
    print(f"\nLabel-Balance-Check:")
    for condition in conditions:
        condition_data = summary_df[summary_df['condition'] == condition]
        label_counts = condition_data.groupby('label')['count'].sum()
        if len(label_counts) > 0:
            balance_ratio = label_counts.max() / label_counts.min() if label_counts.min() > 0 else float('inf')
            print(f"  {condition}: Balance-Ratio = {balance_ratio:.2f} (ideal: 1.0)")


def main():
    """Hauptfunktion - lädt Daten und erstellt alle Plots."""
    # Pfade definieren - korrekte relative Pfade vom Script-Verzeichnis
    script_dir = Path(__file__).parent  # src/eeg_pipeline/scripts/
    project_root = script_dir.parent.parent.parent  # Zurück zum Projekt-Root
    root_dir = project_root / "results" / "processed"
    save_dir = project_root / "results" / "plots"

    print(f"Script-Verzeichnis: {script_dir}")
    print(f"Projekt-Root: {project_root}")
    print(f"Suche Epochen in: {root_dir}")

    if not root_dir.exists():
        print(f"Fehler: Verzeichnis {root_dir} existiert nicht!")
        # Zeige verfügbare Verzeichnisse zur Debugging
        if project_root.exists():
            print(f"Verfügbare Verzeichnisse in {project_root}:")
            for item in project_root.iterdir():
                if item.is_dir():
                    print(f"  - {item.name}")
        return

    print("=" * 60)
    print("FIF EPOCHEN DATA LOADER")
    print("=" * 60)

    # 1. Lade alle Epochen
    print("\n1. Lade Epochen-Dateien...")
    experiments = load_epochs_from_fif(root_dir)

    if not experiments:
        print("Keine Experimente gefunden!")
        # Debugging: Zeige was im processed Verzeichnis ist
        if root_dir.exists():
            print(f"\nVerfügbare Dateien/Ordner in {root_dir}:")
            for item in root_dir.iterdir():
                print(f"  - {item.name} ({'Verzeichnis' if item.is_dir() else 'Datei'})")
        return

    # 2. Erstelle Verteilungs-Summary
    print("\n2. Erstelle Verteilungs-Zusammenfassung...")
    summary_df = create_label_distribution_summary(experiments)

    # 3. Drucke Statistiken
    print_summary_statistics(summary_df)

    # 4. Erstelle Plots
    print("\n3. Erstelle Übersichts-Plots...")
    plot_epoch_distributions(summary_df, save_dir)

    print("\n4. Erstelle detaillierte Teilnehmer-Plots...")
    plot_detailed_participant_distributions(summary_df, save_dir)

    # 5. Speichere Summary als CSV
    if save_dir:
        save_dir.mkdir(exist_ok=True)
        csv_path = save_dir / "epoch_distribution_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\nZusammenfassung gespeichert: {csv_path}")

    print("\n✓ Analyse abgeschlossen!")


if __name__ == '__main__':
    main()
