import mne
import pandas as pd
from pathlib import Path

from braindecode import EEGClassifier
from braindecode.datasets import create_from_mne_epochs
from braindecode.models import EEGNet
from sklearn.model_selection import GroupKFold
import numpy as np
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
import torch

# MNE Verbose-Level setzen um die vielen Meldungen zu reduzieren
mne.set_log_level('ERROR')


def load_and_prepare_data():
    base_dir = Path(__file__).parent.parent.parent.parent
    epochs_path = base_dir / "results" / "processed" / "Aliaa" / "indoor_processed-epo.fif"

    if not epochs_path.exists():
        raise FileNotFoundError(f"Epochs file not found: {epochs_path}")

    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose=False)

    baseline_condition_name = 'baseline'

    if baseline_condition_name not in epochs.event_id:
        raise ValueError(
            f"Baseline-Bedingung '{baseline_condition_name}' nicht in den Epochs gefunden. "
            f"Verfügbare Bedingungen: {list(epochs.event_id.keys())}"
        )

    # Trennen der Epochen
    baseline_epochs = epochs[baseline_condition_name]
    task_epochs = epochs[["1-back", "2-back", "3-back"]]

    print(f"Loaded {len(task_epochs)} task epochs.")
    print(f"Found {len(baseline_epochs)} baseline epochs for normalization.")

    return task_epochs, baseline_epochs


def add_metadata_with_targets(epochs):
    target_map = {'1-back': 0, '2-back': 1, '3-back': 2}

    try:
        event_mapping = {
            original_id: target_map[condition]
            for condition, original_id in epochs.event_id.items()
            if condition in target_map
        }
    except KeyError as e:
        raise RuntimeError(
            f"Bedingung '{e.args[0]}' aus target_map nicht in epochs.event_id gefunden. "
            f"Verfügbare Bedingungen: {list(epochs.event_id.keys())}"
        )

    epochs_with_meta = epochs.copy()

    try:
        targets = [event_mapping[event_id] for event_id in epochs_with_meta.events[:, 2]]
    except KeyError as e:
        raise RuntimeError(
            f"Event-ID '{e.args[0]}' aus den Daten konnte nicht im Mapping gefunden werden. "
            f"Erstelltes Mapping: {event_mapping}"
        )

    for i, original_event_id in enumerate(epochs_with_meta.events[:, 2]):
        epochs_with_meta.events[i, 2] = event_mapping[original_event_id]

    epochs_with_meta.metadata = pd.DataFrame({'target': targets})
    epochs_with_meta.event_id = target_map

    print("\nMetadaten mit 'target'-Spalte erfolgreich hinzugefügt.")
    print("Verwendetes direktes Mapping (Original-ID -> Target):", event_mapping)
    print("Events wurden auf 0, 1, 2 remapped")
    print("Beispiel-Metadaten:")
    print(epochs_with_meta.metadata.head())

    return epochs_with_meta


def normalize_epochs_with_baseline(task_epochs, baseline_epochs):
    print("\nNormalisiere Daten anhand der Baseline...")

    baseline_data = baseline_epochs.get_data(copy=False)  # Shape: (n_epochs, n_chans, n_times)

    # Berechne Mittelwert und Std über Epochen und Zeit für jeden Kanal
    mean_per_channel = baseline_data.mean(axis=(0, 2), keepdims=True)
    std_per_channel = baseline_data.std(axis=(0, 2), keepdims=True)

    # Division durch Null vermeiden, falls ein Kanal flatt ist
    std_per_channel[std_per_channel == 0] = 1

    # 2. Aufgaben-Daten holen
    task_data = task_epochs.get_data(copy=True)

    # 3. Z-Score Normalisierung anwenden (nutzt NumPy Broadcasting)
    normalized_task_data = (task_data - mean_per_channel) / std_per_channel

    # 4. Ein neues MNE Epochs-Objekt mit den normalisierten Daten erstellen
    normalized_epochs = mne.EpochsArray(normalized_task_data, task_epochs.info,
                                        events=task_epochs.events, tmin=task_epochs.tmin,
                                        event_id=task_epochs.event_id,
                                        metadata=task_epochs.metadata,
                                        verbose=False)

    print("Normalisierung abgeschlossen.")
    return normalized_epochs


def train_eegnet_with_cv():
    epochs, baseline_epochs = load_and_prepare_data()
    epochs = normalize_epochs_with_baseline(epochs, baseline_epochs)
    epochs = add_metadata_with_targets(epochs)

    sfreq = epochs.info['sfreq']

    epoch_length_s = (epochs.times[-1] - epochs.times[0])
    window_size_samples = len(epochs.times)

    window_stride_samples = window_size_samples

    print(f"Epochenlänge: {epoch_length_s:.2f}s ({window_size_samples} samples)")
    print(f"Sampling frequency: {sfreq}Hz")

    n_chans = epochs.info['nchan']
    n_classes = len(epochs.event_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nVerwende Gerät: {device}")
    print(f"Anzahl Klassen: {n_classes}")
    print(f"Anzahl Kanäle: {n_chans}")

    n_splits = 3

    epoch_indices = np.arange(len(epochs))
    groups = np.floor(epoch_indices / (len(epochs) / n_splits)).astype(int)

    groups[groups >= n_splits] = n_splits - 1

    print(f"\n{len(epochs)} Epochen in {n_splits} chronologische Blöcke für die Kreuzvalidierung aufgeteilt.")

    gkf = GroupKFold(n_splits=n_splits)

    all_fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X=epochs, y=epochs.metadata['target'], groups=groups)):
        print(f"\n--- Starte Fold {fold + 1}/{n_splits} ---")

        model = EEGNet(
            input_window_seconds=epoch_length_s,
            sfreq=sfreq,
            n_chans=n_chans,
            n_outputs=n_classes,
            final_conv_length='auto',
            pool_mode='max',
            kernel_length=64,
        )

        train_epochs = epochs[train_idx]
        test_epochs = epochs[test_idx]

        print(f"Train/Test Split: {len(train_epochs)}/{len(test_epochs)} Epochen")

        # Erstelle Datasets mit korrigiertem Windowing
        train_dataset = create_from_mne_epochs(
            [train_epochs],
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
            drop_last_window=False
        )

        test_dataset = create_from_mne_epochs(
            [test_epochs],
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
            drop_last_window=False
        )

        print(f"Trainingsdaten: {len(train_epochs)} Epochen -> {len(train_dataset)} Fenster")
        print(f"Testdaten:      {len(test_epochs)} Epochen -> {len(test_dataset)} Fenster")

        clf = EEGClassifier(
            model,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=0.001,
            optimizer__weight_decay=0.01,
            batch_size=16,
            max_epochs=100,
            train_split=predefined_split(test_dataset),
            device=device,
            classes=[0, 1, 2],
            callbacks=[
                ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=100 - 1)),
            ],
            verbose=1,
        )

        print("Starte Training für diesen Fold...")
        clf.fit(train_dataset, y=None)

        # Genauigkeit für diesen Fold speichern
        valid_acc = clf.history[-1, 'valid_acc']
        all_fold_accuracies.append(valid_acc)
        print(f"Fold {fold + 1} - Validation Accuracy: {valid_acc:.4f}")

    # --- 3. Ergebnisse zusammenfassen ---
    mean_accuracy = np.mean(all_fold_accuracies)
    std_accuracy = np.std(all_fold_accuracies)

    print("\n\n--- Kreuzvalidierung abgeschlossen ---")
    print(f"Genauigkeiten der einzelnen Folds: {[f'{acc:.4f}' for acc in all_fold_accuracies]}")
    print(f"Durchschnittliche Genauigkeit: {mean_accuracy:.4f}")
    print(f"Standardabweichung der Genauigkeit: {std_accuracy:.4f}")

    return all_fold_accuracies


if __name__ == "__main__":
    try:
        results = train_eegnet_with_cv()
        print("\nTraining and cross-validation completed successfully!")
    except Exception as e:
        print(f"\nEin Fehler ist aufgetreten: {e}")
        import traceback

        traceback.print_exc()
