import mne
import pandas as pd
from pathlib import Path

from braindecode import EEGClassifier
from braindecode.datasets import create_from_mne_epochs
from braindecode.models import EEGNet, ShallowFBCSPNet, Deep4Net, EEGNetv4, ATCNet, AttentionBaseNet

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
import torch
from sklearn.metrics import confusion_matrix, classification_report, f1_score

mne.set_log_level("ERROR")


def load_and_prepare_data():
    base_dir = Path(__file__).parent.parent.parent.parent
    epochs_path = (
        base_dir / "results" / "processed" / "Aliaa" / "indoor_processed-epo.fif"
    )

    if not epochs_path.exists():
        raise FileNotFoundError(f"Epochs file not found: {epochs_path}")

    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose=False)

    baseline_condition_name = "baseline"

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
    target_map = {"1-back": 0, "2-back": 1, "3-back": 2}

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
        targets = [
            event_mapping[event_id] for event_id in epochs_with_meta.events[:, 2]
        ]
    except KeyError as e:
        raise RuntimeError(
            f"Event-ID '{e.args[0]}' aus den Daten konnte nicht im Mapping gefunden werden. "
            f"Erstelltes Mapping: {event_mapping}"
        )

    for i, original_event_id in enumerate(epochs_with_meta.events[:, 2]):
        epochs_with_meta.events[i, 2] = event_mapping[original_event_id]

    epochs_with_meta.metadata = pd.DataFrame({"target": targets})
    epochs_with_meta.event_id = target_map

    print("\nMetadaten mit 'target'-Spalte erfolgreich hinzugefügt.")
    print("Verwendetes direktes Mapping (Original-ID -> Target):", event_mapping)
    print("Events wurden auf 0, 1, 2 remapped")
    print("Beispiel-Metadaten:")
    print(epochs_with_meta.metadata.head())

    return epochs_with_meta


def normalize_epochs_with_baseline(task_epochs, baseline_epochs, config=None):
    print("\nNormalisiere Daten anhand der Baseline mit erweiterten Filtern...")

    baseline_data = baseline_epochs.get_data(
        copy=False
    )  # Shape: (n_epochs, n_chans, n_times)

    filtered_data = task_epochs.get_data(copy=True)

    # 3. RobustScaler für robuste Normalisierung (weniger sensitiv zu Outliers)
    baseline_reshaped = baseline_data.transpose(1, 0, 2).reshape(baseline_data.shape[1], -1).T
    filtered_reshaped = filtered_data.transpose(1, 0, 2).reshape(filtered_data.shape[1], -1).T
    
    # RobustScaler auf Baseline-Daten fitten
    scaler = StandardScaler()
    scaler.fit(baseline_reshaped)
    
    # Normalisierung auf gefilterte Task-Daten anwenden
    normalized_reshaped = scaler.transform(filtered_reshaped)
    
    # Zurück in ursprüngliche Form bringen: (n_epochs, n_channels, n_times)
    normalized_filtered_data = normalized_reshaped.T.reshape(
        filtered_data.shape[1], filtered_data.shape[0], filtered_data.shape[2]
    ).transpose(1, 0, 2)

    # 3. Ein neues MNE Epochs-Objekt mit den normalisierten und gefilterten Daten erstellen
    normalized_epochs = mne.EpochsArray(
        normalized_filtered_data,
        task_epochs.info,
        events=task_epochs.events,
        tmin=task_epochs.tmin,
        event_id=task_epochs.event_id,
        metadata=task_epochs.metadata,
        verbose=False,
    )

    print("Normalisierung und Filterung abgeschlossen.")
    return normalized_epochs


def augment_eeg_data(epochs, augment_factor=2):
    print(f"\nAugmentiere Daten mit Faktor {augment_factor} (erweiterte Methoden)...")

    original_data = epochs.get_data()
    original_events = epochs.events
    original_metadata = epochs.metadata

    augmented_data_list = [original_data]
    augmented_events_list = [original_events]
    augmented_metadata_list = [original_metadata]

    for aug_idx in range(augment_factor - 1):
        augmented_data = original_data.copy()
        
        # 1. Gaussian Noise (bewährt)
        noise_level = 0.03  # Reduziert für weniger Störung
        augmented_data += np.random.normal(0, noise_level, original_data.shape)
        
        # 2. Time Jittering - leichte zeitliche Verschiebung
        max_shift = 5  # Max 5 Samples Verschiebung
        for epoch_idx in range(len(augmented_data)):
            shift = np.random.randint(-max_shift, max_shift)
            if shift != 0:
                augmented_data[epoch_idx] = np.roll(augmented_data[epoch_idx], shift, axis=1)
        
        # 3. Amplitude Scaling pro Kanal
        for epoch_idx in range(len(augmented_data)):
            for ch_idx in range(augmented_data.shape[1]):
                scale_factor = np.random.uniform(0.9, 1.1)  # 10% Amplitude Varianz
                augmented_data[epoch_idx, ch_idx] *= scale_factor
        
        # 4. Channel Dropout (zufällig einzelne Kanäle nullsetzen)
        if np.random.random() < 0.1:  # 10% Chance
            dropout_ch = np.random.randint(0, augmented_data.shape[1])
            augmented_data[:, dropout_ch] *= 0.1  # Stark reduzieren statt nullsetzen

        # Events und Metadata kopieren
        augmented_events = original_events.copy()
        # Event-Zeiten leicht verschieben für Realismus
        augmented_events[:, 0] += len(original_data) * (aug_idx + 1)

        augmented_metadata = original_metadata.copy()

        augmented_data_list.append(augmented_data)
        augmented_events_list.append(augmented_events)
        augmented_metadata_list.append(augmented_metadata)

    # Kombiniere alle augmentierten Daten
    combined_data = np.concatenate(augmented_data_list, axis=0)
    combined_events = np.concatenate(augmented_events_list, axis=0)
    combined_metadata = pd.concat(augmented_metadata_list, ignore_index=True)

    # Erstelle neue Epochs mit augmentierten Daten
    augmented_epochs = mne.EpochsArray(
        combined_data,
        epochs.info,
        events=combined_events,
        tmin=epochs.tmin,
        event_id=epochs.event_id,
        metadata=combined_metadata,
        verbose=False,
    )

    print(f"Daten augmentiert: {len(epochs)} → {len(augmented_epochs)} Epochen")
    return augmented_epochs


def create_cv_splits(epochs, n_splits):
    targets = epochs.metadata["target"].values

    print(
        f"\nVerwende StratifiedKFold mit {n_splits} Splits (shuffle=True, random_state=42)"
    )

    # Zeige Klassenverteilung
    unique_targets, target_counts = np.unique(targets, return_counts=True)
    print("Gesamte Klassenverteilung:")
    for target, count in zip(unique_targets, target_counts):
        print(f"  Klasse {target}: {count} Epochen ({count / len(targets) * 100:.1f}%)")

    cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cv_splitter


def get_model(model_name, n_chans, n_classes, epoch_length_s, sfreq, config):
    if model_name == "EEGNet":
        return EEGNet(
            input_window_seconds=epoch_length_s,
            sfreq=sfreq,
            n_chans=n_chans,
            n_outputs=n_classes,
            final_conv_length="auto",
            pool_mode="mean",
            kernel_length=config["kernel_length"],
            F1=config["F1"],
            D=config["D"],
            F2=config["F2"],
            drop_prob=config["dropout_rate"],
        )
    
    elif model_name == "ShallowFBCSPNet":
        return ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_seconds=epoch_length_s,
            sfreq=sfreq,
            n_filters_time=40,
            n_filters_spat=40,
            final_conv_length='auto',
            drop_prob=config["dropout_rate"]
        )
    
    elif model_name == "ATCNet":
        return ATCNet(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_seconds=epoch_length_s,
            sfreq=sfreq
        )
    
    elif model_name == "AttentionBaseNet":
        return AttentionBaseNet(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_seconds=epoch_length_s,
            sfreq=sfreq,
        )
    
    else:
        raise ValueError(f"Unbekanntes Modell: {model_name}. Verfügbar: EEGNet, ShallowFBCSPNet, Deep4Net, ATCNet, EEGNetv4")


def train_eegnet_with_cv():
    config = {
        "model_name": "AttentionBaseNet",  # Optimiertes AttentionBaseNet
        "n_splits": 3,  # 3 Folds für Balance zwischen Genauigkeit und Geschwindigkeit
        
        # Optimierte Hyperparameter für AttentionBaseNet
        "lr": 0.0005,  # Niedrigere LR für stabileres Training bei Attention-Modellen
        "batch_size": 12,  # Kleinere Batch Size für bessere Gradientenqualität
        "max_epochs": 120,  # Mehr Epochen da Attention-Modelle langsamer konvergieren
        
        # Modell-spezifische Parameter
        "kernel_length": 32,  # Kürzere Kernel für feinere Features
        "F1": 24,  # Mehr Filter in erster Schicht
        "D": 6,   # Tiefere Depthwise Convolution
        "F2": 48, # Mehr Filter in zweiter Schicht
        
        # Regularisierung optimiert für Attention
        "dropout_rate": 0.5,  # Höherer Dropout da Attention-Modelle zu Overfitting neigen
        "weight_decay": 0.005,  # Stärkere L2-Regularisierung
        
        # Training-Optimierungen
        "patience": 25,  # Mehr Geduld für Attention-Konvergenz
        "augmentation_factor": 3,  # Mehr Datenaugmentation für robustere Features
        
        # Preprocessing optimiert für kognitive Aufgaben
        "filter_low": 0.5,   # Niedrigere untere Grenze für langsame Wellen
        "filter_high": 35.0, # Höhere obere Grenze für Gamma-Band
        
        # Zusätzliche Attention-spezifische Parameter
        "label_smoothing": 0.15,  # Höheres Label Smoothing
        "warmup_epochs": 10,      # Warmup für stabilere Attention-Gewichte
    }

    epochs, baseline_epochs = load_and_prepare_data()
    epochs = normalize_epochs_with_baseline(epochs, baseline_epochs, config)
    epochs = add_metadata_with_targets(epochs)

    sfreq = epochs.info["sfreq"]

    epoch_length_s = epochs.times[-1] - epochs.times[0]
    window_size_samples = len(epochs.times)

    window_stride_samples = window_size_samples

    print(f"Epochenlänge: {epoch_length_s:.2f}s ({window_size_samples} samples)")
    print(f"Sampling frequency: {sfreq}Hz")

    n_chans = epochs.info["nchan"]
    n_classes = len(epochs.event_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nVerwende Gerät: {device}")
    print(f"Anzahl Klassen: {n_classes}")
    print(f"Anzahl Kanäle: {n_chans}")

    n_splits = config["n_splits"]

    # Erstelle einfache StratifiedKFold Splits
    cv_splitter = create_cv_splits(epochs, n_splits)

    all_fold_accuracies, all_fold_f1_scores = [], []

    # Cross-validation durchführen
    for fold, (train_idx, test_idx) in enumerate(
        cv_splitter.split(X=epochs.get_data(), y=epochs.metadata["target"])
    ):
        print(f"\n--- Starte Fold {fold + 1}/{n_splits} ---")

        model = get_model(
            model_name=config["model_name"],
            n_chans=n_chans,
            n_classes=n_classes,
            epoch_length_s=epoch_length_s,
            sfreq=sfreq,
            config=config,
        )

        train_epochs = epochs[train_idx]
        test_epochs = epochs[test_idx]

        # Data Augmentation mit erhöhtem Faktor
        train_epochs_augmented = augment_eeg_data(train_epochs, augment_factor=config["augmentation_factor"])

        print(
            f"Train/Test Split: {len(train_epochs_augmented)}/{len(test_epochs)} Epochen (nach Augmentation)"
        )

        # Erstelle Datasets mit korrigiertem Windowing
        train_dataset = create_from_mne_epochs(
            [train_epochs_augmented],
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
            drop_last_window=False,
        )

        test_dataset = create_from_mne_epochs(
            [test_epochs],
            window_size_samples=window_size_samples,
            window_stride_samples=window_stride_samples,
            drop_last_window=False,
        )

        print(
            f"Trainingsdaten: {len(train_epochs_augmented)} Epochen -> {len(train_dataset)} Fenster"
        )
        print(
            f"Testdaten:      {len(test_epochs)} Epochen -> {len(test_dataset)} Fenster"
        )

        # Berechne Klassengewichte für unbalancierte Daten
        from sklearn.utils.class_weight import compute_class_weight

        unique_targets = np.unique(epochs.metadata["target"])
        class_weights = compute_class_weight(
            "balanced", classes=unique_targets, y=epochs.metadata["target"]
        )
        class_weight_dict = dict(zip(unique_targets, class_weights))
        print(f"Klassengewichte: {class_weight_dict}")

        clf = EEGClassifier(
            model,
            criterion=torch.nn.CrossEntropyLoss,
            criterion__weight=torch.FloatTensor(
                [class_weights[i] for i in range(len(class_weights))]
            ),
            criterion__label_smoothing=config["label_smoothing"],  # Optimiertes Label Smoothing
            optimizer=torch.optim.AdamW,
            optimizer__lr=config["lr"],
            optimizer__weight_decay=config["weight_decay"],
            optimizer__betas=(0.9, 0.999),  # Optimierte Adam Parameter
            optimizer__eps=1e-8,  # Numerische Stabilität
            batch_size=config["batch_size"],
            max_epochs=config["max_epochs"],
            train_split=predefined_split(test_dataset),
            device=device,
            classes=[0, 1, 2],
            callbacks=[
                (
                    "lr_scheduler",
                    LRScheduler("CosineAnnealingLR", T_max=config["max_epochs"] - 1),
                ),
            ],
            verbose=1,
        )

        print("Starte Training für diesen Fold...")
        clf.fit(train_dataset, y=None)

        y_true = [y for x, y, i in test_dataset]
        y_pred = clf.predict(test_dataset)

        # Analysiere Vorhersageverteilung
        pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
        true_unique, true_counts = np.unique(y_true, return_counts=True)

        print(f"Training beendet nach {len(clf.history)} Epochen.")
        print(
            f"\nTest-Set Klassenverteilung (Ground Truth): {dict(zip(true_unique, true_counts))}"
        )
        print(f"Vorhergesagte Klassenverteilung: {dict(zip(pred_unique, pred_counts))}")

        print("\nKonfusionsmatrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=["1-back", "2-back", "3-back"],
                zero_division=0,
            )
        )

        valid_acc = clf.history[-1, "valid_acc"]
        valid_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        all_fold_accuracies.append(valid_acc)
        all_fold_f1_scores.append(valid_f1)
        print(
            f"Fold {fold + 1} - Validation Accuracy: {valid_acc:.4f}, F1-Score: {valid_f1:.4f}"
        )

    # --- 3. Ergebnisse zusammenfassen ---
    mean_accuracy = np.mean(all_fold_accuracies)
    std_accuracy = np.std(all_fold_accuracies)

    print("\n\n--- Kreuzvalidierung abgeschlossen ---")
    print(
        f"Genauigkeiten der einzelnen Folds: {[f'{acc:.4f}' for acc in all_fold_accuracies]}"
    )
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
