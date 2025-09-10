import mne
import pandas as pd
from pathlib import Path

from braindecode import EEGClassifier
from braindecode.datasets import create_from_mne_epochs
from braindecode.models import EEGNet, ShallowFBCSPNet, ATCNet, AttentionBaseNet

from sklearn.preprocessing import StandardScaler
import numpy as np
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Import epoching functionality for dynamic epoching
from eeg_pipeline.epoching import EpochingConfig

mne.set_log_level("ERROR")


class FocalLoss(nn.Module):
    """Focal Loss für Class Imbalance - besonders effektiv gegen Majority Class Bias"""
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer('weight', weight)  # Register als buffer für automatisches device handling
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


def load_and_prepare_raw_data():
    """Lädt Raw-Daten statt vor-epochierter Daten um Data Leakage zu vermeiden."""
    base_dir = Path(__file__).parent.parent.parent.parent
    raw_path = (
        base_dir / "results" / "processed" / "Aliaa" / "indoor_processed_raw.fif"
    )

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    # Lade preprocessed Raw-Daten mit Annotationen
    raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose=False)
    
    # Prüfe verfügbare Annotationen
    available_annotations = set(raw.annotations.description)
    print(f"Available annotations: {available_annotations}")
    
    expected_conditions = {"baseline", "1-back", "2-back", "3-back"}
    missing_conditions = expected_conditions - available_annotations
    
    if missing_conditions:
        raise ValueError(
            f"Fehlende Annotationen: {missing_conditions}. "
            f"Verfügbare: {available_annotations}"
        )

    print(f"Raw data loaded: {raw.info['nchan']} channels, {raw.times[-1]:.1f}s duration")
    print(f"Annotations: {len(raw.annotations)} total")
    
    # Zähle Annotationen nach Typ
    for condition in expected_conditions:
        count = sum(1 for desc in raw.annotations.description if desc == condition)
        total_duration = sum(annot['duration'] for annot in raw.annotations if annot['description'] == condition)
        print(f"  {condition}: {count} blocks, {total_duration:.1f}s total")

    return raw


def create_epochs_for_split(raw, annotation_indices, use_overlap=True, segment_length=4.0, overlap=2.0):
    """Erstellt Epochen für spezifische Annotationen (für einen CV-Split).
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw-Daten mit Annotationen
    annotation_indices : list
        Liste der Annotation-Indices für diesen Split
    use_overlap : bool
        Ob Overlap verwendet werden soll (True für Training, False für Test)
    segment_length : float
        Länge der Segmente in Sekunden
    overlap : float
        Overlap in Sekunden (nur wenn use_overlap=True)
        
    Returns
    -------
    mne.Epochs
        Epochierte Daten mit Metadaten
    """
    # Erstelle temporäre Annotations nur für diesen Split
    temp_annotations = mne.Annotations(
        onset=[raw.annotations.onset[i] for i in annotation_indices],
        duration=[raw.annotations.duration[i] for i in annotation_indices], 
        description=[raw.annotations.description[i] for i in annotation_indices],
        orig_time=raw.annotations.orig_time
    )
    
    # Temporäres Raw-Objekt mit gefilterten Annotationen
    temp_raw = raw.copy()
    temp_raw.set_annotations(temp_annotations)
    
    # Epoching-Konfiguration
    actual_overlap = overlap if use_overlap else 0.0
    
    config = EpochingConfig(
        tmin=0.0,
        tmax=segment_length,
        baseline=None
    )
    
    print(f"  Creating epochs: {len(annotation_indices)} annotations, "
          f"segment_length={segment_length}s, overlap={actual_overlap}s")
    
    # Event-Generierung für alle Annotationen
    # Erstelle event_id dynamisch basierend auf vorhandenen Annotationen
    unique_descriptions = set(temp_raw.annotations.description)
    print(f"    Unique annotations in this split: {unique_descriptions}")
    
    # Basis event_id mapping
    all_event_ids = {
        'baseline': 0,
        '1-back': 1,
        '2-back': 2,
        '3-back': 3
    }
    
    # Filtere nur die tatsächlich vorhandenen Events
    event_id = {desc: event_id for desc, event_id in all_event_ids.items() 
                if desc in unique_descriptions}
    
    all_events = []
    
    for annot in temp_raw.annotations:
        description = annot['description']
        start_time = annot['onset']
        duration = annot['duration']
        
        if description not in event_id:
            print(f"    Warning: Unknown annotation '{description}', skipping")
            continue
            
        # Erstelle Events für diese Annotation mit oder ohne Overlap
        block_events = mne.make_fixed_length_events(
            temp_raw,
            id=event_id[description],
            start=start_time,
            stop=start_time + duration,
            duration=segment_length,
            overlap=actual_overlap
        )
        
        if len(block_events) > 0:
            all_events.append(block_events)
    
    if not all_events:
        print("    Warning: No valid events created")
        return None
        
    events = np.vstack(all_events)
    events = events[events[:, 0].argsort()]  # Sort by time
    
    # Erstelle Epochen
    epochs = mne.Epochs(
        temp_raw,
        events=events,
        event_id=event_id,
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=config.baseline,
        picks=config.picks,
        reject=config.reject,
        preload=True,
        verbose=False
    )
    
    print(f"    Created {len(epochs)} epochs from {len(annotation_indices)} annotations")
    
    return epochs


def create_simple_train_test_split(raw):
    """Erstellt einen einfachen Train/Test Split basierend auf den ersten/zweiten Phasen.
    
    Jede n-back Bedingung hat genau 2 Phasen:
    - Training: Erste Phase jeder Bedingung
    - Testing: Zweite Phase jeder Bedingung
    
    Returns
    -------
    tuple
        (train_annotation_indices, test_annotation_indices)
    """
    # Sammle alle Annotationen und gruppiere nach Task-Typ
    task_groups = {'1-back': [], '2-back': [], '3-back': []}
    baseline_indices = []
    
    for i, annot in enumerate(raw.annotations):
        desc = annot['description']
        if desc in task_groups:
            task_groups[desc].append((i, annot['onset']))
        elif desc == 'baseline':
            baseline_indices.append(i)
    
    print("Task annotation distribution:")
    for task, annotations in task_groups.items():
        print(f"  {task}: {len(annotations)} blocks")
    
    train_indices = []
    test_indices = []
    
    # Für jede Task-Kategorie: erste Phase -> Train, zweite Phase -> Test
    for task_type, annotations in task_groups.items():
        if len(annotations) != 2:
            raise ValueError(f"Erwarte genau 2 Phasen für {task_type}, gefunden: {len(annotations)}")
        
        # Sortiere nach Zeit (onset)
        annotations_sorted = sorted(annotations, key=lambda x: x[1])
        
        # Erste Phase -> Training, Zweite Phase -> Testing
        train_indices.append(annotations_sorted[0][0])  # Index der ersten Phase
        test_indices.append(annotations_sorted[1][0])   # Index der zweiten Phase
        
        print(f"  {task_type}: Phase 1 (Train) at {annotations_sorted[0][1]:.1f}s, Phase 2 (Test) at {annotations_sorted[1][1]:.1f}s")
    
    # Baseline immer zum Training hinzufügen
    train_indices.extend(baseline_indices)
    
    print(f"\nSplit: {len(train_indices)} train annotations ({len(baseline_indices)} baseline + {len(train_indices)-len(baseline_indices)} task), {len(test_indices)} test annotations")
    
    return sorted(train_indices), sorted(test_indices)


def add_metadata_with_targets(epochs):
    """Fügt Metadaten mit Targets hinzu, filtert Baseline-Epochen aus."""
    target_map = {"1-back": 0, "2-back": 1, "3-back": 2}

    # Filtere nur Task-Epochen (keine Baseline)
    task_conditions = list(target_map.keys())
    
    # Prüfe welche Task-Bedingungen in den Epochen vorhanden sind
    available_task_conditions = [cond for cond in task_conditions if cond in epochs.event_id]
    
    if not available_task_conditions:
        print("Warning: Keine Task-Bedingungen in den Epochen gefunden!")
        return None
    
    print(f"Verfügbare Task-Bedingungen: {available_task_conditions}")
    
    # Filtere Epochen: nur Task-Epochen behalten
    task_epochs = epochs[available_task_conditions]
    
    try:
        event_mapping = {
            original_id: target_map[condition]
            for condition, original_id in task_epochs.event_id.items()
            if condition in target_map
        }
    except KeyError as e:
        raise RuntimeError(
            f"Bedingung '{e.args[0]}' aus target_map nicht in epochs.event_id gefunden. "
            f"Verfügbare Bedingungen: {list(task_epochs.event_id.keys())}"
        )

    epochs_with_meta = task_epochs.copy()

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
    print(f"Gefilterte Epochen: {len(epochs)} -> {len(epochs_with_meta)} (nur Tasks)")
    print("Verwendetes direktes Mapping (Original-ID -> Target):", event_mapping)
    print("Events wurden auf 0, 1, 2 remapped")
    if len(epochs_with_meta.metadata) > 0:
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
        raise ValueError(f"Unbekanntes Modell: {model_name}. Verfügbar: EEGNet, ShallowFBCSPNet, ATCNet, AttentionBaseNet")


def train_eegnet_simplified():
    """Vereinfachtes Training mit Focal Loss und ShallowFBCSPNet für bessere EEG-Performance."""
    config = {
        "model_name": "ATCNet",  # Bewährtes Modell für EEG
        
        # Optimierte Hyperparameter für ShallowFBCSP + Focal Loss
        "lr": 0.0005,  # Höher als EEGNet, aber kontrolliert
        "batch_size": 16,
        "max_epochs": 20,  # Mehr Epochen für bessere Konvergenz
        
        # ShallowFBCSPNet-spezifische Parameter
        "kernel_length": 64,
        "F1": 8,
        "D": 2,
        "F2": 16,
        
        # Optimierte Regularisierung
        "dropout_rate": 0.35,
        "weight_decay": 0.01,
        
        # Training-Optimierungen
        "patience": 25,
        "augmentation_factor": 1,
        
        # Standard Preprocessing
        "filter_low": 1.0,
        "filter_high": 40.0, 
        
        # Focal Loss Configuration
        "use_focal_loss": True,
        "focal_alpha": 1.0,
        "focal_gamma": 2.0,
        "label_smoothing": 0.0,  # Deaktiviert bei Focal Loss
        "use_manual_class_weights": False,  # Focal Loss handhabt Balance
    }

    # Lade Raw-Daten
    print("Loading raw data...")
    raw = load_and_prepare_raw_data()
    
    # Erstelle einfachen Train/Test Split basierend auf ersten/zweiten Phasen
    print("\nCreating simple train/test split...")
    train_annotation_indices, test_annotation_indices = create_simple_train_test_split(raw)
    
    # Baseline-Epochen für Normalisierung
    print("\nCreating baseline epochs for normalization...")
    baseline_annotation_indices = [i for i, annot in enumerate(raw.annotations) 
                                   if annot['description'] == 'baseline']
    baseline_epochs = create_epochs_for_split(raw, baseline_annotation_indices, use_overlap=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Epoching für Training und Test
    print("\nCreating training epochs (with overlap)...")
    train_epochs = create_epochs_for_split(
        raw, train_annotation_indices, use_overlap=True, segment_length=4.0, overlap=2.0
    )
    
    print("Creating test epochs (without overlap)...")
    test_epochs = create_epochs_for_split(
        raw, test_annotation_indices, use_overlap=False, segment_length=4.0, overlap=0.0
    )
    
    if train_epochs is None or test_epochs is None:
        raise RuntimeError("Failed to create epochs")
        
    # Normalisierung mit Baseline
    train_epochs = normalize_epochs_with_baseline(train_epochs, baseline_epochs, config)
    test_epochs = normalize_epochs_with_baseline(test_epochs, baseline_epochs, config)
    
    # Metadaten und Targets hinzufügen
    train_epochs = add_metadata_with_targets(train_epochs)
    test_epochs = add_metadata_with_targets(test_epochs)
    
    # Datenaugmentation nur auf Training-Daten (nur wenn Faktor > 1)
    if config["augmentation_factor"] > 1:
        train_epochs_augmented = augment_eeg_data(train_epochs, augment_factor=config["augmentation_factor"])
        print(f"Data augmented: {len(train_epochs)} -> {len(train_epochs_augmented)} epochs")
    else:
        train_epochs_augmented = train_epochs
        print(f"No augmentation applied: {len(train_epochs_augmented)} training epochs")
    
    print(f"\nFinal Train/Test Split: {len(train_epochs_augmented)}/{len(test_epochs)} epochs")
    
    # Model-Parameter
    sfreq = train_epochs.info["sfreq"] 
    epoch_length_s = train_epochs.times[-1] - train_epochs.times[0]
    window_size_samples = len(train_epochs.times)
    window_stride_samples = window_size_samples
    n_chans = train_epochs.info["nchan"]
    n_classes = len(train_epochs.event_id)
    
    print(f"Epoch length: {epoch_length_s:.2f}s ({window_size_samples} samples)")
    print(f"Sampling frequency: {sfreq}Hz")
    print(f"Channels: {n_chans}, Classes: {n_classes}")
    
    # Model erstellen
    model = get_model(
        model_name=config["model_name"],
        n_chans=n_chans,
        n_classes=n_classes,
        epoch_length_s=epoch_length_s,
        sfreq=sfreq,
        config=config,
    )
    
    # Datasets erstellen
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

    print(f"Training data: {len(train_epochs_augmented)} epochs -> {len(train_dataset)} windows")
    print(f"Test data: {len(test_epochs)} epochs -> {len(test_dataset)} windows")

    # Klassengewichte berechnen
    from sklearn.utils.class_weight import compute_class_weight
    
    # Zeige Training-Daten-Verteilung für Debugging
    train_target_counts = train_epochs_augmented.metadata["target"].value_counts().sort_index()
    print(f"Training data class distribution: {dict(train_target_counts)}")
    
    unique_targets = np.unique(train_epochs.metadata["target"])  # Nutze ursprüngliche Daten für Balance
    
    if config.get("use_manual_class_weights", False):
        # Manuelle gleichmäßige Gewichtung
        class_weights = np.ones(len(unique_targets))
        print("Using manual equal class weights: [1.0, 1.0, 1.0]")
    else:
        # Automatische Berechnung
        class_weights = compute_class_weight(
            "balanced", classes=unique_targets, y=train_epochs.metadata["target"]
        )
        print(f"Using computed balanced class weights: {class_weights}")
    
    class_weight_dict = dict(zip(unique_targets, class_weights))
    print(f"Final class weight mapping: {class_weight_dict}")

    # Loss Function auswählen
    if config.get("use_focal_loss", False):
        criterion = FocalLoss(
            alpha=config["focal_alpha"],
            gamma=config["focal_gamma"],
            weight=torch.FloatTensor([class_weights[i] for i in range(len(class_weights))])
        )
        criterion_params = {}
        print(f"🎯 Using Focal Loss (alpha={config['focal_alpha']}, gamma={config['focal_gamma']}) for Class Imbalance")
    else:
        criterion = torch.nn.CrossEntropyLoss
        criterion_params = {
            "weight": torch.FloatTensor([class_weights[i] for i in range(len(class_weights))]),
            "label_smoothing": config["label_smoothing"]
        }
        print("Using standard CrossEntropy Loss")

    # Classifier erstellen und trainieren
    clf = EEGClassifier(
        model,
        criterion=criterion,
        **{f"criterion__{k}": v for k, v in criterion_params.items()},
        optimizer=torch.optim.AdamW,
        optimizer__lr=config["lr"],
        optimizer__weight_decay=config["weight_decay"],
        optimizer__betas=(0.9, 0.999),
        optimizer__eps=1e-8,
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

    print("\n🚀 Starting training with ADVANCED configuration...")
    print(f"Model: {config['model_name']}")  
    print(f"Loss: {'Focal Loss' if config.get('use_focal_loss') else 'CrossEntropy'}")
    print(f"LR: {config['lr']}, Batch Size: {config['batch_size']}, Epochs: {config['max_epochs']}")
    print(f"Augmentation: {config['augmentation_factor']}x")
    
    clf.fit(train_dataset, y=None)

    # Evaluation
    y_true = [y for x, y, i in test_dataset]
    y_pred = clf.predict(test_dataset)

    # Analysiere Vorhersageverteilung
    pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
    true_unique, true_counts = np.unique(y_true, return_counts=True)

    print(f"\nTraining completed after {len(clf.history)} epochs.")
    print(f"Test-Set class distribution (Ground Truth): {dict(zip(true_unique, true_counts))}")
    print(f"Predicted class distribution: {dict(zip(pred_unique, pred_counts))}")

    print("\nConfusion Matrix:")
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

    final_acc = clf.history[-1, "valid_acc"]
    final_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n--- Final Results (Phase-based Train/Test Split) ---")
    print(f"Test Accuracy: {final_acc:.4f}")
    print(f"Test F1-Score: {final_f1:.4f}")
    print("\nNote: This approach uses first phases for training, second phases for testing.")
    print("This eliminates data leakage while maintaining temporal structure.")

    return final_acc, final_f1


if __name__ == "__main__":
    try:
        accuracy, f1_score_result = train_eegnet_simplified()
        print("\nTraining completed successfully!")
        print(f"Final Test Accuracy: {accuracy:.4f}")
        print(f"Final Test F1-Score: {f1_score_result:.4f}")
    except Exception as e:
        print(f"\nEin Fehler ist aufgetreten: {e}")
        import traceback

        traceback.print_exc()
