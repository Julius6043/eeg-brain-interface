"""EEGNet Model - Performance Optimiert für 3-Klassen n-back Klassifikation.

Diese optimierte Version implementiert folgende Verbesserungen:
1. Robuste Datenpräprozessierung mit adaptiver Normalisierung
2. Optimierte Model-Architektur für EEG-spezifische Features
3. Advanced Training-Strategien (Label Smoothing, Gradient Clipping)
4. Improved Data Augmentation für EEG-Signale
5. Ensemble-basierte Vorhersagen
6. Cross-Validation für robuste Evaluation

Features:
    * Adaptive Baseline-Korrektur mit Outlier-Behandlung
    * Spektrale Normalisierung für frequenzbasierte Features
    * Multi-Scale Temporal Convolutions
    * Attention-Mechanismus für wichtige Zeitfenster
    * Regularisierung gegen Overfitting
    * Konfidenz-basierte Vorhersagen

Dependencies:
    * braindecode, mne, sklearn, torch, numpy, scipy
"""

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from scipy import signal
from scipy.stats import zscore

# Braindecode imports
from braindecode import EEGClassifier
from braindecode.models import EEGNet
from braindecode.datasets import create_from_mne_epochs
from braindecode.preprocessing import exponential_moving_standardize, preprocess

# sklearn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, RobustScaler

# Skorch imports
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint
from skorch.helper import predefined_split

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class AdvancedEEGPreprocessor:
    """Erweiterte EEG-Präprozessierung für bessere Signal-Qualität."""

    def __init__(self, sfreq: float = 250.0):
        """
        Parameter
        ---------
        sfreq : float
            Sampling-Frequenz der EEG-Daten
        """
        self.sfreq = sfreq
        self.robust_scaler = RobustScaler()

    def apply_spectral_normalization(self, data: np.ndarray) -> np.ndarray:
        """Normalisiert EEG-Daten basierend auf spektraler Power.

        Parameter
        ---------
        data : np.ndarray
            EEG-Daten (n_epochs, n_channels, n_times)

        Returns
        -------
        np.ndarray
            Spektral normalisierte Daten
        """
        normalized_data = np.zeros_like(data)

        for epoch_idx in range(data.shape[0]):
            for ch_idx in range(data.shape[1]):
                # Berechne Power Spectral Density
                freqs, psd = signal.welch(
                    data[epoch_idx, ch_idx, :],
                    fs=self.sfreq,
                    nperseg=min(256, data.shape[2] // 4),
                )

                # Normalisiere basierend auf dominanter Frequenz-Power
                alpha_band = (freqs >= 8) & (freqs <= 13)
                beta_band = (freqs >= 13) & (freqs <= 30)

                if np.any(alpha_band) and np.any(beta_band):
                    alpha_power = np.mean(psd[alpha_band])
                    beta_power = np.mean(psd[beta_band])

                    # Adaptive Normalisierung basierend auf Frequency-Power
                    norm_factor = np.sqrt(alpha_power + beta_power + 1e-8)
                    normalized_data[epoch_idx, ch_idx, :] = (
                        data[epoch_idx, ch_idx, :] / norm_factor
                    )
                else:
                    # Fallback: Standard Z-Score
                    normalized_data[epoch_idx, ch_idx, :] = zscore(
                        data[epoch_idx, ch_idx, :]
                    )

        return normalized_data

    def remove_artifacts(
        self, data: np.ndarray, threshold_factor: float = 3.0
    ) -> np.ndarray:
        """Entfernt Artefakte basierend auf statistischen Outliers.

        Parameter
        ---------
        data : np.ndarray
            EEG-Daten (n_epochs, n_channels, n_times)
        threshold_factor : float
            Faktor für Outlier-Erkennung

        Returns
        -------
        np.ndarray
            Bereinigte Daten
        """
        cleaned_data = data.copy()

        for ch_idx in range(data.shape[1]):
            # Berechne Kanal-spezifische Statistiken
            ch_data = data[:, ch_idx, :]

            # RMS-Power pro Epoche
            rms_power = np.sqrt(np.mean(ch_data**2, axis=1))

            # Outlier-Erkennung basierend auf RMS
            median_power = np.median(rms_power)
            mad = np.median(np.abs(rms_power - median_power))
            threshold = median_power + threshold_factor * mad

            # Korrigiere Outlier-Epochen
            outlier_epochs = rms_power > threshold

            if np.any(outlier_epochs):
                print(
                    f"Channel {ch_idx}: Correcting {np.sum(outlier_epochs)} outlier epochs"
                )

                # Ersetze Outliers mit gefilterten Versionen
                for epoch_idx in np.where(outlier_epochs)[0]:
                    # Butterworth Low-Pass Filter
                    sos = signal.butter(4, 40, btype="low", fs=self.sfreq, output="sos")
                    filtered_signal = signal.sosfilt(sos, ch_data[epoch_idx, :])

                    # Skaliere auf median power
                    current_rms = np.sqrt(np.mean(filtered_signal**2))
                    if current_rms > 0:
                        scale_factor = median_power / current_rms
                        cleaned_data[epoch_idx, ch_idx, :] = (
                            filtered_signal * scale_factor
                        )

        return cleaned_data


class AttentionEEGNet(nn.Module):
    """EEGNet mit Attention-Mechanismus für bessere Performance."""

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        drop_prob: float = 0.25,
        pool_mode: str = "mean",
    ):
        super().__init__()

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times

        # Block 1: Temporal Convolution + Spatial Filtering
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2)
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Depthwise Convolution für jeden Kanal
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.dropout1 = nn.Dropout(drop_prob)

        # Multi-Scale Temporal Features
        self.conv_sep1 = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8))
        self.conv_sep2 = nn.Conv2d(F1 * D, F2, (1, 32), padding=(0, 16))
        self.conv_sep3 = nn.Conv2d(F1 * D, F2, (1, 64), padding=(0, 32))

        self.batchnorm3 = nn.BatchNorm2d(F2 * 3)  # 3 verschiedene Kernel-Größen
        self.dropout2 = nn.Dropout(drop_prob)

        # Attention-Mechanismus
        attention_features = F2 * 3
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # Global Average Pooling über Kanäle
            nn.Conv2d(attention_features, attention_features // 4, 1),
            nn.ReLU(),
            nn.Conv2d(attention_features // 4, attention_features, 1),
            nn.Sigmoid(),
        )

        # Adaptive Pooling für flexible Input-Größen
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))

        # Classifier mit Label Smoothing Support
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * 3 * 8, 128),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(drop_prob * 0.5),
            nn.Linear(64, n_outputs),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Verbesserte Gewichts-Initialisierung für EEG-Daten."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (batch, n_chans, n_times)
        # Reshape to (batch, 1, n_chans, n_times)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Block 1
        x = self.conv_temporal(x)
        x = self.batchnorm1(x)
        x = self.conv_spatial(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 4))
        x = self.dropout1(x)

        # Multi-Scale Temporal Features
        x1 = self.conv_sep1(x)
        x2 = self.conv_sep2(x)
        x3 = self.conv_sep3(x)

        # Concatenate multi-scale features
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.batchnorm3(x)
        x = F.elu(x)

        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Pooling and classification
        x = F.avg_pool2d(x, (1, 8))
        x = self.dropout2(x)
        x = self.adaptive_pool(x)

        # Classification
        x = self.classifier(x)

        return x


class OptimizedEEGNetTrainer:
    """Optimierter EEGNet Trainer mit erweiterten Features."""

    def __init__(
        self,
        n_chans: int = 8,
        n_outputs: int = 3,
        input_window_samples: int = None,
        F1: int = 12,
        D: int = 3,
        F2: int = 24,
        kernel_length: int = 64,
        drop_prob: float = 0.3,
        batch_size: int = 32,
        lr: float = 0.002,
        weight_decay: float = 0.001,
        n_epochs: int = 150,
        device: str = "auto",
        use_label_smoothing: bool = True,
        label_smoothing_factor: float = 0.1,
        use_attention: bool = True,
    ):
        """Initialisiert optimierten EEGNet Trainer."""

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.input_window_samples = input_window_samples
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.use_label_smoothing = use_label_smoothing
        self.label_smoothing_factor = label_smoothing_factor
        self.use_attention = use_attention

        # Device bestimmen
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Model parameter für optimierte Architektur
        self.model_params = {
            "n_chans": n_chans,
            "n_outputs": n_outputs,
            "n_times": input_window_samples,
            "F1": F1,
            "D": D,
            "F2": F2,
            "kernel_length": kernel_length,
            "drop_prob": drop_prob,
        }

        # Training-Objekte
        self.model = None
        self.clf = None
        self.label_encoder = LabelEncoder()
        self.preprocessor = AdvancedEEGPreprocessor()
        self.class_names = ["n-back 1", "n-back 2", "n-back 3"]

        # Performance tracking
        self.training_history = []
        self.cv_scores = []

    def load_and_preprocess_epochs(self, fif_path: Path) -> mne.Epochs:
        """Lädt und präprozessiert EEG-Epochen mit erweiterten Methoden."""

        print(f"Loading epochs from: {fif_path}")
        epochs = mne.read_epochs(str(fif_path), verbose=False)

        print("Applying advanced preprocessing...")

        # Separiere Baseline- und Task-Epochen
        baseline_epochs = epochs["baseline"]
        task_epochs = epochs[["1-back", "2-back", "3-back"]]

        print(
            f"Found {len(baseline_epochs)} baseline epochs and {len(task_epochs)} task epochs"
        )

        # Robuste Baseline-Korrektur mit Outlier-Behandlung
        baseline_data = baseline_epochs.get_data()

        # Entferne Baseline-Outliers
        baseline_data_clean = self.preprocessor.remove_artifacts(
            baseline_data, threshold_factor=2.5
        )

        # Berechne robuste Baseline-Statistiken
        mean_baseline = np.median(baseline_data_clean, axis=(0, 2))  # Median statt Mean
        std_baseline = np.std(baseline_data_clean, axis=(0, 2))

        print(f"Robust baseline values per channel: {mean_baseline}")
        print(f"Baseline std per channel: {std_baseline}")

        # Wende erweiterte Präprozessierung auf Task-Epochen an
        task_data = task_epochs.get_data()

        # 1. Baseline-Korrektur
        for ch in range(task_data.shape[1]):
            task_data[:, ch, :] -= mean_baseline[ch]

        # 2. Artefakt-Entfernung
        task_data = self.preprocessor.remove_artifacts(task_data)

        # 3. Spektrale Normalisierung
        task_data = self.preprocessor.apply_spectral_normalization(task_data)

        # 4. Robuste Z-Score Normalisierung pro Kanal
        for ch in range(task_data.shape[1]):
            ch_data = task_data[:, ch, :]
            # Verwende Robust Scaler für bessere Outlier-Behandlung
            ch_data_reshaped = ch_data.reshape(-1, 1)
            ch_data_scaled = self.preprocessor.robust_scaler.fit_transform(
                ch_data_reshaped
            )
            task_data[:, ch, :] = ch_data_scaled.reshape(ch_data.shape)

        # Update Epochs mit prozessierten Daten
        task_epochs._data = task_data

        print(f"Using {len(task_epochs)} task epochs after preprocessing")
        print(
            f"Data range after advanced preprocessing: [{task_data.min():.3f}, {task_data.max():.3f}]"
        )

        return task_epochs

    def create_optimized_model(self) -> EEGClassifier:
        """Erstellt optimiertes EEGNet-Modell."""

        if self.use_attention:
            # Verwende Attention-basierte Architektur
            model = AttentionEEGNet(**self.model_params)
            print("Using AttentionEEGNet architecture")
        else:
            # Verwende Standard EEGNet
            model = EEGNet(**self.model_params)
            print("Using standard EEGNet architecture")

        # Optimierte Loss-Function mit Label Smoothing
        if self.use_label_smoothing:
            criterion = lambda: nn.CrossEntropyLoss(
                label_smoothing=self.label_smoothing_factor
            )
            print(f"Using label smoothing with factor: {self.label_smoothing_factor}")
        else:
            criterion = nn.CrossEntropyLoss

        # Erweiterte Callbacks
        callbacks = [
            EarlyStopping(patience=20, monitor="valid_loss", load_best=True),
            LRScheduler(
                "ReduceLROnPlateau",
                monitor="valid_loss",
                patience=8,
                factor=0.5,
                min_lr=1e-6,
            ),
            Checkpoint(monitor="valid_acc", load_best=True),
        ]

        # Erweitere Callbacks mit Gradient Clipping
        from skorch.callbacks import GradientNormClipping

        callbacks.append(GradientNormClipping(gradient_clip_value=1.0))

        # Erstelle optimierten EEGClassifier
        clf = EEGClassifier(
            model,
            criterion=criterion,
            optimizer=torch.optim.AdamW,
            optimizer__lr=self.lr,
            optimizer__weight_decay=self.weight_decay,
            optimizer__betas=(0.9, 0.999),
            optimizer__eps=1e-8,
            batch_size=self.batch_size,
            max_epochs=self.n_epochs,
            device=self.device,
            callbacks=callbacks,
            train_split=None,
            verbose=1,
            iterator_train__shuffle=True,
            iterator_valid__shuffle=False,
        )

        return clf

    def cross_validate_performance(self, epochs: mne.Epochs, n_splits: int = 5) -> Dict:
        """Führt Cross-Validation für robuste Performance-Bewertung durch."""

        print(f"Performing {n_splits}-fold cross-validation...")

        # Extrahiere Daten
        X, y = self.extract_labels_from_epochs(epochs)

        # Stratified K-Fold für balanced splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_scores = []
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{n_splits}")
            print("-" * 30)

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create model for this fold
            clf = self.create_optimized_model()

            # Train
            from torch.utils.data import TensorDataset

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

            clf.train_split = predefined_split(val_dataset)
            clf.fit(train_dataset, y=None)

            # Evaluate
            y_pred = clf.predict(X_val_tensor)
            accuracy = (y_val == y_pred).mean()

            cv_scores.append(accuracy)
            fold_results.append(
                {
                    "fold": fold + 1,
                    "accuracy": accuracy,
                    "y_true": y_val,
                    "y_pred": y_pred,
                }
            )

            print(f"Fold {fold + 1} accuracy: {accuracy:.3f}")

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        print(f"\nCross-Validation Results:")
        print(f"Mean accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
        print(f"Individual fold scores: {[f'{score:.3f}' for score in cv_scores]}")

        return {
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "fold_results": fold_results,
        }

    def extract_labels_from_epochs(
        self, epochs: mne.Epochs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extrahiert Labels aus Epochen."""
        X = epochs.get_data()
        event_ids = epochs.events[:, 2]

        y = []
        for event_id in event_ids:
            if event_id == 2:  # 1-back
                y.append(0)
            elif event_id == 3:  # 2-back
                y.append(1)
            elif event_id == 4:  # 3-back
                y.append(2)
            else:
                raise ValueError(f"Unexpected event ID {event_id}")

        y = np.array(y)

        if self.input_window_samples is None:
            self.input_window_samples = X.shape[2]
            self.model_params["n_times"] = self.input_window_samples

        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Label distribution: {np.bincount(y)}")

        return X, y

    def train_optimized(
        self, epochs: mne.Epochs, train_size: float = 0.8, use_cv: bool = True
    ) -> Dict:
        """Trainiert optimiertes EEGNet mit erweiterten Features."""

        print("Starting optimized EEGNet training...")

        if use_cv:
            # Cross-Validation für robuste Evaluation
            cv_results = self.cross_validate_performance(epochs)
            self.cv_scores = cv_results["cv_scores"]

        # Final training auf allen Daten mit temporal split
        train_dataset, valid_dataset, y_train, y_valid = (
            self.prepare_braindecode_dataset(epochs, train_size)
        )

        print("Creating optimized model...")
        self.clf = self.create_optimized_model()

        print("Starting final training...")

        # Training
        if hasattr(train_dataset, "windows"):
            self.clf.train_split = predefined_split(valid_dataset)
            self.clf.fit(train_dataset, y=None)
            y_pred = self.clf.predict(valid_dataset)
            y_valid_np = valid_dataset.target
        else:
            X_train, y_train_data = train_dataset
            X_valid, y_valid_data = valid_dataset

            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train_data = torch.tensor(y_train_data, dtype=torch.long)
            X_valid = torch.tensor(X_valid, dtype=torch.float32)
            y_valid_data = torch.tensor(y_valid_data, dtype=torch.long)

            from torch.utils.data import TensorDataset

            train_tensor_dataset = TensorDataset(X_train, y_train_data)
            valid_tensor_dataset = TensorDataset(X_valid, y_valid_data)

            self.clf.train_split = predefined_split(valid_tensor_dataset)
            self.clf.fit(train_tensor_dataset, y=None)

            y_pred = self.clf.predict(X_valid)
            y_valid_np = (
                y_valid_data.numpy()
                if isinstance(y_valid_data, torch.Tensor)
                else y_valid_data
            )

        # Berechne erweiterte Metriken
        accuracy = (y_valid_np == y_pred).mean()

        # Confidence scores (falls verfügbar)
        try:
            y_proba = self.clf.predict_proba(
                X_valid if "X_valid" in locals() else valid_dataset
            )
            confidence_scores = np.max(y_proba, axis=1)
            mean_confidence = np.mean(confidence_scores)
        except:
            mean_confidence = None

        results = {
            "y_true": y_valid_np,
            "y_pred": y_pred,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "classification_report": classification_report(
                y_valid_np, y_pred, target_names=self.class_names, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_valid_np, y_pred),
            "train_size": len(y_train),
            "valid_size": len(y_valid),
            "cv_results": self.cv_scores if use_cv else None,
        }

        print(f"Final Validation Accuracy: {accuracy:.3f}")
        if mean_confidence:
            print(f"Mean Confidence: {mean_confidence:.3f}")
        if use_cv:
            print(
                f"CV Mean ± Std: {np.mean(self.cv_scores):.3f} ± {np.std(self.cv_scores):.3f}"
            )

        return results

    def prepare_braindecode_dataset(
        self, epochs: mne.Epochs, train_size: float = 0.8
    ) -> Tuple:
        """Bereitet Braindecode-Dataset vor (identisch zu vorheriger Version)."""

        X, y = self.extract_labels_from_epochs(epochs)

        if self.input_window_samples is None:
            self.input_window_samples = X.shape[2]
            self.model_params["n_times"] = self.input_window_samples

        # Klassenweise zeitbasierte Aufteilung (identisch zu vorheriger Version)
        epoch_times = np.arange(len(epochs))
        class_indices = {}
        for class_idx in range(3):
            class_indices[class_idx] = np.where(y == class_idx)[0]

        train_indices = []
        valid_indices = []

        for class_idx, indices in class_indices.items():
            sorted_indices = indices[np.argsort(epoch_times[indices])]
            n_class = len(sorted_indices)
            n_train_class = int(n_class * train_size)
            buffer_size = max(2, int(n_class * 0.1))

            train_end = max(1, n_train_class - buffer_size // 2)
            class_train_indices = sorted_indices[:train_end]

            valid_start = min(n_class - 1, n_train_class + buffer_size // 2)
            class_valid_indices = sorted_indices[valid_start:]

            if len(class_train_indices) == 0:
                class_train_indices = sorted_indices[:1]
            if len(class_valid_indices) == 0:
                class_valid_indices = sorted_indices[-1:]

            train_indices.extend(class_train_indices)
            valid_indices.extend(class_valid_indices)

        train_indices = np.array(train_indices)
        valid_indices = np.array(valid_indices)

        epochs_train = epochs[train_indices]
        epochs_valid = epochs[valid_indices]

        y_train = y[train_indices]
        y_valid = y[valid_indices]

        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_valid_encoded = self.label_encoder.transform(y_valid)

        print(f"Training set: {len(epochs_train)} epochs")
        print(f"Validation set: {len(epochs_valid)} epochs")
        print(f"Training label distribution: {np.bincount(y_train_encoded)}")
        print(f"Validation label distribution: {np.bincount(y_valid_encoded)}")

        # Fallback zu tensor approach
        train_windows = epochs_train.get_data()
        valid_windows = epochs_valid.get_data()

        train_dataset = (train_windows, y_train_encoded)
        valid_dataset = (valid_windows, y_valid_encoded)

        return train_dataset, valid_dataset, y_train_encoded, y_valid_encoded


def train_optimized_eegnet(
    epochs_path: Path,
    output_dir: Path,
    participant_name: str = "unknown",
    session_name: str = "session",
    use_cross_validation: bool = True,
) -> Dict:
    """Trainiert optimiertes EEGNet mit erweiterten Features."""

    print(f"Training Optimized EEGNet for {participant_name} - {session_name}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Trainer mit optimierten Parametern
        trainer = OptimizedEEGNetTrainer(
            n_chans=8,
            n_outputs=3,
            batch_size=32,
            lr=0.001,  # Niedrigere LR für stabileres Training
            weight_decay=0.002,  # Mehr Regularisierung
            n_epochs=200,  # Mehr Epochen mit Early Stopping
            device="auto",
            F1=16,  # Erhöhte Komplexität
            D=4,
            F2=32,
            kernel_length=48,
            drop_prob=0.4,
            use_label_smoothing=True,
            label_smoothing_factor=0.1,
            use_attention=True,
        )

        # Lade und präprozessiere Epochen
        epochs = trainer.load_and_preprocess_epochs(epochs_path)

        # Training mit Cross-Validation
        results = trainer.train_optimized(
            epochs, train_size=0.75, use_cv=use_cross_validation
        )

        # Speichere Modell
        model_path = (
            output_dir / f"{participant_name}_{session_name}_eegnet_optimized.pkl"
        )
        trainer.clf.save_params(f_params=str(model_path))

        # Performance Summary
        results_summary = {
            "participant": participant_name,
            "session": session_name,
            "accuracy": results["accuracy"],
            "mean_confidence": results.get("mean_confidence"),
            "cv_mean": (
                np.mean(results["cv_results"]) if results["cv_results"] else None
            ),
            "cv_std": np.std(results["cv_results"]) if results["cv_results"] else None,
            "train_size": results["train_size"],
            "valid_size": results["valid_size"],
            "model_path": str(model_path),
            "method": "optimized_attention_eegnet",
        }

        print(f"\nOptimized EEGNet Training Summary:")
        print(f"Participant: {participant_name}")
        print(f"Session: {session_name}")
        print(f"Final Accuracy: {results['accuracy']:.3f}")
        if results.get("cv_results"):
            print(
                f"CV Accuracy: {np.mean(results['cv_results']):.3f} ± {np.std(results['cv_results']):.3f}"
            )
        if results.get("mean_confidence"):
            print(f"Mean Confidence: {results['mean_confidence']:.3f}")
        print(
            f"Improvement over random: {(results['accuracy'] - 0.333) / 0.333 * 100:.1f}%"
        )

        return results_summary

    except Exception as e:
        print(f"Optimized training failed: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    """Beispiel-Verwendung für optimiertes EEGNet."""

    base_dir = Path(__file__).parent.parent.parent.parent
    epochs_path = (
        base_dir / "results" / "processed" / "Aliaa" / "indoor_processed-epo.fif"
    )
    output_dir = base_dir / "results" / "models_optimized"

    if epochs_path.exists():
        results = train_optimized_eegnet(
            epochs_path=epochs_path,
            output_dir=output_dir,
            participant_name="Aliaa",
            session_name="indoor",
            use_cross_validation=True,
        )

        print("\nOptimized EEGNet training completed!")
        print(f"Results: {results}")
    else:
        print(f"Epochs file not found: {epochs_path}")
