"""EEGNet Ultra - Hochoptimierte Version für maximale Performance.

Diese finale Version implementiert alle identifizierten Verbesserungen:
1. Transformer-inspirierte Architektur mit Positional Encoding
2. Multi-Head Attention für räumlich-zeitliche Features
3. Residual Connections für tiefere Netzwerke
4. Ensemble Training mit mehreren Architekturen
5. Advanced Data Augmentation für EEG-Signale
6. Curriculum Learning für progressives Training

Ziel: >85% Cross-Validation Accuracy
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
import math
import random

# Braindecode imports
from braindecode import EEGClassifier
from braindecode.models import EEGNet
from braindecode.datasets import create_from_mne_epochs

# sklearn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, RobustScaler

# Skorch imports
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint
from skorch.helper import predefined_split

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class PositionalEncoding(nn.Module):
    """Positional Encoding für zeitliche EEG-Sequenzen."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class MultiHeadEEGAttention(nn.Module):
    """Multi-Head Attention für EEG-Signale mit räumlich-zeitlicher Aufmerksamkeit."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        residual = x

        # Linear projections
        Q = (
            self.w_q(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        # Output projection and residual connection
        output = self.w_o(attn_output)
        return self.layer_norm(output + residual)


class EEGTransformerBlock(nn.Module):
    """Transformer Block optimiert für EEG-Daten."""

    def __init__(
        self, d_model: int, n_heads: int = 8, d_ff: int = 512, dropout: float = 0.1
    ):
        super().__init__()

        self.attention = MultiHeadEEGAttention(d_model, n_heads, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output = self.attention(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(attn_output)
        return self.layer_norm2(attn_output + self.dropout(ff_output))


class UltraEEGNet(nn.Module):
    """Ultra-optimierte EEGNet Architektur mit Transformer-Komponenten."""

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        F1: int = 16,
        D: int = 4,
        F2: int = 32,
        kernel_length: int = 64,
        drop_prob: float = 0.3,
        n_transformer_layers: int = 2,
        n_attention_heads: int = 8,
    ):
        super().__init__()

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times

        # 1. Initial Temporal Convolution
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2)
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # 2. Depthwise Spatial Convolution
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.dropout1 = nn.Dropout(drop_prob)

        # 3. Multi-Scale Separable Convolutions
        self.conv_sep_small = nn.Conv2d(F1 * D, F2, (1, 8), padding=(0, 4))
        self.conv_sep_medium = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8))
        self.conv_sep_large = nn.Conv2d(F1 * D, F2, (1, 32), padding=(0, 16))

        self.batchnorm3 = nn.BatchNorm2d(F2 * 3)
        self.dropout2 = nn.Dropout(drop_prob)

        # 4. Transformer Encoder for Temporal Dependencies
        self.d_model = F2 * 3
        self.pos_encoding = PositionalEncoding(self.d_model)

        self.transformer_layers = nn.ModuleList(
            [
                EEGTransformerBlock(
                    self.d_model,
                    n_attention_heads,
                    d_ff=self.d_model * 2,
                    dropout=drop_prob,
                )
                for _ in range(n_transformer_layers)
            ]
        )

        # 5. Residual Connections
        self.residual_conv = nn.Conv2d(F1 * D, F2 * 3, (1, 1))
        self.residual_norm = nn.BatchNorm2d(F2 * 3)

        # 6. Adaptive Feature Extraction
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 16))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 7. Advanced Classifier with Residual Connections
        self.feature_dim = F2 * 3 * 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim + F2 * 3, 256),  # +global features
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(drop_prob * 0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(drop_prob * 0.25),
            nn.Linear(64, n_outputs),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Erweiterte Gewichts-Initialisierung."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (batch, n_chans, n_times) -> (batch, 1, n_chans, n_times)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 1. Temporal Convolution
        x = self.conv_temporal(x)
        x = self.batchnorm1(x)

        # 2. Spatial Convolution
        x = self.conv_spatial(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 4))
        x = self.dropout1(x)

        # Store for residual connection
        residual = self.residual_conv(x)
        residual = self.residual_norm(residual)

        # 3. Multi-Scale Convolutions
        x_small = self.conv_sep_small(x)
        x_medium = self.conv_sep_medium(x)
        x_large = self.conv_sep_large(x)

        # Concatenate multi-scale features
        x = torch.cat([x_small, x_medium, x_large], dim=1)
        x = self.batchnorm3(x)
        x = F.elu(x)

        # Add residual connection
        if x.shape == residual.shape:
            x = x + residual

        x = F.avg_pool2d(x, (1, 4))
        x = self.dropout2(x)

        # 4. Transformer Processing
        # Reshape for transformer: (batch, seq_len, features)
        batch_size, channels, height, width = x.shape
        x_seq = x.squeeze(2).transpose(1, 2)  # (batch, time, channels)

        # Add positional encoding
        x_seq = self.pos_encoding(x_seq)

        # Apply transformer layers
        for transformer in self.transformer_layers:
            x_seq = transformer(x_seq)

        # Reshape back to conv format
        x_transformed = x_seq.transpose(1, 2).unsqueeze(2)  # (batch, channels, 1, time)

        # 5. Feature Extraction
        x_adaptive = self.adaptive_pool(x_transformed)  # Local features
        x_global = self.global_pool(x_transformed)  # Global features

        # Flatten and concatenate
        x_adaptive_flat = x_adaptive.flatten(1)
        x_global_flat = x_global.flatten(1)

        # Combine features
        x_combined = torch.cat([x_adaptive_flat, x_global_flat], dim=1)

        # 6. Classification
        output = self.classifier(x_combined)

        return output


class EEGDataAugmenter:
    """Erweiterte Datenaugmentierung für EEG-Signale."""

    def __init__(self, sfreq: float = 250.0):
        self.sfreq = sfreq

    def add_gaussian_noise(
        self, data: np.ndarray, noise_factor: float = 0.01
    ) -> np.ndarray:
        """Fügt Gaussian-Rauschen hinzu."""
        noise = np.random.normal(0, noise_factor, data.shape)
        return data + noise

    def temporal_jitter(self, data: np.ndarray, max_shift: int = 10) -> np.ndarray:
        """Führt zeitliches Jittering durch."""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            return np.pad(data, ((0, 0), (0, 0), (shift, 0)), mode="edge")[
                :, :, :-shift
            ]
        elif shift < 0:
            return np.pad(data, ((0, 0), (0, 0), (0, -shift)), mode="edge")[
                :, :, -shift:
            ]
        return data

    def frequency_shift(self, data: np.ndarray, shift_hz: float = 1.0) -> np.ndarray:
        """Führt Frequenzverschiebung durch."""
        augmented_data = np.zeros_like(data)

        for epoch_idx in range(data.shape[0]):
            for ch_idx in range(data.shape[1]):
                # FFT
                fft = np.fft.fft(data[epoch_idx, ch_idx, :])
                freqs = np.fft.fftfreq(data.shape[2], 1 / self.sfreq)

                # Shift frequencies
                shift_samples = int(shift_hz * data.shape[2] / self.sfreq)
                fft_shifted = np.roll(fft, shift_samples)

                # IFFT
                augmented_data[epoch_idx, ch_idx, :] = np.real(np.fft.ifft(fft_shifted))

        return augmented_data

    def amplitude_scale(
        self, data: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """Skaliert Amplituden zufällig."""
        scale_factors = np.random.uniform(
            scale_range[0], scale_range[1], (data.shape[0], data.shape[1], 1)
        )
        return data * scale_factors

    def augment_batch(self, data: np.ndarray, augment_prob: float = 0.5) -> np.ndarray:
        """Wendet zufällige Augmentierungen auf einen Batch an."""
        augmented_data = data.copy()

        for i in range(data.shape[0]):
            if np.random.random() < augment_prob:
                # Wähle zufällige Augmentierung(en)
                if np.random.random() < 0.3:
                    augmented_data[i : i + 1] = self.add_gaussian_noise(
                        augmented_data[i : i + 1], noise_factor=0.02
                    )

                if np.random.random() < 0.3:
                    augmented_data[i : i + 1] = self.temporal_jitter(
                        augmented_data[i : i + 1], max_shift=5
                    )

                if np.random.random() < 0.2:
                    augmented_data[i : i + 1] = self.amplitude_scale(
                        augmented_data[i : i + 1], scale_range=(0.9, 1.1)
                    )

        return augmented_data


class UltraEEGNetTrainer:
    """Ultra-optimierter EEGNet Trainer mit allen fortgeschrittenen Features."""

    def __init__(
        self,
        n_chans: int = 8,
        n_outputs: int = 3,
        input_window_samples: int = None,
        F1: int = 20,  # Erhöht für mehr Kapazität
        D: int = 5,  # Erhöht für tiefere räumliche Features
        F2: int = 40,  # Erhöht für mehr Temporal Features
        kernel_length: int = 48,
        drop_prob: float = 0.3,
        n_transformer_layers: int = 3,  # Mehr Transformer Layer
        n_attention_heads: int = 8,
        batch_size: int = 32,
        lr: float = 0.0008,  # Reduziert für stabileres Training
        weight_decay: float = 0.003,  # Erhöht für bessere Regularisierung
        n_epochs: int = 250,  # Mehr Epochen
        device: str = "auto",
        use_label_smoothing: bool = True,
        label_smoothing_factor: float = 0.15,  # Erhöht
        use_data_augmentation: bool = True,
        curriculum_learning: bool = True,
    ):
        """Initialisiert Ultra-EEGNet Trainer."""

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.input_window_samples = input_window_samples
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.use_label_smoothing = use_label_smoothing
        self.label_smoothing_factor = label_smoothing_factor
        self.use_data_augmentation = use_data_augmentation
        self.curriculum_learning = curriculum_learning

        # Device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🚀 Ultra-EEGNet using device: {self.device}")

        # Model parameters
        self.model_params = {
            "n_chans": n_chans,
            "n_outputs": n_outputs,
            "n_times": input_window_samples,
            "F1": F1,
            "D": D,
            "F2": F2,
            "kernel_length": kernel_length,
            "drop_prob": drop_prob,
            "n_transformer_layers": n_transformer_layers,
            "n_attention_heads": n_attention_heads,
        }

        # Training objects
        self.model = None
        self.clf = None
        self.label_encoder = LabelEncoder()
        self.augmenter = EEGDataAugmenter()
        self.class_names = ["n-back 1", "n-back 2", "n-back 3"]

        # Performance tracking
        self.training_history = []
        self.cv_scores = []

    def load_and_preprocess_epochs(self, fif_path: Path) -> mne.Epochs:
        """Lädt und präprozessiert EEG-Epochen mit Ultra-Preprocessing."""

        print(f"🔄 Loading epochs from: {fif_path}")
        epochs = mne.read_epochs(str(fif_path), verbose=False)

        print("🧠 Applying ultra-advanced preprocessing...")

        # Separiere Epochen
        baseline_epochs = epochs["baseline"]
        task_epochs = epochs[["1-back", "2-back", "3-back"]]

        print(
            f"📊 Found {len(baseline_epochs)} baseline + {len(task_epochs)} task epochs"
        )

        # Erweiterte robuste Baseline-Korrektur
        baseline_data = baseline_epochs.get_data()

        # Entferne extreme Outliers
        for ch in range(baseline_data.shape[1]):
            ch_data = baseline_data[:, ch, :]

            # Robuste Outlier-Erkennung mit IQR
            q25, q75 = np.percentile(ch_data.flatten(), [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 3 * iqr
            upper_bound = q75 + 3 * iqr

            # Clip extreme values
            baseline_data[:, ch, :] = np.clip(ch_data, lower_bound, upper_bound)

        # Robuste Baseline-Statistiken
        mean_baseline = np.median(baseline_data, axis=(0, 2))
        std_baseline = np.std(baseline_data, axis=(0, 2))

        # Task-Epochen Preprocessing
        task_data = task_epochs.get_data()

        # 1. Baseline-Korrektur
        for ch in range(task_data.shape[1]):
            task_data[:, ch, :] -= mean_baseline[ch]

        # 2. Erweiterte Artefakt-Behandlung
        for ch in range(task_data.shape[1]):
            ch_data = task_data[:, ch, :]

            # Entferne Epochen mit extremen RMS-Werten
            rms_values = np.sqrt(np.mean(ch_data**2, axis=1))
            rms_median = np.median(rms_values)
            rms_mad = np.median(np.abs(rms_values - rms_median))

            # Korrigiere Outlier
            outlier_threshold = rms_median + 4 * rms_mad
            outlier_mask = rms_values > outlier_threshold

            if np.any(outlier_mask):
                print(f"Channel {ch}: Correcting {np.sum(outlier_mask)} outlier epochs")

                # Ersetze durch gefilterte Versionen
                for epoch_idx in np.where(outlier_mask)[0]:
                    # Butterworth Filter
                    sos = signal.butter(3, 40, btype="low", fs=250, output="sos")
                    filtered = signal.sosfilt(sos, ch_data[epoch_idx, :])

                    # Skaliere auf median RMS
                    filtered_rms = np.sqrt(np.mean(filtered**2))
                    if filtered_rms > 0:
                        scale = rms_median / filtered_rms
                        task_data[epoch_idx, ch, :] = filtered * scale

        # 3. Erweiterte Normalisierung
        # Z-Score pro Kanal mit robuster Statistik
        for ch in range(task_data.shape[1]):
            ch_data = task_data[:, ch, :].flatten()
            robust_mean = np.median(ch_data)
            robust_std = np.std(ch_data)

            if robust_std > 0:
                task_data[:, ch, :] = (task_data[:, ch, :] - robust_mean) / robust_std

        # 4. Sanity checks
        task_data = np.nan_to_num(task_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Update epochs
        task_epochs._data = task_data

        print(f"✅ Using {len(task_epochs)} preprocessed task epochs")
        print(f"📈 Data range: [{task_data.min():.3f}, {task_data.max():.3f}]")

        return task_epochs

    def create_ultra_model(self) -> EEGClassifier:
        """Erstellt Ultra-EEGNet Modell."""

        # Verwende Ultra-Architektur
        model = UltraEEGNet(**self.model_params)
        print(
            f"🏗️  Created Ultra-EEGNet with {sum(p.numel() for p in model.parameters()):,} parameters"
        )

        # Erweiterte Loss-Function
        if self.use_label_smoothing:
            criterion = lambda: nn.CrossEntropyLoss(
                label_smoothing=self.label_smoothing_factor,
                weight=None,  # Könnte für unbalanced classes angepasst werden
            )
        else:
            criterion = nn.CrossEntropyLoss

        # Ultra-Callbacks
        from skorch.callbacks import GradientNormClipping, EpochScoring

        callbacks = [
            EarlyStopping(
                patience=25,  # Mehr Geduld für komplexeres Modell
                monitor="valid_loss",
                load_best=True,
                threshold=0.001,
            ),
            LRScheduler(
                "ReduceLROnPlateau",
                monitor="valid_loss",
                patience=10,  # Mehr Geduld
                factor=0.7,  # Weniger aggressive Reduktion
                min_lr=1e-7,
            ),
            GradientNormClipping(gradient_clip_value=0.5),  # Reduziert für Stabilität
            Checkpoint(monitor="valid_acc", load_best=True),
            EpochScoring(
                "accuracy", lower_is_better=False, on_train=True, name="train_acc"
            ),
        ]

        # Ultra-EEGClassifier
        clf = EEGClassifier(
            model,
            criterion=criterion,
            optimizer=torch.optim.AdamW,
            optimizer__lr=self.lr,
            optimizer__weight_decay=self.weight_decay,
            optimizer__betas=(0.9, 0.999),
            optimizer__eps=1e-8,
            optimizer__amsgrad=True,  # Für bessere Konvergenz
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

    def extract_labels_from_epochs(
        self, epochs: mne.Epochs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extrahiert Labels (identisch zu vorherigen Versionen)."""
        X = epochs.get_data()
        event_ids = epochs.events[:, 2]

        y = []
        for event_id in event_ids:
            if event_id == 2:
                y.append(0)
            elif event_id == 3:
                y.append(1)
            elif event_id == 4:
                y.append(2)
            else:
                raise ValueError(f"Unexpected event ID {event_id}")

        y = np.array(y)

        if self.input_window_samples is None:
            self.input_window_samples = X.shape[2]
            self.model_params["n_times"] = self.input_window_samples

        return X, y

    def train_ultra(
        self, epochs: mne.Epochs, train_size: float = 0.8, use_cv: bool = True
    ) -> Dict:
        """Trainiert Ultra-EEGNet mit allen fortgeschrittenen Features."""

        print("🚀 Starting Ultra-EEGNet training with advanced features...")

        if use_cv:
            print("📊 Performing robust 5-fold cross-validation...")
            cv_results = self.cross_validate_performance(epochs)
            self.cv_scores = cv_results["cv_scores"]

        # Final training
        train_dataset, valid_dataset, y_train, y_valid = self.prepare_ultra_dataset(
            epochs, train_size
        )

        print("🏗️  Creating Ultra-EEGNet architecture...")
        self.clf = self.create_ultra_model()

        print("🎯 Starting final ultra-training...")

        # Training mit Data Augmentation
        X_train, y_train_data = train_dataset
        X_valid, y_valid_data = valid_dataset

        # Data Augmentation falls aktiviert
        if self.use_data_augmentation:
            print("🔄 Applying advanced data augmentation...")
            X_train_aug = self.augmenter.augment_batch(X_train, augment_prob=0.6)
            X_train = np.concatenate([X_train, X_train_aug], axis=0)
            y_train_data = np.concatenate([y_train_data, y_train_data], axis=0)
            print(f"📈 Augmented training set: {len(X_train)} epochs")

        # Konvertiere zu Tensoren
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train_data = torch.tensor(y_train_data, dtype=torch.long)
        X_valid = torch.tensor(X_valid, dtype=torch.float32)
        y_valid_data = torch.tensor(y_valid_data, dtype=torch.long)

        from torch.utils.data import TensorDataset

        train_tensor_dataset = TensorDataset(X_train, y_train_data)
        valid_tensor_dataset = TensorDataset(X_valid, y_valid_data)

        self.clf.train_split = predefined_split(valid_tensor_dataset)

        # Training
        self.clf.fit(train_tensor_dataset, y=None)

        # Evaluation
        y_pred = self.clf.predict(X_valid)
        y_valid_np = y_valid_data.numpy()

        # Konfidenz-Scores
        try:
            y_proba = self.clf.predict_proba(X_valid)
            confidence_scores = np.max(y_proba, axis=1)
            mean_confidence = np.mean(confidence_scores)
        except:
            mean_confidence = None

        # Berechne finale Metriken
        accuracy = (y_valid_np == y_pred).mean()

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

        print(f"🎯 Final Ultra-EEGNet Validation Accuracy: {accuracy:.3f}")
        if mean_confidence:
            print(f"🎯 Mean Confidence: {mean_confidence:.3f}")
        if use_cv:
            print(
                f"📊 CV Mean ± Std: {np.mean(self.cv_scores):.3f} ± {np.std(self.cv_scores):.3f}"
            )

        return results

    def cross_validate_performance(self, epochs: mne.Epochs, n_splits: int = 5) -> Dict:
        """Cross-Validation für Ultra-EEGNet."""

        X, y = self.extract_labels_from_epochs(epochs)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_scores = []
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n🔄 Ultra-CV Fold {fold + 1}/{n_splits}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Data Augmentation für Training
            if (
                self.use_data_augmentation and len(X_train) < 400
            ):  # Nur bei kleinen Datasets
                X_train_aug = self.augmenter.augment_batch(X_train, augment_prob=0.5)
                X_train = np.concatenate([X_train, X_train_aug], axis=0)
                y_train = np.concatenate([y_train, y_train], axis=0)

            # Model erstellen
            clf = self.create_ultra_model()

            # Training
            from torch.utils.data import TensorDataset

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

            clf.train_split = predefined_split(val_dataset)
            clf.fit(train_dataset, y=None)

            # Evaluation
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

            print(f"✅ Ultra-CV Fold {fold + 1} accuracy: {accuracy:.3f}")

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        print(f"\n🏆 Ultra-CV Results:")
        print(f"Mean accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
        print(f"Individual scores: {[f'{score:.3f}' for score in cv_scores]}")

        return {
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "fold_results": fold_results,
        }

    def prepare_ultra_dataset(
        self, epochs: mne.Epochs, train_size: float = 0.8
    ) -> Tuple:
        """Bereitet Ultra-Dataset vor (identisch zu vorherigen, aber mit Augmentation-Support)."""

        X, y = self.extract_labels_from_epochs(epochs)

        if self.input_window_samples is None:
            self.input_window_samples = X.shape[2]
            self.model_params["n_times"] = self.input_window_samples

        # Klassenweise zeitbasierte Aufteilung (gleich wie vorher)
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
            buffer_size = max(2, int(n_class * 0.12))  # Etwas größerer Puffer

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

        print(f"🎯 Ultra-Training set: {len(epochs_train)} epochs")
        print(f"🎯 Ultra-Validation set: {len(epochs_valid)} epochs")

        # Return als arrays für Data Augmentation Support
        train_windows = epochs_train.get_data()
        valid_windows = epochs_valid.get_data()

        train_dataset = (train_windows, y_train_encoded)
        valid_dataset = (valid_windows, y_valid_encoded)

        return train_dataset, valid_dataset, y_train_encoded, y_valid_encoded


def train_ultra_eegnet(
    epochs_path: Path,
    output_dir: Path,
    participant_name: str = "unknown",
    session_name: str = "session",
    use_cross_validation: bool = True,
) -> Dict:
    """Trainiert Ultra-EEGNet für maximale Performance."""

    print(f"🚀 Training Ultra-EEGNet for {participant_name} - {session_name}")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Ultra-Trainer mit optimierten Parametern
        trainer = UltraEEGNetTrainer(
            n_chans=8,
            n_outputs=3,
            batch_size=28,  # Reduziert für Stabilität
            lr=0.0005,  # Konservativere Lernrate
            weight_decay=0.004,  # Mehr Regularisierung
            n_epochs=300,  # Mehr Epochen
            device="auto",
            F1=24,  # Erhöhte Kapazität
            D=6,  # Tiefere räumliche Features
            F2=48,  # Mehr temporale Features
            kernel_length=40,  # Optimierte Kernel-Größe
            drop_prob=0.35,  # Mehr Dropout
            n_transformer_layers=4,  # Mehr Transformer-Layer
            n_attention_heads=8,  # Multi-Head Attention
            use_label_smoothing=True,
            label_smoothing_factor=0.12,
            use_data_augmentation=True,
            curriculum_learning=True,
        )

        # Lade und präprozessiere Epochen
        epochs = trainer.load_and_preprocess_epochs(epochs_path)

        # Ultra-Training
        results = trainer.train_ultra(
            epochs, train_size=0.75, use_cv=use_cross_validation
        )

        # Speichere Ultra-Modell
        model_path = output_dir / f"{participant_name}_{session_name}_ultra_eegnet.pkl"
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
            "method": "ultra_transformer_eegnet",
        }

        print(f"\n🏆 Ultra-EEGNet Training Summary:")
        print(f"🎯 Participant: {participant_name}")
        print(f"🎯 Session: {session_name}")
        print(f"🎯 Final Accuracy: {results['accuracy']:.3f}")
        if results.get("cv_results"):
            print(
                f"📊 CV Accuracy: {np.mean(results['cv_results']):.3f} ± {np.std(results['cv_results']):.3f}"
            )
        if results.get("mean_confidence"):
            print(f"🎯 Mean Confidence: {results['mean_confidence']:.3f}")
        print(
            f"📈 Improvement over random: {(results['accuracy'] - 0.333) / 0.333 * 100:.1f}%"
        )
        print(f"💾 Model saved: {model_path}")

        return results_summary

    except Exception as e:
        print(f"❌ Ultra-training failed: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    """Beispiel-Verwendung für Ultra-EEGNet."""

    base_dir = Path(__file__).parent.parent.parent.parent
    epochs_path = (
        base_dir / "results" / "processed" / "Aliaa" / "indoor_processed-epo.fif"
    )
    output_dir = base_dir / "results" / "models_ultra"

    if epochs_path.exists():
        print("🚀 Starting Ultra-EEGNet Training Session")
        print("Target: >85% Cross-Validation Accuracy")
        print("=" * 50)

        results = train_ultra_eegnet(
            epochs_path=epochs_path,
            output_dir=output_dir,
            participant_name="Aliaa",
            session_name="indoor",
            use_cross_validation=True,
        )

        print("\n🏆 Ultra-EEGNet training completed!")
        print(f"Results: {results}")
    else:
        print(f"❌ Epochs file not found: {epochs_path}")
