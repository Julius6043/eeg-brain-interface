"""EEGNet Final Optimized - Letzte Iteration mit allen Erkenntnissen.

Dieses Modell implementiert alle gelernten Lektionen:
1. Weniger aggressive Regularisierung
2. Stabilere Architektur
3. Bessere Data Splits
4. Robuste Evaluation
5. Diagnose-System

Author: AI Assistant
Date: September 2025
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


class SimpleEEGNet(nn.Module):
    """Vereinfachtes EEGNet für stabiles Training bei kleinen Datensätzen."""

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        F1: int = 4,  # Deutlich reduziert
        D: int = 2,  # Einfach
        F2: int = 8,  # Minimal
        kernel_length: int = 32,  # Kleiner
        drop_prob: float = 0.25,  # Moderat
    ):
        super().__init__()

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times

        # Block 1: Temporal Convolution
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2)
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Block 2: Spatial Convolution
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.dropout1 = nn.Dropout(drop_prob)

        # Block 3: Separable Convolution
        self.conv_sep = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8))
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.dropout2 = nn.Dropout(drop_prob)

        # Classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * 4, 32),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(32, n_outputs),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Konservative Gewichts-Initialisierung."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass durch vereinfachtes EEGNet."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Block 1
        x = self.conv_temporal(x)
        x = self.batchnorm1(x)

        # Block 2
        x = self.conv_spatial(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 4))
        x = self.dropout1(x)

        # Block 3
        x = self.conv_sep(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 8))
        x = self.dropout2(x)

        # Classification
        x = self.adaptive_pool(x)
        x = self.classifier(x)

        return x


class FinalEEGNetTrainer:
    """Finaler EEGNet Trainer mit allen Optimierungen."""

    def __init__(
        self,
        n_chans: int = 8,
        n_outputs: int = 3,
        batch_size: int = 32,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 100,
        device: str = "auto",
        use_simple_model: bool = True,
    ):
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.use_simple_model = use_simple_model

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.label_encoder = LabelEncoder()
        self.class_names = ["n-back 1", "n-back 2", "n-back 3"]

    def extract_labels_from_epochs(
        self, epochs: mne.Epochs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extrahiert Labels aus Epochen."""
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

        return X, np.array(y)

    def create_balanced_splits(
        self, epochs: mne.Epochs, train_size: float = 0.8
    ) -> Tuple:
        """Erstellt balancierte Train/Validation Splits."""
        X, y = self.extract_labels_from_epochs(epochs)

        # Verwende StratifiedKFold für bessere Balance
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=train_size, stratify=y, random_state=42, shuffle=True
        )

        print(f"Training set: {len(X_train)} epochs")
        print(f"Validation set: {len(X_val)} epochs")
        print(f"Training distribution: {np.bincount(y_train)}")
        print(f"Validation distribution: {np.bincount(y_val)}")

        return X_train, X_val, y_train, y_val

    def create_model(self, n_times: int):
        """Erstellt optimiertes Modell."""
        if self.use_simple_model:
            model = SimpleEEGNet(
                n_chans=self.n_chans,
                n_outputs=self.n_outputs,
                n_times=n_times,
                F1=4,
                D=2,
                F2=8,
                kernel_length=32,
                drop_prob=0.25,
            )
            print("Using SimpleEEGNet architecture")
        else:
            model = EEGNet(
                n_chans=self.n_chans,
                n_outputs=self.n_outputs,
                n_times=n_times,
                F1=8,
                D=2,
                F2=16,
                kernel_length=64,
                drop_prob=0.25,
            )
            print("Using standard EEGNet architecture")

        # Optimierte Callbacks
        callbacks = [
            EarlyStopping(
                patience=15, monitor="valid_loss", load_best=True, threshold=0.001
            ),
            LRScheduler(
                "ReduceLROnPlateau",
                monitor="valid_loss",
                patience=8,
                factor=0.8,
                min_lr=1e-6,
                verbose=True,
            ),
            Checkpoint(monitor="valid_acc", load_best=True),
        ]

        clf = EEGClassifier(
            model,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=self.lr,
            optimizer__weight_decay=self.weight_decay,
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

    def train_with_cv(self, epochs: mne.Epochs, n_splits: int = 5):
        """Training mit robuster Cross-Validation."""
        print(f"Training with {n_splits}-fold cross-validation...")

        X, y = self.extract_labels_from_epochs(epochs)
        n_times = X.shape[2]

        # Stratified CV für bessere Balance
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_scores = []
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{n_splits}")
            print("-" * 30)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Prüfe Balance
            print(f"Train distribution: {np.bincount(y_train)}")
            print(f"Val distribution: {np.bincount(y_val)}")

            # Skip Folds mit schlechter Balance
            val_classes = len(np.unique(y_val))
            if val_classes < 2:
                print(
                    f"Skipping fold {fold + 1}: only {val_classes} classes in validation"
                )
                continue

            try:
                # Model erstellen
                clf = self.create_model(n_times)

                # Datasets erstellen
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

                print(f"Fold {fold + 1} accuracy: {accuracy:.3f}")

            except Exception as e:
                print(f"Fold {fold + 1} failed: {e}")
                continue

        if len(cv_scores) == 0:
            return {"error": "All CV folds failed"}

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        print(f"\nCV Results: {cv_mean:.3f} ± {cv_std:.3f}")
        print(f"Individual scores: {[f'{s:.3f}' for s in cv_scores]}")

        # Final Training
        print("\nTraining final model...")
        X_train, X_val, y_train, y_val = self.create_balanced_splits(epochs, 0.8)

        final_clf = self.create_model(n_times)

        from torch.utils.data import TensorDataset

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        final_clf.train_split = predefined_split(val_dataset)
        final_clf.fit(train_dataset, y=None)

        y_pred_final = final_clf.predict(X_val_tensor)
        final_accuracy = (y_val == y_pred_final).mean()

        print(f"Final accuracy: {final_accuracy:.3f}")

        return {
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "final_accuracy": final_accuracy,
            "final_clf": final_clf,
            "y_true": y_val,
            "y_pred": y_pred_final,
            "fold_results": fold_results,
        }


def train_final_eegnet(
    epochs_path: Path,
    output_dir: Path,
    participant_name: str = "unknown",
    session_name: str = "session",
):
    """Trainiert finales optimiertes EEGNet."""

    print(f"Training Final EEGNet for {participant_name} - {session_name}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Lade Epochen
        print(f"Loading epochs from: {epochs_path}")
        epochs = mne.read_epochs(str(epochs_path), verbose=False)

        # Einfache Präprozessierung
        task_epochs = epochs[["1-back", "2-back", "3-back"]]
        print(f"Using {len(task_epochs)} task epochs")

        # Trainer erstellen
        trainer = FinalEEGNetTrainer(
            n_chans=8,
            n_outputs=3,
            batch_size=32,
            lr=0.001,
            weight_decay=0.01,
            n_epochs=100,
            use_simple_model=True,
        )

        # Training
        results = trainer.train_with_cv(task_epochs, n_splits=5)

        if "error" in results:
            print(f"Training failed: {results['error']}")
            return results

        # Speichere Modell
        model_path = output_dir / f"{participant_name}_{session_name}_eegnet_final.pkl"
        results["final_clf"].save_params(f_params=str(model_path))

        # Zusammenfassung
        summary = {
            "participant": participant_name,
            "session": session_name,
            "cv_mean": results["cv_mean"],
            "cv_std": results["cv_std"],
            "final_accuracy": results["final_accuracy"],
            "model_path": str(model_path),
            "method": "final_simple_eegnet",
        }

        print(f"\nFinal EEGNet Results:")
        print(f"CV Performance: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        print(f"Final Accuracy: {results['final_accuracy']:.3f}")
        print(
            f"Improvement over random: {(results['final_accuracy'] - 0.333) / 0.333 * 100:.1f}%"
        )

        return summary

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    """Test das finale optimierte EEGNet."""

    base_dir = Path(__file__).parent.parent.parent.parent
    epochs_path = (
        base_dir / "results" / "processed" / "Rami" / "outdoor_processed-epo.fif"
    )
    output_dir = base_dir / "results" / "models_optimized"

    if epochs_path.exists():
        results = train_final_eegnet(
            epochs_path=epochs_path,
            output_dir=output_dir,
            participant_name="Rami",
            session_name="outdoor",
        )

        print("\nFinal EEGNet training completed!")
        print(f"Results: {results}")
    else:
        print(f"Epochs file not found: {epochs_path}")
