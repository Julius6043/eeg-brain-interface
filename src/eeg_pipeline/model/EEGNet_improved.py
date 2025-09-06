"""EEGNet Model Training mit Braindecode - Verbessert für 3-Klassen n-back Klassifikation.

Dieses Modul implementiert ein EEGNet für die Klassifikation von n-back Schwierigkeitsgraden
basierend auf epochierten EEG-Daten aus der Pipeline.

Features:
    * Lädt epochierte EEG-Daten (FIF-Format)
    * Nutzt Baseline-Epochen für intelligente Baseline-Korrektur
    * Trainiert nur auf n-back 1, 2, 3 (ohne 0-back und baseline)
    * Vermeidet Data Leakage durch zeitbasierte Train/Validation-Aufteilung
    * Evaluiert Performance für 3-Klassen-Problem
    * Speichert trainiertes Modell

Dependencies:
    * braindecode
    * mne
    * sklearn
    * torch
    * numpy
"""

import mne
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Braindecode imports
from braindecode import EEGClassifier
from braindecode.models import EEGNet
from braindecode.datasets import create_from_mne_epochs
from braindecode.preprocessing import exponential_moving_standardize, preprocess

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Skorch imports
from skorch.callbacks import LRScheduler, EarlyStopping
from skorch.helper import predefined_split

import matplotlib.pyplot as plt
import seaborn as sns


class EEGNetTrainer:
    """EEGNet Trainer für n-back Klassifikation (1-back, 2-back, 3-back)."""

    def __init__(
        self,
        n_chans: int = 8,
        n_outputs: int = 3,  # 3 n-back levels: 1, 2, 3 (ohne 0-back und baseline)
        input_window_samples: int = None,
        final_conv_length: str = "auto",
        pool_mode: str = "mean",
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        drop_prob: float = 0.25,
        batch_size: int = 32,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        n_epochs: int = 100,
        device: str = "auto",
    ):
        """
        Parameter
        ---------
        n_chans : int
            Anzahl EEG-Kanäle
        n_outputs : int
            Anzahl Ausgabeklassen (nur n-back 1, 2, 3)
        input_window_samples : int
            Länge der Eingabe-Epochen in Samples
        batch_size : int
            Batch-Größe für Training
        lr : float
            Lernrate
        n_epochs : int
            Anzahl Trainings-Epochen
        device : str
            Device für Training ('cuda', 'cpu', oder 'auto')
        """
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.input_window_samples = input_window_samples
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs

        # Device bestimmen
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Model parameter (ohne deprecated third_kernel_size)
        self.model_params = {
            "n_chans": n_chans,
            "n_outputs": n_outputs,
            "n_times": input_window_samples,  # Wichtig für EEGNet
            "final_conv_length": final_conv_length,
            "pool_mode": pool_mode,
            "F1": F1,
            "D": D,
            "F2": F2,
            "kernel_length": kernel_length,
            "drop_prob": drop_prob,
        }

        # Model und Training-Objekte
        self.model = None
        self.clf = None
        self.label_encoder = LabelEncoder()
        self.class_names = ["n-back 1", "n-back 2", "n-back 3"]

    def load_epochs_from_fif(self, fif_path: Path) -> mne.Epochs:
        """Lädt epochierte EEG-Daten aus FIF-Datei und führt intelligente Baseline-Korrektur durch.

        Parameter
        ---------
        fif_path : Path
            Pfad zur FIF-Datei mit Epochen

        Returns
        -------
        mne.Epochs
            Geladene Epochen (nur n-back 1, 2, 3)
        """
        print(f"Loading epochs from: {fif_path}")
        epochs = mne.read_epochs(str(fif_path), verbose=False)

        print("Applying intelligent baseline correction using baseline epochs...")

        # Separiere Baseline- und Task-Epochen
        baseline_epochs = epochs["baseline"]  # Event-ID 0
        task_epochs = epochs[["1-back", "2-back", "3-back"]]  # Event-IDs 2, 3, 4

        print(
            f"Found {len(baseline_epochs)} baseline epochs and {len(task_epochs)} task epochs"
        )

        # Berechne mittlere Baseline pro Kanal aus Baseline-Epochen
        baseline_data = (
            baseline_epochs.get_data()
        )  # Shape: (n_baseline_epochs, n_channels, n_times)
        mean_baseline = np.mean(
            baseline_data, axis=(0, 2)
        )  # Mittelwert über Epochen und Zeit

        print(f"Baseline values per channel: {mean_baseline}")

        # Wende Baseline-Korrektur auf Task-Epochen an
        task_data = task_epochs.get_data()
        for ch in range(task_data.shape[1]):
            task_data[:, ch, :] -= mean_baseline[ch]

        # Z-Score Normalisierung pro Kanal nach Baseline-Korrektur
        for ch in range(task_data.shape[1]):
            ch_data = task_data[:, ch, :]
            std_val = np.std(ch_data)
            if std_val > 0:
                task_data[:, ch, :] = ch_data / std_val

        # Erstelle neue Epochs nur mit Task-Daten
        task_epochs._data = task_data

        print(f"Using {len(task_epochs)} task epochs (1-back, 2-back, 3-back)")
        print(f"Event IDs in task epochs: {task_epochs.event_id}")
        print(f"Sampling rate: {task_epochs.info['sfreq']} Hz")
        print(f"Epoch duration: {task_epochs.times[-1] - task_epochs.times[0]:.2f}s")
        print(
            f"Data range after preprocessing: [{task_data.min():.3f}, {task_data.max():.3f}]"
        )

        return task_epochs

    def extract_labels_from_epochs(
        self, epochs: mne.Epochs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extrahiert n-back Labels aus Epoch-Events (nur 1-back, 2-back, 3-back).

        Parameter
        ---------
        epochs : mne.Epochs
            MNE Epochs-Objekt mit Event-IDs (bereits gefiltert auf Task-Epochen)

        Returns
        -------
        X : np.ndarray
            EEG-Daten (n_epochs, n_channels, n_times)
        y : np.ndarray
            n-back Labels (n_epochs,) - 0=1-back, 1=2-back, 2=3-back
        """
        # Hole EEG-Daten
        X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)

        # Extrahiere n-back Labels aus Event-IDs
        event_ids = epochs.events[:, 2]  # Event-IDs aus Events-Array

        # Mapping von Event-ID zu Klassen-Index
        # Event-IDs: 2=1-back, 3=2-back, 4=3-back
        # Klassen-Indices: 0=1-back, 1=2-back, 2=3-back
        y = []
        for event_id in event_ids:
            if event_id == 2:  # 1-back
                y.append(0)
            elif event_id == 3:  # 2-back
                y.append(1)
            elif event_id == 4:  # 3-back
                y.append(2)
            else:
                raise ValueError(
                    f"Unexpected event ID {event_id}. Expected 2, 3, or 4."
                )

        y = np.array(y)

        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Label distribution: {np.bincount(y)} (0=1-back, 1=2-back, 2=3-back)")
        print(f"Unique labels: {np.unique(y)}")

        return X, y

    def prepare_braindecode_dataset(
        self, epochs: mne.Epochs, train_size: float = 0.8
    ) -> Tuple:
        """Bereitet Braindecode-Dataset vor mit Data-Leakage-Vermeidung.

        Da Epochen 50% Overlap haben, verwenden wir eine spezielle Aufteilung:
        - Zeitbasierte Aufteilung statt zufällige Aufteilung
        - Sicherheitspuffer zwischen Training und Validation

        Parameter
        ---------
        epochs : mne.Epochs
            MNE Epochs-Objekt (nur Task-Epochen)
        train_size : float
            Anteil der Trainingsdaten

        Returns
        -------
        tuple
            (train_dataset, valid_dataset, y_train, y_valid)
        """
        # Extrahiere Daten und Labels
        X, y = self.extract_labels_from_epochs(epochs)

        # Update input_window_samples falls nicht gesetzt
        if self.input_window_samples is None:
            self.input_window_samples = X.shape[2]
            self.model_params["n_times"] = self.input_window_samples
            print(f"Set input_window_samples to: {self.input_window_samples}")

        # WICHTIG: Stratifizierte zeitbasierte Aufteilung wegen 50% Overlap
        # Berechne Zeitstempel für jede Epoche
        epoch_times = []
        for i in range(len(epochs)):
            epoch_times.append(i)

        epoch_times = np.array(epoch_times)

        # Gruppiere Epochen nach Klassen
        class_indices = {}
        for class_idx in range(3):
            class_indices[class_idx] = np.where(y == class_idx)[0]

        print(f"Class distribution before split:")
        for class_idx, indices in class_indices.items():
            print(f"  Class {class_idx} (n-back {class_idx+1}): {len(indices)} epochs")

        # Zeitbasierte Aufteilung pro Klasse
        train_indices = []
        valid_indices = []

        for class_idx, indices in class_indices.items():
            # Sortiere Indices nach Zeit für diese Klasse
            sorted_indices = indices[np.argsort(epoch_times[indices])]

            # Berechne Split-Position für diese Klasse
            n_class = len(sorted_indices)
            n_train_class = int(n_class * train_size)

            # Konservativer Puffer von 10% der Klassen-Epochen
            buffer_size = max(2, int(n_class * 0.1))

            # Training: Erste train_size% minus Puffer
            train_end = max(1, n_train_class - buffer_size // 2)
            class_train_indices = sorted_indices[:train_end]

            # Validation: Letzte Epochen mit Puffer
            valid_start = min(n_class - 1, n_train_class + buffer_size // 2)
            class_valid_indices = sorted_indices[valid_start:]

            # Stelle sicher, dass jede Klasse mindestens eine Epoche in beiden Sets hat
            if len(class_train_indices) == 0:
                class_train_indices = sorted_indices[:1]
            if len(class_valid_indices) == 0:
                class_valid_indices = sorted_indices[-1:]

            train_indices.extend(class_train_indices)
            valid_indices.extend(class_valid_indices)

            print(
                f"  Class {class_idx}: {len(class_train_indices)} train, {len(class_valid_indices)} valid"
            )

        # Konvertiere zu numpy arrays
        train_indices = np.array(train_indices)
        valid_indices = np.array(valid_indices)

        print(f"Total temporal split with class-wise buffer:")
        print(f"  Training epochs: {len(train_indices)} total")
        print(f"  Validation epochs: {len(valid_indices)} total")

        # Erstelle separate Epochs-Objekte für Train und Validation
        epochs_train = epochs[train_indices]
        epochs_valid = epochs[valid_indices]

        # Labels für Train/Valid
        y_train = y[train_indices]
        y_valid = y[valid_indices]

        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_valid_encoded = self.label_encoder.transform(y_valid)

        print(f"Training set: {len(epochs_train)} epochs")
        print(f"Validation set: {len(epochs_valid)} epochs")
        print(
            f"Training label distribution: {np.bincount(y_train_encoded)} (0=1-back, 1=2-back, 2=3-back)"
        )
        print(
            f"Validation label distribution: {np.bincount(y_valid_encoded)} (0=1-back, 1=2-back, 2=3-back)"
        )

        # Überprüfe Klassenbalance
        train_class_counts = np.bincount(y_train_encoded)
        valid_class_counts = np.bincount(y_valid_encoded)

        if np.any(train_class_counts == 0) or np.any(valid_class_counts == 0):
            print("WARNING: Some classes missing in train or validation set!")
            print("Consider adjusting buffer size or train_size parameter.")

        # Erstelle proper Braindecode Datasets
        try:
            # Use create_from_mne_epochs for proper Braindecode integration
            train_dataset = create_from_mne_epochs(
                [epochs_train],
                description=pd.Series(["train"]),
                target=pd.Series([1]),  # Dummy target, will be overridden
            )[0]

            valid_dataset = create_from_mne_epochs(
                [epochs_valid],
                description=pd.Series(["valid"]),
                target=pd.Series([1]),  # Dummy target, will be overridden
            )[0]

            # Set the actual targets
            train_dataset.target = y_train_encoded
            valid_dataset.target = y_valid_encoded

            print(f"Train dataset windows: {len(train_dataset.windows)}")
            print(f"Valid dataset windows: {len(valid_dataset.windows)}")

            return train_dataset, valid_dataset, y_train_encoded, y_valid_encoded

        except Exception as e:
            print(f"Braindecode dataset creation failed: {e}")
            print("Falling back to simple tensor approach...")

            # Fallback: Manually create windows datasets
            train_windows = (
                epochs_train.get_data()
            )  # Shape: (n_epochs, n_channels, n_times)
            valid_windows = epochs_valid.get_data()

            print(f"Train windows shape: {train_windows.shape}")
            print(f"Valid windows shape: {valid_windows.shape}")

            # Create simple dataset tuples for direct use with skorch
            train_dataset = (train_windows, y_train_encoded)
            valid_dataset = (valid_windows, y_valid_encoded)

            return train_dataset, valid_dataset, y_train_encoded, y_valid_encoded

    def create_model(self) -> EEGClassifier:
        """Erstellt EEGNet-Modell mit Braindecode.

        Returns
        -------
        EEGClassifier
            Braindecode EEGClassifier mit EEGNet
        """
        # Erstelle EEGNet-Modell (verwende neue EEGNet Klasse statt deprecated EEGNetv4)
        from braindecode.models import EEGNet

        model = EEGNet(**self.model_params)

        # Verbesserte Callbacks für Training
        callbacks = [
            EarlyStopping(patience=15, monitor="valid_loss"),  # Mehr Geduld
            LRScheduler(
                "ReduceLROnPlateau", monitor="valid_loss", patience=5, factor=0.5
            ),
        ]

        # Erstelle EEGClassifier mit optimierten Parametern
        clf = EEGClassifier(
            model,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=self.lr,
            optimizer__weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_epochs=self.n_epochs,
            device=self.device,
            callbacks=callbacks,
            train_split=None,  # Wir verwenden predefined_split
            verbose=1,
            # Optimierte Training-Parameter
            iterator_train__shuffle=True,
            iterator_valid__shuffle=False,
        )

        return clf

    def train(self, epochs: mne.Epochs, train_size: float = 0.8) -> Dict:
        """Trainiert EEGNet auf epochierten Daten.

        Parameter
        ---------
        epochs : mne.Epochs
            MNE Epochs-Objekt
        train_size : float
            Anteil der Trainingsdaten

        Returns
        -------
        Dict
            Training-Ergebnisse und Metriken
        """
        print("Preparing datasets...")
        train_dataset, valid_dataset, y_train, y_valid = (
            self.prepare_braindecode_dataset(epochs, train_size)
        )

        print("Creating model...")
        self.clf = self.create_model()

        print("Starting training...")

        # Check if we have proper Braindecode datasets or simple tuples
        if hasattr(train_dataset, "windows"):  # Proper Braindecode dataset
            print("Using proper Braindecode datasets...")
            # Set validation split manually with proper dataset
            self.clf.train_split = predefined_split(valid_dataset)
            # Fit the model with Braindecode dataset
            self.clf.fit(train_dataset, y=None)  # y is already in dataset

            # Get validation predictions
            y_pred = self.clf.predict(valid_dataset)
            y_valid_np = valid_dataset.target

        else:  # Simple tuple approach
            print("Using simple tensor approach...")
            # Extract data and labels from dataset tuples
            X_train, y_train_data = train_dataset
            X_valid, y_valid_data = valid_dataset

            # Convert to torch tensors for compatibility
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train_data = torch.tensor(y_train_data, dtype=torch.long)
            X_valid = torch.tensor(X_valid, dtype=torch.float32)
            y_valid_data = torch.tensor(y_valid_data, dtype=torch.long)

            # For simple approach, we need to create a custom dataset
            from torch.utils.data import TensorDataset

            # Create TensorDatasets
            train_tensor_dataset = TensorDataset(X_train, y_train_data)
            valid_tensor_dataset = TensorDataset(X_valid, y_valid_data)

            # Set validation split manually
            self.clf.train_split = predefined_split(valid_tensor_dataset)

            # Fit the model
            self.clf.fit(train_tensor_dataset, y=None)

            # Evaluiere auf Validierungsset
            print("Evaluating on validation set...")
            y_pred = self.clf.predict(X_valid)
            y_valid_np = (
                y_valid_data.numpy()
                if isinstance(y_valid_data, torch.Tensor)
                else y_valid_data
            )

        print("Training completed!")

        # Metriken berechnen - robuste Behandlung für fehlende Klassen
        unique_labels = np.unique(np.concatenate([y_valid_np, y_pred]))
        n_classes_present = len(unique_labels)

        if n_classes_present < self.n_outputs:
            print(
                f"WARNING: Only {n_classes_present} out of {self.n_outputs} classes present in validation set!"
            )
            # Erstelle angepasste Target-Namen nur für vorhandene Klassen
            target_names = [f"n-back {label+1}" for label in unique_labels]
        else:
            target_names = [f"n-back {i+1}" for i in range(self.n_outputs)]

        # Metriken berechnen
        results = {
            "y_true": y_valid_np,
            "y_pred": y_pred,
            "classification_report": classification_report(
                y_valid_np,
                y_pred,
                labels=unique_labels,
                target_names=target_names,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_valid_np, y_pred, labels=unique_labels
            ),
            "accuracy": (y_valid_np == y_pred).mean(),
            "train_size": len(y_train),
            "valid_size": len(y_valid),
            "classes_present": unique_labels.tolist(),
        }

        print(f"Validation Accuracy: {results['accuracy']:.3f}")
        print(f"Classification Report:\n{results['classification_report']}")

        return results

    def plot_confusion_matrix(self, results: Dict, save_path: Optional[Path] = None):
        """Plottet Confusion Matrix.

        Parameter
        ---------
        results : Dict
            Training-Ergebnisse mit Confusion Matrix
        save_path : Path, optional
            Pfad zum Speichern der Abbildung
        """
        cm = results["confusion_matrix"]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[f"n-back {i+1}" for i in range(self.n_outputs)],  # 1, 2, 3
            yticklabels=[f"n-back {i+1}" for i in range(self.n_outputs)],  # 1, 2, 3
        )
        plt.title("Confusion Matrix - EEGNet n-back Classification (1, 2, 3)")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to: {save_path}")

        plt.close()  # Verhindert blocking

    def save_model(self, save_path: Path):
        """Speichert trainiertes Modell.

        Parameter
        ---------
        save_path : Path
            Pfad zum Speichern des Modells
        """
        if self.clf is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Speichere das trainierte Modell
        self.clf.save_params(f_params=str(save_path))
        print(f"Model saved to: {save_path}")

    def load_model(self, model_path: Path, epochs: mne.Epochs):
        """Lädt trainiertes Modell.

        Parameter
        ---------
        model_path : Path
            Pfad zum gespeicherten Modell
        epochs : mne.Epochs
            Beispiel-Epochen für Modell-Initialisierung
        """
        # Erstelle Modell-Architektur
        self.clf = self.create_model()

        # Initialisiere mit Dummy-Daten
        dummy_dataset, _, _, _ = self.prepare_braindecode_dataset(
            epochs, train_size=0.8
        )
        self.clf.initialize()

        # Lade Parameter
        self.clf.load_params(f_params=str(model_path))
        print(f"Model loaded from: {model_path}")


def train_single_session_eegnet(
    epochs_path: Path,
    output_dir: Path,
    participant_name: str = "unknown",
    session_name: str = "session",
    use_braindecode: bool = True,
) -> Dict:
    """Trainiert EEGNet für eine einzelne Session (nur n-back 1, 2, 3).

    Parameter
    ---------
    epochs_path : Path
        Pfad zur FIF-Datei mit Epochen
    output_dir : Path
        Ausgabeordner für Modell und Plots
    participant_name : str
        Name des Teilnehmers
    session_name : str
        Name der Session
    use_braindecode : bool
        Ob Braindecode verwendet werden soll, sonst Fallback zu SimpleEEGNet

    Returns
    -------
    Dict
        Training-Ergebnisse
    """
    print(f"Training EEGNet for {participant_name} - {session_name}")
    print("=" * 60)

    # Ausgabeordner erstellen
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_braindecode:
        try:
            print(
                "Attempting optimized Braindecode EEGNet training for 3-class problem..."
            )

            # Trainer initialisieren mit optimierten Parametern für 3-Klassen-Problem
            trainer = EEGNetTrainer(
                n_chans=8,  # Anzahl Kanäle aus Pipeline
                n_outputs=3,  # n-back levels: 1, 2, 3 (ohne 0-back und baseline)
                batch_size=32,  # Größere Batch-Size für stabileres Training
                lr=0.003,  # Etwas höhere Lernrate für 3-Klassen-Problem
                weight_decay=0.001,  # Mehr Regularisierung
                n_epochs=120,  # Mehr Epochen für bessere Konvergenz bei 3 Klassen
                device="auto",
                # Optimierte EEGNet-Parameter für 3-Klassen-Problem
                F1=12,  # Angepasste Filter für 3 Klassen
                D=3,  # Reduzierte Tiefe für weniger Komplexität
                F2=24,  # Angepasste Filter in zweiter Schicht
                kernel_length=48,  # Mittelere Kernel-Größe
                drop_prob=0.4,  # Reduziertes Dropout für 3-Klassen-Problem
            )

            # Lade Epochen
            epochs = trainer.load_epochs_from_fif(epochs_path)

            # Training mit zeitbasierter Aufteilung zur Vermeidung von Data Leakage
            results = trainer.train(
                epochs, train_size=0.75
            )  # 75% Training, 25% Validation (mit Puffer)

            # Speichere Modell
            model_path = (
                output_dir / f"{participant_name}_{session_name}_eegnet_3class.pkl"
            )
            trainer.save_model(model_path)

            # Speichere Confusion Matrix
            plot_path = (
                output_dir
                / f"{participant_name}_{session_name}_confusion_matrix_3class.png"
            )
            trainer.plot_confusion_matrix(results, plot_path)

            # Speichere Ergebnisse
            results_summary = {
                "participant": participant_name,
                "session": session_name,
                "accuracy": results["accuracy"],
                "train_size": results["train_size"],
                "valid_size": results["valid_size"],
                "model_path": str(model_path),
                "plot_path": str(plot_path),
                "method": "braindecode_3class",
            }

            print(f"\nOptimized 3-Class Braindecode Training Summary:")
            print(f"Participant: {participant_name}")
            print(f"Session: {session_name}")
            print(f"Accuracy: {results['accuracy']:.3f}")
            print(f"Classes: n-back 1, 2, 3")
            print(f"Random chance: 33.3%")
            print(f"Improvement: {(results['accuracy'] - 0.333) / 0.333 * 100:.1f}%")
            print(f"Model saved: {model_path}")

            return results_summary

        except Exception as e:
            print(f"Braindecode training failed: {e}")
            print("Falling back to SimpleEEGNet...")
            use_braindecode = False

    if not use_braindecode:
        # Fallback zu SimpleEEGNet
        try:
            from .SimpleEEGNet import train_simple_eegnet
        except ImportError:
            # Für direkten Import wenn nicht als Modul ausgeführt
            import sys
            from pathlib import Path

            sys.path.append(str(Path(__file__).parent))
            from SimpleEEGNet import train_simple_eegnet

        print("Using SimpleEEGNet fallback...")
        results_summary = train_simple_eegnet(
            epochs_path=epochs_path,
            output_dir=output_dir,
            participant_name=participant_name,
            session_name=session_name,
        )
        results_summary["method"] = "simple_eegnet_fallback"

        return results_summary


if __name__ == "__main__":
    """Beispiel-Verwendung für 3-Klassen n-back Klassifikation."""
    # Pfade definieren
    base_dir = Path(__file__).parent.parent.parent.parent
    epochs_path = (
        base_dir / "results" / "processed" / "Aliaa" / "indoor_processed-epo.fif"
    )
    output_dir = base_dir / "results" / "models_3class"

    if epochs_path.exists():
        # Trainiere EEGNet für Single Session (3-Klassen-Problem)
        results = train_single_session_eegnet(
            epochs_path=epochs_path,
            output_dir=output_dir,
            participant_name="Aliaa",
            session_name="indoor",
        )

        print("\n3-Class EEGNet training completed successfully!")
        print(f"Results: {results}")
    else:
        print(f"Epochs file not found: {epochs_path}")
        print("Please run the pipeline first to generate epoched data.")
