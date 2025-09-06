"""Vereinfachtes EEGNet Training mit direktem PyTorch.

Da Braindecode Probleme mit dem Dataset-Format hat, implementieren wir
ein direktes PyTorch-Training für EEGNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mne
from pathlib import Path
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


class EEGNet(nn.Module):
    """EEGNet Implementation in PyTorch.

    Based on the original EEGNet paper:
    Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018).
    EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.
    Journal of Neural Engineering, 15(5), 056013.
    """

    def __init__(
        self,
        n_chans=8,
        n_outputs=4,
        n_times=1001,
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        dropout=0.25,
    ):
        super(EEGNet, self).__init__()

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Block 2: Depthwise Convolution
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # Block 3: Separable Convolution
        self.conv3 = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # Calculate size for classifier
        self._calculate_classifier_input_size()

        # Classifier
        self.classifier = nn.Linear(self.classifier_input_size, n_outputs)

    def _calculate_classifier_input_size(self):
        """Calculate the input size for the classifier layer."""
        # Simulate forward pass to get size
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_chans, self.n_times)
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = self.elu1(x)
            x = self.avgpool1(x)
            x = self.conv3(x)
            x = self.batchnorm3(x)
            x = self.elu2(x)
            x = self.avgpool2(x)
            self.classifier_input_size = x.view(x.size(0), -1).size(1)

    def forward(self, x):
        # Input shape: (batch_size, n_chans, n_times)
        # Reshape to (batch_size, 1, n_chans, n_times) for Conv2d
        x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)

        # Block 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        # Block 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        # Classifier
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class SimpleEEGNetTrainer:
    """Vereinfachter EEGNet Trainer."""

    def __init__(self, device="auto", lr=0.001, batch_size=32, n_epochs=50):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model = None

        print(f"Using device: {self.device}")

    def load_epochs_from_fif(self, fif_path: Path) -> mne.Epochs:
        """Lädt epochierte EEG-Daten aus FIF-Datei."""
        print(f"Loading epochs from: {fif_path}")
        epochs = mne.read_epochs(str(fif_path), verbose=False)
        print(f"Loaded {len(epochs)} epochs with {len(epochs.ch_names)} channels")
        return epochs

    def prepare_data(self, epochs: mne.Epochs, train_size=0.7) -> Tuple:
        """Bereitet Daten für Training vor."""
        # Hole EEG-Daten
        X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)

        # Extrahiere Labels aus Events
        event_ids = epochs.events[:, 2]
        y = []
        for event_id in event_ids:
            if event_id == 0:  # Baseline
                y.append(0)
            else:
                n_back_level = event_id - 1
                y.append(n_back_level)

        y = np.array(y)

        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Label distribution: {np.bincount(y)}")

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42, stratify=y
        )

        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        return X_train, X_test, y_train, y_test

    def create_data_loader(self, X, y, shuffle=True):
        """Erstellt PyTorch DataLoader."""
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
        )

    def train(self, epochs: mne.Epochs, train_size=0.7) -> Dict:
        """Trainiert EEGNet."""
        print("Preparing data...")
        X_train, X_test, y_train, y_test = self.prepare_data(epochs, train_size)

        # Erstelle Model
        n_chans = X_train.shape[1]
        n_times = X_train.shape[2]
        n_outputs = len(np.unique(y_train.numpy()))

        self.model = EEGNet(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs).to(
            self.device
        )

        print(
            f"Model created: {n_chans} channels, {n_times} time points, {n_outputs} classes"
        )

        # DataLoaders
        train_loader = self.create_data_loader(X_train, y_train, shuffle=True)
        test_loader = self.create_data_loader(X_test, y_test, shuffle=False)

        # Optimizer und Loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        # Training Loop
        print("Starting training...")
        train_losses = []

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

        # Evaluation
        print("Evaluating...")
        self.model.eval()
        y_pred = []
        y_true = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)

                y_pred.extend(pred.cpu().numpy())
                y_true.extend(target.cpu().numpy())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        # Metriken
        accuracy = accuracy_score(y_true, y_pred)
        class_report = classification_report(
            y_true, y_pred, target_names=[f"n-back {i}" for i in range(n_outputs)]
        )
        conf_matrix = confusion_matrix(y_true, y_pred)

        results = {
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix,
            "y_true": y_true,
            "y_pred": y_pred,
            "train_losses": train_losses,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        print(f"Test Accuracy: {accuracy:.3f}")
        print(f"Classification Report:\\n{class_report}")

        return results

    def plot_confusion_matrix(self, results: Dict, save_path: Path = None):
        """Plottet Confusion Matrix."""
        cm = results["confusion_matrix"]
        n_classes = cm.shape[0]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[f"n-back {i}" for i in range(n_classes)],
            yticklabels=[f"n-back {i}" for i in range(n_classes)],
        )
        plt.title("EEGNet - n-back Classification Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to: {save_path}")

        plt.close()  # Close instead of show to avoid blocking

    def save_model(self, save_path: Path):
        """Speichert trainiertes Modell."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_params": {
                    "n_chans": self.model.n_chans,
                    "n_outputs": self.model.n_outputs,
                    "n_times": self.model.n_times,
                    "F1": self.model.F1,
                    "D": self.model.D,
                    "F2": self.model.F2,
                },
            },
            save_path,
        )
        print(f"Model saved to: {save_path}")


def train_simple_eegnet(
    epochs_path: Path,
    output_dir: Path,
    participant_name: str = "unknown",
    session_name: str = "session",
) -> Dict:
    """Trainiert vereinfachtes EEGNet für eine Session."""
    print(f"Training Simple EEGNet for {participant_name} - {session_name}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Trainer
    trainer = SimpleEEGNetTrainer(
        device="cpu", lr=0.001, batch_size=16, n_epochs=30  # CPU für Stabilität
    )

    # Lade Daten
    epochs = trainer.load_epochs_from_fif(epochs_path)

    # Training
    results = trainer.train(epochs, train_size=0.7)

    # Speichere Modell
    model_path = output_dir / f"{participant_name}_{session_name}_simple_eegnet.pth"
    trainer.save_model(model_path)

    # Speichere Confusion Matrix
    plot_path = output_dir / f"{participant_name}_{session_name}_confusion_matrix.png"
    trainer.plot_confusion_matrix(results, plot_path)

    summary = {
        "participant": participant_name,
        "session": session_name,
        "accuracy": results["accuracy"],
        "train_size": results["train_size"],
        "test_size": results["test_size"],
        "model_path": str(model_path),
        "plot_path": str(plot_path),
    }

    print(f"\\nTraining Summary:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Model saved: {model_path}")

    return summary


if __name__ == "__main__":
    # Test
    base_dir = Path(__file__).parent.parent.parent.parent
    epochs_path = (
        base_dir / "results" / "processed" / "Aliaa" / "indoor_processed-epo.fif"
    )
    output_dir = base_dir / "results" / "models"

    if epochs_path.exists():
        results = train_simple_eegnet(
            epochs_path=epochs_path,
            output_dir=output_dir,
            participant_name="Aliaa",
            session_name="indoor",
        )
        print(f"\\nResults: {results}")
    else:
        print(f"Epochs file not found: {epochs_path}")
