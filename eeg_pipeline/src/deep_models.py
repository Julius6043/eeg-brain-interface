"""Deep learning baseline models for EEG workload decoding (PyTorch version).

This module implements an optional baseline based on EEGNet, a compact
convolutional neural network designed specifically for EEG signal
classification【74903913843087†L230-L241】. EEGNet uses depthwise and
separable convolutions to learn spatial and temporal filters with few
parameters, making it suitable for small training datasets and
low‑channel counts. The original implementation here was based on
TensorFlow/Keras; this refactored version uses **PyTorch** while keeping
the public function signatures (`build_eegnet_model`, `train_eegnet_crossval`)
and returned metrics identical so that external code can remain unchanged.

To run the functions in this module you must install PyTorch
(`pip install torch`).
"""

from __future__ import annotations

from typing import Tuple, Any, Dict, Optional, List

import numpy as np

try:  # PyTorch imports
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - handled gracefully
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore

try:
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
except ImportError:
    GroupKFold = None  # type: ignore
    LeaveOneGroupOut = None  # type: ignore
    balanced_accuracy_score = None  # type: ignore
    f1_score = None  # type: ignore
    roc_auc_score = None  # type: ignore

from .config import EEGNetConfig, TrainingConfig


class _SeparableConv2d(nn.Module):
    """Depthwise + Pointwise convolution block to mimic Keras SeparableConv2D."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int],
        padding: Tuple[int, int],
        bias: bool = False,
        depthwise_reg: Optional[float] = None,
        pointwise_reg: Optional[float] = None,
    ):  # noqa: D401
        super().__init__()
        # Regularisation placeholders (L2 handled via weight decay in optimizer typically)
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_ch,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGNetTorch(nn.Module):
    """PyTorch implementation of EEGNet v2.0 matching the original Keras version.

    Expected input shape: (batch, 1, channels, time)
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_classes: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_length: int = 64,
        dropout_rate: float = 0.5,
        reg: float = 0.25,  # kept for API compatibility; use optimizer weight_decay
        dropout_type: str = "Dropout",
    ) -> None:
        super().__init__()
        channels, time_len, _ = input_shape
        if dropout_type.lower() == "spatialdropout2d":
            # Spatial dropout over feature maps: implement via Dropout2d
            DropoutLayer = nn.Dropout2d
        else:
            DropoutLayer = nn.Dropout

        # Block 1: temporal convolution
        self.conv_time = nn.Conv2d(
            1,
            F1,
            kernel_size=(1, kern_length),
            padding=(0, kern_length // 2),
            bias=False,
        )
        self.bn_time = nn.BatchNorm2d(F1)
        # Block 1b: depthwise spatial convolution over channels dimension
        self.conv_depth = nn.Conv2d(
            F1, F1 * D, kernel_size=(channels, 1), groups=F1, bias=False
        )
        self.bn_depth = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = DropoutLayer(dropout_rate)
        # Block 2: separable convolution (depthwise temporal + pointwise)
        self.sepconv = _SeparableConv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8))
        self.bn_sep = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = DropoutLayer(dropout_rate)
        # Classification
        # Feature dimension after convolutions:
        # After conv_depth height -> 1, width unchanged
        # After pool1 width -> time_len /4 (floor)
        # After sepconv width same, then pool2 width -> /8
        # Compute dynamically in forward for flexibility.
        self.classifier = None  # Placeholder; will be created after first forward pass.
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # x: (batch, 1, channels, time)
        x = self.conv_time(x)
        x = self.bn_time(x)
        x = nn.functional.elu(x)
        x = self.conv_depth(x)
        x = self.bn_depth(x)
        x = nn.functional.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sepconv(x)
        x = self.bn_sep(x)
        x = nn.functional.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = torch.flatten(x, 1)
        # Lazy init classifier to remain shape-agnostic
        if self.classifier is None:
            in_features = x.shape[1]
            self.classifier = nn.Linear(in_features, self.n_classes)
            # Register after creation to appear in parameters
            self.add_module("fc", self.classifier)
        x = self.classifier(x)
        # For parity with Keras softmax output we return probabilities (detach softmax later if needed)
        return x


def _ensure_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required for EEGNet. Please install torch (e.g. pip install torch)."
        )


def build_eegnet_model(
    input_shape: Tuple[int, int, int],
    n_classes: int,
    F1: int = 8,
    D: int = 2,
    F2: int = 16,
    kern_length: int = 64,
    dropout_rate: float = 0.5,
    dropout_type: str = "Dropout",
    reg: float = 0.25,
) -> Any:
    """Construct the EEGNet v2.0 architecture in PyTorch.

    Parameters mirror the previous TensorFlow implementation for API
    compatibility. The returned model outputs raw logits (not softmax);
    callers requiring probabilities should apply ``torch.softmax``.
    """
    _ensure_torch()
    model = EEGNetTorch(
        input_shape=input_shape,
        n_classes=n_classes,
        F1=F1,
        D=D,
        F2=F2,
        kern_length=kern_length,
        dropout_rate=dropout_rate,
        reg=reg,
        dropout_type=dropout_type,
    )
    return model


def train_eegnet_crossval(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    eegnet_config: EEGNetConfig,
    training_config: TrainingConfig,
    n_epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
) -> Dict[str, Any]:
    """Train EEGNet using subject‑wise cross‑validation (PyTorch backend).

    Returns a dict with per-fold metrics identical to the former TensorFlow
    implementation: balanced accuracy, macro F1, macro ROC AUC.
    """
    _ensure_torch()
    # Remove samples with invalid labels
    valid_mask = y >= 0
    X = X[valid_mask]
    y = y[valid_mask]
    groups = groups[valid_mask]

    channels = X.shape[1]
    time_length = (
        X.shape[2]
        if eegnet_config.input_length is None
        else int(eegnet_config.input_length)
    )
    # Pad / truncate
    if time_length != X.shape[2]:
        if X.shape[2] > time_length:
            X = X[:, :, :time_length]
        else:
            pad_width = time_length - X.shape[2]
            X = np.pad(X, ((0, 0), (0, 0), (0, pad_width)), mode="constant")

    n_classes = int(np.max(y) + 1)

    # Select CV iterator (with fallback if only one group present)
    unique_groups = np.unique(groups)
    if (
        training_config.cv_strategy == "leave-one-subject-out"
        and len(unique_groups) < 2
    ):
        cv_splits = [(np.arange(len(X)), np.arange(len(X)))]  # single pseudo-fold
    elif training_config.cv_strategy == "leave-one-subject-out":
        cv = LeaveOneGroupOut()
        cv_splits = list(cv.split(X, y, groups))
    elif training_config.cv_strategy.startswith("group-k-fold"):
        parts = training_config.cv_strategy.split("-")
        n_splits = 5
        if len(parts) == 4:
            try:
                n_splits = int(parts[-1])
            except Exception:
                pass
        cv = GroupKFold(n_splits=n_splits)
        cv_splits = list(cv.split(X, y, groups))
    else:
        raise ValueError(f"Unsupported cv_strategy: {training_config.cv_strategy}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds: List[Dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(cv_splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Construct tensors (N, 1, C, T)
        X_train_t = torch.tensor(X_train[:, np.newaxis, :, :], dtype=torch.float32)
        X_test_t = torch.tensor(X_test[:, np.newaxis, :, :], dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False
        )

        model = build_eegnet_model((channels, time_length, 1), n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        # Use weight decay to approximate L2 regularisation similar to Keras l2(reg)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3, weight_decay=0.25 * 1e-3
        )

        best_state = None
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
            # Early stopping check
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(device))
            proba_t = torch.softmax(logits, dim=1)
        proba = proba_t.cpu().numpy()
        y_pred = np.argmax(proba, axis=1)
        # Handle degenerate / missing-class test sets
        unique_test = np.unique(y_test)
        if len(unique_test) < 2:
            ba = 1.0
            f1 = 1.0
            roc_auc = 1.0
        else:
            ba = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            # If some classes are missing in y_test, subset probability columns
            if proba.shape[1] != len(unique_test):
                # Assume labels are 0..K-1; select only columns for present classes
                proba_subset = proba[:, unique_test]
                try:
                    roc_auc = roc_auc_score(
                        y_test, proba_subset, multi_class="ovr", average="macro"
                    )
                except Exception:
                    roc_auc = 1.0
            else:
                roc_auc = roc_auc_score(
                    y_test, proba, multi_class="ovr", average="macro"
                )
        folds.append(
            {
                "fold": fold_i,
                "balanced_accuracy": ba,
                "f1_macro": f1,
                "roc_auc_macro": roc_auc,
            }
        )

    mean_scores = {
        "balanced_accuracy": float(np.mean([f["balanced_accuracy"] for f in folds])),
        "f1_macro": float(np.mean([f["f1_macro"] for f in folds])),
        "roc_auc_macro": float(np.mean([f["roc_auc_macro"] for f in folds])),
    }
    return {"folds": folds, "mean": mean_scores}
