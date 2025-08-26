"""Einfaches EEGNet-ähnliches CNN (stark reduziert)."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Tuple
from config import settings


class SimpleEEGNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        reduced = n_samples // 4  # zwei Pooling-Schichten
        self.fc1 = nn.Linear(64 * reduced, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@dataclass
class TrainResult:
    model: nn.Module
    history: list[float]


class EpochDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_deep_model(
    X,
    y,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
) -> TrainResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if epochs is None:
        epochs = settings.DL_EPOCHS_DEFAULT
    if batch_size is None:
        batch_size = settings.DL_BATCH_SIZE
    if lr is None:
        lr = settings.DL_LEARNING_RATE

    n_epochs, n_channels, n_times = X.shape
    n_classes = len(set(y.tolist())) if hasattr(y, "tolist") else len(set(y))
    model = SimpleEEGNet(n_channels, n_times, n_classes).to(device)
    ds = EpochDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        history.append(running / len(ds))
    return TrainResult(model, history)
