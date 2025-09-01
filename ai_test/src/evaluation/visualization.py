"""Visualisierungs-Helfer (simple Beispiele)."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(losses):
    plt.figure(figsize=(4, 3))
    plt.plot(losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    return plt.gcf()


def plot_band_feature_hist(X, feature_names, limit: int = 10):
    plt.figure(figsize=(6, 4))
    for i in range(min(limit, X.shape[1])):
        plt.hist(X[:, i], alpha=0.5, label=feature_names[i])
    plt.legend(fontsize=6)
    plt.tight_layout()
    return plt.gcf()
