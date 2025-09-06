"""Model package for EEG classification.

Dieses Package enthält verschiedene Deep Learning Modelle für EEG-Klassifikation:
- EEGNet_improved: Optimiertes EEGNet für 3-Klassen n-back mit intelligenter Baseline-Korrektur
- SimpleEEGNet: Vereinfachte PyTorch-Implementation von EEGNet
"""

from .EEGNet_improved import EEGNetTrainer, train_single_session_eegnet
from .SimpleEEGNet import SimpleEEGNetTrainer, train_simple_eegnet

__all__ = [
    "EEGNetTrainer",
    "train_single_session_eegnet",
    "SimpleEEGNetTrainer",
    "train_simple_eegnet",
]
