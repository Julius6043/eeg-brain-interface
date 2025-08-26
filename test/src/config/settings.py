"""Zentrale Konfiguration & Default-Parameter für das Test-BCI-Projekt."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # bis zum Repo-Root
TEST_SRC_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TEST_SRC_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Preprocessing Parameter
RESAMPLE_SFREQ = 125
BANDPASS = (1.0, 40.0)
NOTCH_FREQS = [50]

# Epoching
TMIN = -0.1
TMAX = 0.4

# Feature / Model Defaults
CSP_COMPONENTS = 4
RANDOM_STATE = 42

# Deep Learning
DL_LEARNING_RATE = 1e-3
DL_BATCH_SIZE = 16
DL_EPOCHS_DEFAULT = 5  # bewusst klein
