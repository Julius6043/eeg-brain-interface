"""Tests for model training functions."""

import numpy as np
import pytest

try:
    import sklearn
except ImportError:
    sklearn = None

from experiments.eeg_pipeline.src.models import train_evaluate_linear_model
from experiments.eeg_pipeline.src.config import ModelConfig, TrainingConfig


@pytest.mark.skipif(sklearn is None, reason="scikit-learn is required for this test")
def test_train_evaluate_linear_model_produces_metrics():
    """train_evaluate_linear_model should return metrics dictionary."""
    # Simple linearly separable dataset
    X = np.vstack(
        [
            np.random.randn(10, 2) + [2, 2],  # class 0
            np.random.randn(10, 2) + [-2, -2],  # class 1
        ]
    )
    y = np.array([0] * 10 + [1] * 10)
    # Create group labels: two groups (subject 0 and subject 1) of equal size
    groups = np.array([0] * 10 + [1] * 10)
    model_cfg = ModelConfig(model_type="logreg")
    train_cfg = TrainingConfig(cv_strategy="leave-one-subject-out")
    results = train_evaluate_linear_model(X, y, groups, model_cfg, train_cfg)
    # Ensure 'folds' and 'mean' keys exist
    assert "folds" in results and "mean" in results
    assert len(results["folds"]) == 2
    # Check that metrics are reasonable (balanced accuracy between 0 and 1)
    for fold in results["folds"]:
        assert 0.0 <= fold["balanced_accuracy"] <= 1.0
        assert 0.0 <= fold["f1_macro"] <= 1.0
        assert 0.0 <= fold["roc_auc_macro"] <= 1.0
