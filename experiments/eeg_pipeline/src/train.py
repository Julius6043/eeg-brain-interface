"""Entry point to run the EEG workload decoding pipeline.

This script orchestrates the full pipeline: loading raw EEG data and
markers, preprocessing, feature extraction, model training and
evaluation. It exposes a command‑line interface for convenience.

Example usage::

    python -m eeg_workload_pipeline.src.train \
        --raw /path/to/raw_initial_eeg.fif \
        --markers /path/to/markers_all.json \
        --model logreg

To include the optional deep learning baseline (EEGNet) add
``--deep``. Note that running the deep baseline requires
TensorFlow/Keras to be installed.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np

from .config import (
    PreprocessingConfig,
    FeatureConfig,
    ModelConfig,
    TrainingConfig,
    EEGNetConfig,
)
from .preprocessing import (
    load_raw_file,
    load_markers,
    preprocess_raw,
    create_sliding_windows,
    zscore_windows,
    create_epochs_for_erp,
)
from .feature_extraction import (
    compute_bandpower_features,
    compute_p300_features,
    fuse_features,
)
from .models import train_evaluate_linear_model
from .deep_models import train_eegnet_crossval


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EEG workload decoding pipeline")
    parser.add_argument(
        "--raw", required=True, help="Path to raw EEG file (.fif or .xdf)"
    )
    parser.add_argument("--markers", required=True, help="Path to JSON markers file")
    parser.add_argument(
        "--model", default="logreg", choices=["logreg", "svm"], help="Linear model type"
    )
    parser.add_argument("--deep", action="store_true", help="Include EEGNet baseline")
    args = parser.parse_args(argv)

    # Load data
    print(f"Loading raw data from {args.raw} ...")
    raw = load_raw_file(args.raw, preload=True)
    print(f"Loading markers from {args.markers} ...")
    markers = load_markers(args.markers)

    # Configuration
    preproc_cfg = PreprocessingConfig()
    feat_cfg = FeatureConfig()
    model_cfg = ModelConfig(model_type=args.model)
    train_cfg = TrainingConfig()
    eegnet_cfg = EEGNetConfig(
        sampling_rate=raw.info["sfreq"],
        input_channels=len(raw.ch_names),
    )

    # Preprocess raw
    print("Preprocessing raw data ...")
    raw_prep = preprocess_raw(raw, preproc_cfg)

    # Window segmentation
    print("Creating sliding windows ...")
    windows, labels, groups, window_times = create_sliding_windows(
        raw_prep,
        markers,
        feat_cfg.window_length,
        feat_cfg.window_overlap,
    )
    print(f"Generated {len(windows)} windows of shape {windows.shape[1:]}.")

    # Optional z‑scoring per window
    if preproc_cfg.zscore_per_subject:
        windows = zscore_windows(windows)

    # Compute bandpower features
    print("Extracting spectral bandpower features ...")
    band_features = compute_bandpower_features(
        windows,
        sfreq=raw_prep.info["sfreq"],
        bands=feat_cfg.bands,
    )

    # Compute ERP features
    print("Extracting ERP (P300) features ...")
    # Create epochs around markers
    epochs, events = create_epochs_for_erp(
        raw_prep,
        markers,
        feat_cfg.erp_interval[0],
        feat_cfg.erp_interval[1],
        event_repeated="drop",  # handle duplicate marker timestamps gracefully
    )
    p300_features = compute_p300_features(
        epochs,
        p300_window=feat_cfg.p300_window,
        picks=None,
    )
    # Align epochs to nearest window by time
    # For each epoch, find the window whose centre is closest to the event time
    epoch_onsets = (
        events[:, 0] / raw_prep.info["sfreq"]
    )  # convert sample index to seconds
    window_centres = window_times + feat_cfg.window_length / 2
    align_indices = []
    for onset in epoch_onsets:
        idx = int(np.argmin(np.abs(window_centres - onset)))
        align_indices.append(idx)

    fused_features = fuse_features(
        band_features, p300_features, align_indices=align_indices
    )

    # Remove windows with invalid label (-1) for training
    valid_mask = labels >= 0
    X = fused_features[valid_mask]
    y = labels[valid_mask]
    grp = groups[valid_mask]

    # Train linear model
    print(f"Training {args.model} model with group‑wise cross‑validation ...")
    results = train_evaluate_linear_model(X, y, grp, model_cfg, train_cfg)
    print("Linear model evaluation results:")
    for fold in results["folds"]:
        print(
            f" Fold {fold['fold']}: BA={fold['balanced_accuracy']:.3f}, F1={fold['f1_macro']:.3f}, AUC={fold['roc_auc_macro']:.3f}"
        )
    print("Mean scores:")
    for metric, value in results["mean"].items():
        print(f" {metric} = {value:.3f}")

    # Optionally train deep model
    if args.deep:
        try:
            print("Training EEGNet baseline ...")
            deep_results = train_eegnet_crossval(
                windows, labels, groups, eegnet_cfg, train_cfg
            )
            print("EEGNet evaluation results:")
            for fold in deep_results["folds"]:
                print(
                    f" Fold {fold['fold']}: BA={fold['balanced_accuracy']:.3f}, F1={fold['f1_macro']:.3f}, AUC={fold['roc_auc_macro']:.3f}"
                )
            print("Mean scores:")
            for metric, value in deep_results["mean"].items():
                print(f" {metric} = {value:.3f}")
        except ImportError as e:
            print(f"Skipping EEGNet baseline: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
