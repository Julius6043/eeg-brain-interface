# Final Submission Deep Learning Notebook – Structured Summary

## 1. Objective

End-to-end deep learning workflow for 3-class N-Back EEG (1‑back, 2‑back, 3‑back) for a single participant/session, including:

- Loading preprocessed MNE epochs (+ baseline segments)
- Baseline-driven z-normalization
- Exploratory visualization (class balance, raw epoch trace, augmentation effect, PSD)
- Cross-validation (Stratified K-Fold) with augmentation + class weighting
- Final model training on all data (optional hold-out split with early stopping)
- Cross-session generalization test (indoor ↔ outdoor)
- Channel importance estimation via occlusion (Δ accuracy)
- Independent “simple phase-based” train/test experiment to rule out leakage and replicate importance

## 2. Data Handling & Preprocessing

Steps:

1. Load processed epochs file: `<participant>/<session>_processed-epo.fif` from `results/`.
2. Split into baseline vs task epochs (labels: 'baseline', '1-back', '2-back', '3-back').
3. Normalize: Fit `StandardScaler` on concatenated baseline samples per channel; apply to task epochs.
4. Relabel events to compact integer targets {0,1,2} and attach as `metadata['target']`.
5. (Later simple split) Alternative raw-based segmentation using fixed-length windows inside annotated blocks.

Key Functions:

- `load_and_prepare_data()` – loads MNE epochs, separates baseline/task.
- `normalize_epochs_with_baseline()` – channel-wise z-score using baseline.
- `add_metadata_with_targets()` – remaps events and builds metadata.

## 3. Data Augmentation

`augment_eeg_data()` creates synthetic variants (factor configurable):

- Gaussian noise (σ ≈ 0.03)
- Small temporal circular shifts (±≤5 samples)
- Per-channel amplitude scaling (±10%)
- Rare channel attenuation (10% probability) to simulate dropout
  Concatenates original + augmented epochs into a new `EpochsArray`.

Used:

- In CV: only training folds are augmented.
- In final model: full training set (or training subset when validation split is used) augmented.
- Not applied to evaluation/test splits.

## 4. Exploratory Visualization

Produced Figures:

- Bar chart of class counts.
- Multi-channel raw trace of a single epoch (first 5 channels, vertically offset).
- Augmentation effect overlay (original vs augmented channel waveform).
- Welch PSD (log scale) averaged per class.

## 5. Cross-Validation Training

Configuration (example defaults):

- Splits: `n_splits = 3` StratifiedKFold (shuffle, seed=42)
- Model variants supported: `EEGNet`, `ShallowFBCSPNet`, `ATCNet`, `AttentionBaseNet` (selected via `MODEL_NAME`).
- Windowing: Non-overlapping full-epoch windows (one window per epoch).
- Loss: CrossEntropy with class weights + label smoothing (0.15).
- Optimizer: AdamW (lr=5e-4 or 5e-4/5e-4 vs config, weight_decay=0.005).
- Scheduler: CosineAnnealingLR (T_max = max_epochs - 1).
- Augmentation: Factor = 3 (training only).
- Metrics captured: validation accuracy per fold, confusion matrix, classification report, macro-F1.
- Learning curves: Train loss and validation accuracy plotted per fold; bar plot of fold accuracies.

Outputs:

- Per-fold validation accuracies and macro-F1 (not persisted to disk, printed & plotted).
- Confusion matrices (per fold) visualized via heatmaps.

## 6. Final Model Training

`train_final_model()`:

- Optionally creates a validation split (StratifiedShuffleSplit on "window" labels) if `final_valid_fraction > 0`.
- Augments only the training portion; validation uses original data.
- Adds EarlyStopping on `valid_loss` (patience configurable) when validation split active.
- Logs training loss history (and validation metrics if split exists).
- After training: computes window-level training accuracy + confusion matrix (absolute & normalized).

## 7. Cross-Session Evaluation

- Loads the opposite session (indoor ↔ outdoor) for same participant.
- Re-applies normalization using that session’s baseline.
- Evaluates the already-trained final model (no fine-tuning) for generalization.
- Produces confusion matrices (absolute & normalized) and classification report.

## 8. Channel Importance (Occlusion Analysis)

`channel_importance_drop()` procedure:

1. Compute baseline accuracy on unmodified epochs.
2. For each channel: replace its data with zero-scaled version (multiplicative mask = 0.0).
3. Recompute accuracy; ΔAcc = baseline_acc − masked_acc.
4. Plot bar chart of ΔAcc; print ordered list of influence.

Repeated for:

- Full CV/final pipeline (using normalized `task_ready`).
- Simple phase-based split test (using `test_epochs`).

Interpretation: Larger ΔAcc indicates greater model reliance on that channel (sensitivity, not causal importance).

## 9. Simple Phase-Based Train/Test Split Experiment

Motivation: Provide an alternative evaluation path with explicit temporal separation (first block per class -> train, second block -> test) and confirm absence of leakage.
Procedure:

- Reads raw file `<session>_processed_raw.fif`.
- Identifies exactly two annotated segments per task class; first used for training, second for testing.
- Baseline segments included in training indices.
- Fixed-length epoch segmentation with optional overlap (train uses overlap; test non-overlapping).
- Baseline normalization identical in spirit to main pipeline.
- Trains `AttentionBaseNet` with a validation split referencing test set (`predefined_split`).
- Reports accuracy, confusion matrices, classification report.
- Performs occlusion-based channel importance on test set.

## 10. Models & Hyperparameters (Key Fields)

Main config fields: learning rate, batch size, max epochs, kernel length (for EEGNet), F1/D/F2 filter params, dropout_rate, weight_decay, patience, augmentation_factor, bandpass (filter_low/high placeholders though filtering not explicitly applied in provided code), label_smoothing, warmup_epochs (not explicitly used in scheduler logic yet), early_stopping parameters.

## 11. Reproducibility Measures

- Global seeding for Python, NumPy, PyTorch (CPU & CUDA), and cuDNN deterministic settings.
- Stratified splits with fixed random_state.

## 12. Outputs & Metrics (In-Notebook)

- Printed environment versions (PyTorch, CUDA availability, MNE, Braindecode).
- Fold metrics and final training accuracy.
- Cross-session accuracy.
- Channel occlusion ΔAcc tables.
- Multiple diagnostic plots (counts, raw epoch, augmentation sample, PSD, learning curves, confusion matrices, channel importance).

## 13. Not Persisted / Missing Persistence

- No saving of trained model weights to disk.
- No CSV/JSON logging of results (only console & plots).
- Channel importance values not exported.

## 14. High-Level Pipeline Flow

1. Load + normalize → metadata target assignment.
2. (Optional) Exploratory visualizations.
3. Cross-validation with augmentation + metrics.
4. Train final model on full set (+ optional validation / early stopping).
5. Evaluate generalization on opposite session.
6. Perform channel occlusion analysis.
7. Run alternative simple split experiment + its own occlusion.

## 15. Key Strengths

- Clear modular functions for loading, normalization, augmentation, model instantiation.
- Multiple evaluation paradigms (K-Fold and temporal split).
- Use of class weighting + label smoothing to stabilize training under imbalance.
- Occlusion-based interpretability included.
- Consistent seeding for reproducibility.

## 16. Limitations (See separate Improvements document for detail)

- No external logging/persistence.
- Potential redundancy in normalization / metadata functions across sections.
- Filtering parameters defined but not actually applied (no explicit bandpass call).
- Limited augmentation diversity (no frequency-domain or mixup approaches).
- Warmup parameter unused.

---

This document summarizes the current functionality and structure of the Deep Learning submission notebook for rapid onboarding and review.
