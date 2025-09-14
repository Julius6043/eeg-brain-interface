# Final Submission Deep Learning Notebook – Improvements, Open Issues & Future Outlook

This document enumerates technical debt, potential pitfalls, and forward-looking enhancements for the EEG N‑Back deep learning pipeline described in the accompanying summary.

## 1. Data & Preprocessing

### 1.1 Baseline Normalization

- Current approach flattens all baseline samples per channel and applies a global StandardScaler.
- Potential Issue: Assumes baseline stationarity and equal relevance across time; does not account for inter-epoch drift.
- Improvement Ideas:
  - Per-epoch adaptive re-scaling using a sliding baseline (if available).
  - Use robust scaling (median/IQR) to mitigate outlier baseline segments.
  - Apply frequency-band specific normalization (e.g., z-score alpha power separately) if physiologically motivated.

### 1.2 Filtering Parameters Unused

- Config includes `filter_low` / `filter_high` but no filtering call is executed.
- Risk: Mismatch between documentation/intent and actual preprocessing; data may contain slow drifts or high-frequency noise.
- Action: Explicitly apply `epochs.filter(l_freq, h_freq)` or document why filtering is omitted (e.g., already filtered upstream).

### 1.3 Event / Metadata Duplication

- There are two implementations of `add_metadata_with_targets()` (main + simple split section) with slight variations.
- Risk: Divergence over time or subtle inconsistencies.
- Action: Consolidate into a single utility imported from a module (e.g., `src/preprocessing/labels.py`).

### 1.4 Raw-Based Simple Split Assumptions

- The simple phase-based split assumes exactly two contiguous segments per task label.
- Failure Mode: If annotation count deviates (missing or extra segments) the code raises an error.
- Improvement: Add graceful fallback (e.g., take first two chronologically), or generalize to variable number and enforce temporal partitioning.

### 1.5 Overlapping Segmentation (Train Only)

- Overlap used only for training in simple split; may inflate sample count without increasing true diversity.
- Improvement: Consider consistent epoch logic across main CV pipeline (currently uses whole epochs only). Evaluate effect of temporal windowing / stride factor as hyperparameter.

### 1.6 Baseline Channel Masking in Occlusion

- Occlusion masking multiplies by zero; this may create an artificial distribution (hard zeros) unseen in normalization.
- Improvement: Replace with channel-wise Gaussian noise drawn from baseline statistics or mean-fill + noise to better emulate plausible absence.

## 2. Augmentation Strategy

### 2.1 Limited Transform Diversity

- Only noise, small temporal shift, amplitude scaling, rare channel attenuation.
- Future Enhancements:
  - Frequency-domain perturbations (random notch, band scaling, spectral mixup).
  - Time-warping, random cropping within fixed window, phase shifting.
  - Mixup / CutMix between epochs to regularize decision boundaries.
  - Channel dropout with stochastic reconstruction via interpolation.

### 2.2 Potential Label Noise Sensitivity

- Augmentation does not enforce plausibility constraints (e.g., excessive shift could misalign cognitive response windows for shorter epochs in other datasets).
- Action: Parameterize augmentation severity; store metadata about synthetic provenance for ablation.

### 2.3 Lack of Augmentation Ablation

- No experiment measuring incremental benefit of augmentation factor.
- Future Work: Run controlled experiments (factor ∈ {1,2,3,4}) and log accuracy/F1 to diminishing returns curve.

## 3. Model Architecture & Training

### 3.1 Warmup Parameter Unused

- Config contains `warmup_epochs`, but no scheduler implements warmup.
- Action: Integrate cosine warm restarts with warmup ramp, or remove parameter to avoid confusion.

### 3.2 Early Stopping Only in Final Model (Optional Path)

- Cross-validation folds run full `max_epochs` without early stopping.
- Improvement: Add optional early stopping inside CV to reduce overfitting variance and accelerate runtime.

### 3.3 Hyperparameter Centralization

- Same hyperparameters replicated across sections (CV vs final vs simple split).
- Action: Single `CONFIG` dict in a Python module; pass subset overrides to functions.

### 3.4 Class Weight Computation Scope

- Class weights computed from global epoch distribution; when heavy augmentation applied, effective training distribution changes.
- Improvement: Recompute or scale weights after augmentation or switch to focal loss to dynamically handle imbalance.

### 3.5 No Model Checkpointing

- Trained models are not saved; reproducibility restricted to single session run.
- Action: Add serialization: `torch.save(model.state_dict(), path)` and optionally export ONNX for portability.

### 3.6 Validation Metric Coverage

- Only accuracy + macro F1 aggregated; no per-class temporal stability or calibration metric.
- Enhancements:
  - Log balanced accuracy, per-class recall curves.
  - Reliability diagrams + Expected Calibration Error (ECE).
  - AUC per one-vs-rest (if probabilities used).

### 3.7 Batch Normalization / Domain Shift

- Cross-session generalization tested, but no domain adaptation (e.g., adaptive batch-stat updates) or channel alignment.
- Future Direction: Domain adversarial training (DANN), CORAL loss, or statistical matching of baseline distributions across sessions.

## 4. Evaluation & Interpretation

### 4.1 Single Random Seed

- While seeds ensure determinism, one seed ≠ robustness.
- Improvement: Multi-seed runs (e.g., n=5) with confidence intervals.

### 4.2 Limited Generalization Scope

- Only intra-participant cross-session. No inter-participant adaptation or pooled training.
- Future: Leave-one-subject-out (LOSO) evaluation for generalization claims.

### 4.3 Occlusion Analysis Limitations

- Accuracy drop aggregation only; lacks statistical confidence.
- Improvement: Bootstrap ΔAcc per channel; add permutation importance; complement with gradient-based saliency (e.g., Integrated Gradients, Layer-wise Relevance Propagation for EEGNet).

### 4.4 Potential Data Leakage Risk Review

- CV uses full epochs with normalization fit on all baseline samples (which might include temporal regions contiguous with task segments). Usually acceptable, but ensure baseline segments are not contaminated with cognitive activity.
- Action: Verify baseline annotation purity; consider splitting baseline by fold to mimic strictly unseen normalization context.

### 4.5 Window-Level vs Epoch-Level Metrics

- Evaluation is performed at window-level (1 window = 1 epoch). If future segmentation differs, need aggregation logic.
- Action: Add epoch-level majority vote aggregator if multiple windows introduced.

## 5. Logging, Reproducibility & MLOps

### 5.1 Missing Persistent Logs

- No CSV/JSON or experiment tracking.
- Action: Introduce minimal logging (e.g., `results/metrics_cv.json`, `results/channel_importance.csv`). Optionally integrate MLflow or Weights & Biases.

### 5.2 Environment Recording

- Only library versions printed to stdout.
- Action: Persist environment snapshot (pip freeze) + hash of key source files for provenance.

### 5.3 Determinism vs Performance

- `torch.backends.cudnn.deterministic = True` may slow training.
- Future: Offer flag to toggle determinism vs speed.

## 6. Code Quality & Structure

### 6.1 Notebook-Centric Logic

- Mixing definition + execution makes reuse harder.
- Action: Move functions to `src/` modules (`data.py`, `augment.py`, `models.py`, `evaluation.py`), keep notebook as orchestrator.

### 6.2 Redundant Imports

- Some imports repeated (e.g., `StratifiedKFold` imported twice).
- Improvement: Clean import blocks; enforce isort/black.

### 6.3 Type Hints & Docstrings

- Partial hints present; docstrings mostly in first half.
- Action: Add full NumPy/Sphinx style docstrings with parameter ranges, expected shapes.

### 6.4 Error Handling Granularity

- Wide try/except blocks re-raise but don’t enrich context.
- Improvement: Provide user-level messages; optionally fallback suggestions.

## 7. Performance Considerations

### 7.1 Augmentation Efficiency

- Augmentation loops are pure Python; scaling may become a bottleneck with larger datasets.
- Action: Vectorize operations or implement on GPU / using PyTorch transforms.

### 7.2 Memory Footprint

- Concatenate augmented epochs in-memory; for large participant pools this will not scale.
- Future: On-the-fly augmentation during batching.

### 7.3 Mixed Precision

- Not used; potential speedup on GPU (fp16) with `torch.cuda.amp`.

## 8. Extended Research Directions

| Direction                                   | Rationale                                | First Step                                         |
| ------------------------------------------- | ---------------------------------------- | -------------------------------------------------- |
| Multi-subject pretraining                   | Improve generalization                   | Aggregate participants, add subject embedding      |
| Self-supervised (contrastive) pretext tasks | Leverage unlabeled baseline/other tasks  | SimCLR-style on epoch segments                     |
| Temporal attention / transformer blocks     | Capture longer dependencies              | Replace first conv stage with temporal transformer |
| Frequency-aware models (e.g., TFCNet)       | Exploit joint time-frequency structure   | Add learnable wavelet/sinc filters                 |
| Domain adaptation across sessions           | Mitigate session/environment shifts      | Adversarial discriminator on session ID            |
| Continual learning                          | Adapt model over time without forgetting | EWC or replay buffers                              |
| Calibration & uncertainty                   | Reliable decision support                | Temperature scaling + ensemble                     |
| Explainability                              | Trust & insight                          | Combine occlusion with layer-wise relevance        |

## 9. Risk Register

| Risk                                       | Impact                    | Mitigation                                        |
| ------------------------------------------ | ------------------------- | ------------------------------------------------- |
| Hidden data leakage via baseline overlap   | Inflated metrics          | Audit annotations; isolate baseline per fold      |
| Overfitting due to aggressive augmentation | Unrealistic performance   | Ablation + validation curves                      |
| Channel masking artifacts                  | Misinterpreted importance | Use noise-injection or distributional replacement |
| Single-participant conclusions             | Limited external validity | Add cross-subject evaluation                      |
| Non-saved models                           | Irreproducibility         | Implement checkpointing & versioning              |

## 10. Immediate Action Checklist (Short-Term Wins)

- [ ] Centralize configs & remove duplicates
- [ ] Apply actual bandpass filtering or document omission
- [ ] Add model saving + metrics JSON export
- [ ] Implement early stopping in CV
- [ ] Add augmentation ablation study (flag controlled)
- [ ] Refactor duplicated metadata & normalization functions
- [ ] Export channel importance values to CSV
- [ ] Remove/implement warmup parameter

## 11. Medium-Term Enhancements

- [ ] Introduce experiment tracking (MLflow/W&B)
- [ ] Add multi-seed evaluation harness
- [ ] Implement on-the-fly augmentation (torch Dataset wrapper)
- [ ] Adopt mixed precision training when GPU present
- [ ] Add calibration evaluation (ECE)

## 12. Long-Term Research

- [ ] Self-supervised pretraining
- [ ] Domain adaptation / session invariance
- [ ] Transformer or hybrid spectral-temporal architectures
- [ ] Multi-subject generalization with subject embeddings
- [ ] Explainability beyond occlusion (LRP, IG)

---

**Summary:** The current notebook provides a solid, transparent baseline with reproducible CV and interpretability. Prioritizing configuration consolidation, persistence, evaluation robustness, and augmentation validation will yield immediate robustness gains. Longer-term, scaling to multi-subject, domain-adaptive, and self-supervised paradigms will significantly elevate research value.
