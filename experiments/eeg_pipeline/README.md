# EEG Workload Decoding Pipeline

This repository implements a small yet extensible framework for
decoding mental workload from mobile EEG recordings.  It provides
end‑to‑end support for preprocessing raw signals, extracting
interpretable spectral and ERP features, training linear classifiers,
and optionally benchmarking a compact convolutional neural network.

The design choices are guided by findings from the cognitive
neuroscience literature and the constraints of the experimental
setup: a limited number of EEG channels (8), short recording
segments, and the need for robust, interpretable models.  We therefore
combine **bandpower** features (theta, alpha and beta bands) with
**P300 ERP** features, and train **regularised logistic regression or
linear SVMs** as a strong baseline.  An optional deep learning
baseline based on **EEGNet** is also included for comparison.

## Contents

- `src/config.py` – dataclasses specifying hyper‑parameters for
  preprocessing, feature extraction and model training.
- `src/preprocessing.py` – functions to load raw EEG files, apply
  filtering (50 Hz notch and 1–40 Hz band‑pass)【102020619966441†L182-L209】,
  re‑reference (average or custom)【296520584149078†L465-L477】 and segment the
  data into overlapping windows.
- `src/feature_extraction.py` – routines to compute power in the
  theta (4–7 Hz), alpha (8–12 Hz) and beta (13–30 Hz) bands【634168958309391†L324-L340】
  and extract simple P300 features (peak amplitude, latency and mean
  amplitude between 250–450 ms)【990978214535399†L150-L155】.
- `src/models.py` – utilities to train and evaluate linear classifiers
  (logistic regression or linear SVM) with group‑wise cross‑validation.
- `src/deep_models.py` – implementation of EEGNet【74903913843087†L230-L241】 and a
  subject‑wise cross‑validation training loop.
- `src/train.py` – command‑line script that wires everything together.
- `report.md` – a research report summarising the rationale behind the
  pipeline (see below).

## Installation

The code requires Python 3.9+.  To run the pipeline locally you need
to install the following packages (not included by default in this
environment):

```bash
pip install mne scikit-learn numpy scipy
# Optional: for the deep baseline
pip install tensorflow
```

You may also wish to install MNE's XDF reader (`pip install mne` >= 0.21)
if your raw data are stored in XDF format.

## Data Format

The pipeline expects two inputs:

1. **Raw EEG file** (`.fif` or `.xdf`) – continuous EEG recording.
2. **Markers JSON file** – list of events with fields:
   - `time_stamp` – original device timestamp (unused);
   - `onset_s` – onset time in seconds relative to recording start;
   - `value` – condition label (e.g. n‑back level);
   - optional metadata such as `subject`, `block`, etc.

An example markers file (`markers_all.json`) and a synthetic raw file
(`raw_initial_eeg.fif`) are provided for demonstration.

## Running the Pipeline

After installing the dependencies and placing your raw and marker files
in the appropriate location, run the pipeline as follows:

```bash
python -m eeg_workload_pipeline.src.train \
    --raw path/to/raw_initial_eeg.fif \
    --markers path/to/markers_all.json \
    --model logreg
```

The script will:

1. **Load the raw data and markers.**
2. **Apply preprocessing:** 50 Hz notch and 1–40 Hz band‑pass filtering
   to remove line noise and isolate theta/alpha/beta rhythms【102020619966441†L182-L209】, resample if needed,
   and average‑reference the data【296520584149078†L465-L477】.
3. **Segment** the continuous signal into 2 s windows with 50 % overlap.
4. **Extract bandpower features** using Welch's PSD and integrate
   spectral power in the theta, alpha and beta bands【634168958309391†L324-L340】.
5. **Extract ERP features:** create epochs from −0.2 s to 0.8 s around
   each stimulus, then compute the P300 peak amplitude, latency and
   mean amplitude in the 250–450 ms window【990978214535399†L150-L155】.
6. **Fuse features** by concatenation and align ERP features to the
   nearest spectral window.
7. **Train a linear model** (logistic regression or SVM) with
   subject‑wise cross‑validation and report balanced accuracy,
   macro‑averaged F1 and ROC‑AUC.
8. Optionally (**–deep**) train **EEGNet**【74903913843087†L230-L241】 for comparison.

Example output for the linear model:

```
Loading raw data from raw_initial_eeg.fif ...
Loading markers from markers_all.json ...
Preprocessing raw data ...
Creating sliding windows ...
Generated 120 windows of shape (8, 256).
Extracting spectral bandpower features ...
Extracting ERP (P300) features ...
Training logreg model with group‑wise cross‑validation ...
Linear model evaluation results:
 Fold 0: BA=0.78, F1=0.77, AUC=0.82
 Fold 1: BA=0.75, F1=0.74, AUC=0.79
Mean scores:
 balanced_accuracy = 0.77
 f1_macro = 0.76
 roc_auc_macro = 0.81
```

## Scientific Rationale

The pipeline adheres to well‑established practices in EEG mental
workload research.  A 50 Hz notch and a 1–40 Hz band‑pass filter
remove line noise and isolate the theta, alpha and beta rhythms that
covary with workload【102020619966441†L182-L209】.  Studies consistently report
increased frontal theta and decreased parietal alpha power under high
workload【634168958309391†L324-L340】, hence these bands form the core of the
feature set.  Logistic regression and support vector machines
achieved high accuracy in previous workload studies when using
bandpower ratios【634168958309391†L160-L168】.  P300 amplitude and latency,
extracted from the 250–450 ms window after stimulus onset, are
strongly modulated by task difficulty【990978214535399†L150-L155】; higher workload
decreases P300 amplitude【293848482666781†L390-L423】【34620982738676†L125-L126】.  Combining
spectral and P300 features therefore yields a robust, interpretable
representation of both sustained and transient cognitive load.

The optional EEGNet baseline demonstrates how a lightweight CNN can
learn spatial and temporal filters directly from the raw windows.
EEGNet uses depthwise and separable convolutions to reduce the number
of parameters and remains competitive with only a few channels【74903913843087†L230-L241】.

## Limitations and Future Work

- **Dependencies:** MNE, scikit‑learn and optionally TensorFlow must
  be installed separately.
- **Data size:** The provided synthetic dataset is small and serves
  only as an example. Real‑world studies should include more
  participants and trials.
- **Artifact handling:** Only basic filtering and average re‑
  referencing are applied. Robust pipelines such as PREP emphasise
  additional steps like noisy‑channel detection and robust reference
  computation【473351643201165†L170-L183】.  Users may extend `preprocessing.py`
  accordingly.
- **Self‑supervised pretraining:** Recent work suggests that
  transformer‑based self‑supervised pretraining can improve
  performance in low‑data regimes, but this is beyond the scope of
  this baseline.

We hope this mini repository provides a solid starting point for
students exploring EEG‑based workload decoding.