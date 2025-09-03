# --- Random Forest on bandpower features ---
import numpy as np, pandas as pd, mne, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Keep only the first 8 channels for decoding
N = min(8, len(epochs.ch_names))
picks8 = epochs.ch_names[:N]
ep8 = epochs.copy().pick(picks=picks8)
print(f"Using {N} channels:", picks8)

epochs = ep8  # use only these channels


# 1) Load epochs if needed
if "epochs" not in globals():
    assert "EPO_FIF" in globals(), "EPO_FIF path missing; run epoching or set EPO_FIF."
    print("Loading epochs from:", EPO_FIF)
    epochs = mne.read_epochs(EPO_FIF, preload=True)

assert (
    epochs.metadata is not None and "difficulty" in epochs.metadata.columns
), "epochs.metadata must contain 'difficulty'."
print(epochs)

# 2) Filter for stable bandpowers
ep_filt = epochs.copy().filter(1.0, 40.0, picks="eeg")

# 3) PSD per epoch (new API first; fallback to old if needed)
try:
    # MNE >= 1.2 style
    psd = ep_filt.compute_psd(
        method="welch",
        fmin=1.0,
        fmax=40.0,
        n_fft=int(ep_filt.info["sfreq"] * 2),
        n_overlap=int(ep_filt.info["sfreq"] * 1),
        picks="eeg",
        verbose=False,
    )
    psds, freqs = psd.get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs)
except Exception:
    # Older MNE fallback (only if your env still has it)
    from mne.time_frequency import psd_welch

    psds, freqs = psd_welch(
        ep_filt,
        fmin=1.0,
        fmax=40.0,
        n_fft=int(ep_filt.info["sfreq"] * 2),
        n_overlap=int(ep_filt.info["sfreq"] * 1),
        picks="eeg",
        average="mean",
        n_per_seg=None,
        verbose=False,
    )


# 4) Band definitions and integration
bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}
bin_mask = {b: (freqs >= lo) & (freqs < hi) for b, (lo, hi) in bands.items()}
total_pow = psds.sum(axis=2) + 1e-12  # avoid div by zero

# Build feature matrix: relative bandpower per channel
feat_list = []
col_names = []
for bi, (b, m) in enumerate(bin_mask.items()):
    bp = psds[:, :, m].sum(axis=2)  # (n_epochs, n_channels)
    rel = bp / total_pow
    feat_list.append(rel)
    col_names += [f"{ch}_{b}" for ch in ep_filt.ch_names]
X = np.concatenate(feat_list, axis=1)  # shape: (n_epochs, n_channels * n_bands)
y = ep_filt.metadata["difficulty"].astype(int).to_numpy()

print("Feature matrix:", X.shape, "| labels:", y.shape)

# 5) Random Forest
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

cv = StratifiedKFold(
    n_splits=min(5, np.bincount(y).min() if len(np.unique(y)) > 1 else 2),
    shuffle=True,
    random_state=42,
)
scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(
    "RF CV accuracy:",
    np.round(scores, 3),
    " | mean±sd:",
    f"{scores.mean():.3f} ± {scores.std():.3f}",
)

y_pred = cross_val_predict(rf, X, y, cv=cv, n_jobs=-1)
print("\nClassification report:\n", classification_report(y, y_pred, digits=3))

cm = confusion_matrix(y, y_pred, labels=sorted(np.unique(y)))
print("Confusion matrix (rows=true, cols=pred):\n", cm)

maj = np.bincount(y).argmax()
print(f"Majority-class baseline: {(y==maj).mean():.3f} (label={maj})")

# 6) Fit on full data and report top features
rf.fit(X, y)
importances = rf.feature_importances_
topk = np.argsort(importances)[::-1][:20]
top_table = pd.DataFrame(
    {"feature": [col_names[i] for i in topk], "importance": importances[topk]}
)
print("\nTop 20 features by importance:")
display(top_table)

# 7) Save artifacts (optional)
MODEL_PATH = (
    EPO_FIF.with_suffix("").with_name(f"{OUT_BASENAME}_rf_bandpower.joblib")
    if "OUT_BASENAME" in globals()
    else Path("rf_bandpower.joblib")
)
FEAT_PATH = MODEL_PATH.with_suffix(".features.csv")
joblib.dump(
    {"model": rf, "bands": bands, "ch_names": ep_filt.ch_names, "col_names": col_names},
    MODEL_PATH,
)
pd.DataFrame(X, columns=col_names).assign(y=y).to_csv(FEAT_PATH, index=False)
print("Saved model to:", MODEL_PATH)
print("Saved features to:", FEAT_PATH)


# 8) Helper: function to transform new epochs -> features
def extract_bandpower_features(ep: mne.Epochs) -> np.ndarray:
    ep2 = ep.copy().filter(1.0, 40.0, picks="eeg")
    psd, fr = mne.time_frequency.psd_welch(
        ep2,
        fmin=1.0,
        fmax=40.0,
        n_fft=int(ep2.info["sfreq"] * 2),
        n_overlap=int(ep2.info["sfreq"] * 1),
        picks="eeg",
        average="mean",
        n_per_seg=None,
        verbose=False,
    )
    bm = {b: (fr >= lo) & (fr < hi) for b, (lo, hi) in bands.items()}
    tp = psd.sum(axis=2) + 1e-12
    feats = []
    for b, m in bm.items():
        bp = psd[:, :, m].sum(axis=2)
        rel = bp / tp
        feats.append(rel)
    return np.concatenate(feats, axis=1)
