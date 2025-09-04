# --- Random Forest für einen einzelnen Teilnehmer (Indoor Session) ---
import numpy as np, pandas as pd, mne, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Konfiguration
PARTICIPANT = "Aliaa"  # Wähle einen Teilnehmer
SESSION_TYPE = "indoor"  # Nur Indoor Sessions

def load_single_participant_session(participant, session_type):
    processed_dir = Path("results/processed")

    epo_file = processed_dir / participant / f"{session_type}_processed-epo.fif"

    if not epo_file.exists():
        raise FileNotFoundError(f"Epoched file not found: {epo_file}")

    print(f"Loading data from: {epo_file}")
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)

    difficulty_mapping = {
        'baseline': 0,
        '0-back': 0,
        '1-back': 1,
        '2-back': 2,
        '3-back': 3
    }

    # Get labels from event names
    event_id_rev = {v: k for k, v in epochs.event_id.items()}
    event_names = [event_id_rev[event_id] for event_id in epochs.events[:, 2]]
    difficulties = [difficulty_mapping.get(name, -1) for name in event_names]

    # Filter out unknown events
    valid_indices = [i for i, d in enumerate(difficulties) if d >= 0]
    if len(valid_indices) < len(difficulties):
        print(f"Filtered out {len(difficulties) - len(valid_indices)} unknown events")
        epochs = epochs[valid_indices]
        difficulties = [difficulties[i] for i in valid_indices]

    # Create metadata
    metadata = pd.DataFrame({
        'difficulty': difficulties,
        'participant': [participant] * len(epochs),
        'session_type': [session_type] * len(epochs)
    })
    epochs.metadata = metadata

    print(f"Loaded {len(epochs)} epochs from {participant} ({session_type})")
    print(f"Event distribution: {dict(zip(*np.unique(difficulties, return_counts=True)))}")

    return epochs

def extract_features(epochs_data):
    """Extract bandpower features from epochs"""

    # Frequency bands for feature extraction
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 40),
    }

    # Filter data
    ep_filt = epochs_data.copy().filter(1.0, 40.0, picks="eeg")

    # Compute PSD
    try:
        psd = ep_filt.compute_psd(
            method="welch",
            fmin=1.0,
            fmax=40.0,
            n_fft=int(ep_filt.info["sfreq"] * 2),
            n_overlap=int(ep_filt.info["sfreq"] * 1),
            picks="eeg",
            verbose=False,
        )
        psds, freqs = psd.get_data(return_freqs=True)
    except Exception:
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

    # Calculate band masks
    bin_mask = {b: (freqs >= lo) & (freqs < hi) for b, (lo, hi) in bands.items()}
    total_pow = psds.sum(axis=2) + 1e-12

    # Extract relative bandpower features
    feat_list = []
    col_names = []

    for b, m in bin_mask.items():
        bp = psds[:, :, m].sum(axis=2)  # (n_epochs, n_channels)
        rel = bp / total_pow  # Relative power
        feat_list.append(rel)
        col_names += [f"{ch}_{b}" for ch in ep_filt.ch_names]

    X = np.concatenate(feat_list, axis=1)

    return X, col_names, ep_filt

# Hauptprogramm
print("=== Random Forest für einzelnen Teilnehmer ===")
print(f"Teilnehmer: {PARTICIPANT}")
print(f"Session: {SESSION_TYPE}")

# 1) Daten laden
epochs = load_single_participant_session(PARTICIPANT, SESSION_TYPE)

# 2) Features extrahieren
print("\n=== Feature Extraction ===")
X, col_names, ep_filt = extract_features(epochs)
y = epochs.metadata["difficulty"].astype(int).to_numpy()

print(f"Feature matrix: {X.shape} | Labels: {y.shape}")
print(f"Channels used: {ep_filt.ch_names}")
print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# 3) Feature preprocessing (optional)
print("\n=== Feature Preprocessing ===")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection (select top features if we have many)
if X_scaled.shape[1] > 20:
    selector = SelectKBest(score_func=f_classif, k=20)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = [col_names[i] for i in selector.get_support(indices=True)]
    print(f"Selected top {X_selected.shape[1]} features")
else:
    X_selected = X_scaled
    selected_features = col_names
    print(f"Using all {X_selected.shape[1]} features")

# 4) Random Forest Training
print("\n=== Random Forest Training ===")

# Check if we have enough samples for cross-validation
min_class_count = np.bincount(y).min()
n_splits = min(5, min_class_count) if min_class_count > 1 else 2

print(f"Using {n_splits}-fold cross-validation")

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# Cross-validation
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_selected, y, cv=cv, scoring="accuracy", n_jobs=-1)

print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
print(f"Individual fold scores: {np.round(scores, 3)}")

# Detailed evaluation
y_pred = cross_val_predict(rf, X_selected, y, cv=cv, n_jobs=-1)

print("\n=== Detailed Results ===")
print("Classification Report:")
print(classification_report(y, y_pred, digits=3))

cm = confusion_matrix(y, y_pred, labels=sorted(np.unique(y)))
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

# Majority class baseline
maj = np.bincount(y).argmax()
baseline_acc = (y == maj).mean()
print(f"\nMajority class baseline: {baseline_acc:.3f} (class {maj})")
print(f"Improvement over baseline: +{scores.mean() - baseline_acc:.3f}")

# 5) Feature Importance
print("\n=== Feature Importance ===")
rf.fit(X_selected, y)
importances = rf.feature_importances_

if len(selected_features) > 0:
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("Top 15 most important features:")
    print(feature_importance.head(15))

# 6) Save results
print("\n=== Saving Results ===")
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# Save model and preprocessing pipeline
model_data = {
    'model': rf,
    'scaler': scaler,
    'feature_selector': selector if X_scaled.shape[1] > 20 else None,
    'feature_names': selected_features,
    'participant': PARTICIPANT,
    'session_type': SESSION_TYPE,
    'cv_scores': scores
}

model_path = output_dir / f"rf_model_{PARTICIPANT}_{SESSION_TYPE}.joblib"
joblib.dump(model_data, model_path)

# Save features and metadata
features_df = pd.DataFrame(X_selected, columns=selected_features)
features_df['y'] = y
features_df['participant'] = PARTICIPANT
features_df['session_type'] = SESSION_TYPE

features_path = output_dir / f"rf_features_{PARTICIPANT}_{SESSION_TYPE}.csv"
features_df.to_csv(features_path, index=False)

# Save feature importance
if len(selected_features) > 0:
    importance_path = output_dir / f"rf_importance_{PARTICIPANT}_{SESSION_TYPE}.csv"
    feature_importance.to_csv(importance_path, index=False)

print(f"Model saved to: {model_path}")
print(f"Features saved to: {features_path}")
if len(selected_features) > 0:
    print(f"Feature importance saved to: {importance_path}")

# 7) Summary
print(f"\n=== Summary ===")
print(f"Participant: {PARTICIPANT}")
print(f"Session: {SESSION_TYPE}")
print(f"Total epochs: {len(epochs)}")
print(f"Features: {X_selected.shape[1]}")
print(f"Classes: {len(np.unique(y))}")
print(f"Best CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
print(f"Baseline accuracy: {baseline_acc:.3f}")

# Performance interpretation
if scores.mean() > baseline_acc + 0.1:
    print("✓ Model shows good discrimination ability")
elif scores.mean() > baseline_acc + 0.05:
    print("~ Model shows moderate discrimination ability")
else:
    print("⚠ Model performance is close to baseline - may need more data or better features")
