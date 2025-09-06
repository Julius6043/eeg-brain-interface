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
PARTICIPANT = "julian"  # Wähle einen Teilnehmer
SESSION_TYPE = "outdoor"  # Nur Indoor Sessions

def load_single_participant_session(participant, session_type):
    processed_dir = Path("results/processed")

    epo_file = processed_dir / participant / f"{session_type}_processed-epo.fif"

    if not epo_file.exists():
        raise FileNotFoundError(f"Epoched file not found: {epo_file}")

    print(f"Loading data from: {epo_file}")
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)

    difficulty_mapping = {
        #'baseline': 0,
        #'0-back': 0,
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
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 40),
    }

    # Filter data
    ep_filt = epochs_data.copy().filter(4.0, 40.0, picks="eeg")

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
    n_estimators=10000,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=6,
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
    
    # 5) Feature Importance + MI + f_classif comparison
print("\n=== Feature Relevance Comparison (RF vs MI vs f_classif) ===")
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr

# 5a) (re)fit RF on the data you're evaluating (X_selected) to get impurity importances
rf.fit(X_selected, y)
rf_imp = pd.Series(rf.feature_importances_, index=selected_features, name="rf_importance")

# 5b) Compute MI and f_classif on the same feature space (X_selected)
#     NOTE: We use the *scaled* features; slice to selected columns for a fair comparison.
#     Build a mapping from feature -> column index in the *scaled* matrix
feat_idx_map = {f: i for i, f in enumerate(col_names)}
sel_idx = [feat_idx_map[f] for f in selected_features]
X_scaled_sel = X_scaled[:, sel_idx]

mi = mutual_info_classif(X_scaled_sel, y, random_state=42)
f_vals, p_vals = f_classif(X_scaled_sel, y)

mi_s = pd.Series(mi, index=selected_features, name="mi")
f_s  = pd.Series(f_vals, index=selected_features, name="f_classif")

# 5c) Merge into one table and compute ranks (higher is better)
comp_df = pd.concat([rf_imp, mi_s, f_s], axis=1)
rank_df = comp_df.rank(ascending=False, method="average").add_suffix("_rank")

# 5d) Spearman rank correlations between methods
rho_rf_mi, p_rf_mi = spearmanr(rank_df["rf_importance_rank"], rank_df["mi_rank"])
rho_rf_f,  p_rf_f  = spearmanr(rank_df["rf_importance_rank"], rank_df["f_classif_rank"])
rho_mi_f,  p_mi_f  = spearmanr(rank_df["mi_rank"], rank_df["f_classif_rank"])

print(f"Spearman(rank RF vs MI):  rho={rho_rf_mi:.3f}, p={p_rf_mi:.3g}")
print(f"Spearman(rank RF vs F ):  rho={rho_rf_f:.3f}, p={p_rf_f:.3g}")
print(f"Spearman(rank MI vs F ):  rho={rho_mi_f:.3f}, p={p_mi_f:.3g}")

# 5e) Show top features side-by-side (by RF importance)

# === Global feature relevance on ALL features (RF vs MI vs f_classif) ===
print("\n=== Global Feature Relevance (ALL features) ===")
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr

# Use ALL features: X_scaled with col_names
X_all = X_scaled
all_features = col_names

# Fit a fresh RF on ALL features (same hyperparams as above)
rf_all = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=6,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_all.fit(X_all, y)

# 1) RF impurity importance
rf_imp_all = pd.Series(rf_all.feature_importances_, index=all_features, name="rf_importance")

# 2) Mutual Information & 3) f_classif (on ALL features)
mi_all = mutual_info_classif(X_all, y, random_state=42)
f_vals_all, p_vals_all = f_classif(X_all, y)

mi_s_all = pd.Series(mi_all, index=all_features, name="mi")
f_s_all  = pd.Series(f_vals_all, index=all_features, name="f_classif")
p_s_all  = pd.Series(p_vals_all, index=all_features, name="f_pvalue")

# Merge & rank
comp_all = pd.concat([rf_imp_all, mi_s_all, f_s_all, p_s_all], axis=1)
rank_all = comp_all[["rf_importance", "mi", "f_classif"]].rank(ascending=False, method="average")
rank_all.columns = [c + "_rank" for c in rank_all.columns]

# Rank correlations
rho_rf_mi, p_rf_mi = spearmanr(rank_all["rf_importance_rank"], rank_all["mi_rank"])
rho_rf_f,  p_rf_f  = spearmanr(rank_all["rf_importance_rank"], rank_all["f_classif_rank"])
rho_mi_f,  p_mi_f  = spearmanr(rank_all["mi_rank"], rank_all["f_classif_rank"])
print(f"Spearman(rank RF vs MI):  rho={rho_rf_mi:.3f}, p={p_rf_mi:.3g}")
print(f"Spearman(rank RF vs F ):  rho={rho_rf_f:.3f}, p={p_rf_f:.3g}")
print(f"Spearman(rank MI vs F ):  rho={rho_mi_f:.3f}, p={p_mi_f:.3g}")

# Combine & sort for viewing; keep ALL features
comp_all_out = pd.concat([comp_all, rank_all], axis=1).sort_values("rf_importance", ascending=False)

# Print just the top N for readability (but we save ALL)
TOP_N = 32
print(f"\nTop {TOP_N} features by RF importance (of {comp_all_out.shape[0]} total):")
print(comp_all_out.head(TOP_N))

# 6) Save results
print("\n=== Saving Results ===")
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

comp_all_path = output_dir / f"rf_mi_fclassif_compare_ALL_{PARTICIPANT}_{SESSION_TYPE}.csv"
comp_all_out.to_csv(comp_all_path, index=True)
print(f"\nSaved full comparison table (ALL features) to: {comp_all_path}")

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
