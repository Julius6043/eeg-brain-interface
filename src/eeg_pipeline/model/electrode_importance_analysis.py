# --- Electrode Importance Analysis: Leave-One-Out per Electrode ---
import numpy as np, pandas as pd, mne, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Konfiguration
PARTICIPANT = "julian"  # Wähle einen Teilnehmer
SESSION_TYPE = "outdoor"  # Session type
ALL_ELECTRODES = ["EEG1", "EEG2", "EEG3", "EEG4", "EEG5", "EEG6", "EEG7", "EEG8"]

def load_single_participant_session(participant, session_type):
    """Load epochs for a single participant and session"""
    processed_dir = Path("results/processed")
    epo_file = processed_dir / participant / f"{session_type}_processed-epo.fif"

    if not epo_file.exists():
        raise FileNotFoundError(f"Epoched file not found: {epo_file}")

    print(f"Loading data from: {epo_file}")
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)

    difficulty_mapping = {
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

def extract_features(epochs_data, exclude_channels=None):
    """Extract bandpower features from epochs, optionally excluding specific channels"""
    
    # Frequency bands for feature extraction
    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 40),
    }

    # Filter data
    ep_filt = epochs_data.copy().filter(4.0, 40.0, picks="eeg")
    
    # Drop excluded channels if any
    if exclude_channels:
        channels_to_drop = [ch for ch in exclude_channels if ch in ep_filt.ch_names]
        if channels_to_drop:
            ep_filt.drop_channels(channels_to_drop)
            print(f"Excluded channels: {channels_to_drop}")

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

def train_and_evaluate_rf(X, y, cv_folds=5):
    """Train Random Forest and return cross-validation scores"""
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection (select top features if we have many)
    if X_scaled.shape[1] > 20:
        selector = SelectKBest(score_func=f_classif, k=20)
        X_selected = selector.fit_transform(X_scaled, y)
    else:
        X_selected = X_scaled

    # Check if we have enough samples for cross-validation
    min_class_count = np.bincount(y).min()
    n_splits = min(cv_folds, min_class_count) if min_class_count > 1 else 2

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=1000,  # Reduced for speed
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
    
    return scores.mean(), scores.std(), scores

def main():
    print("=== Electrode Importance Analysis ===")
    print(f"Participant: {PARTICIPANT}")
    print(f"Session: {SESSION_TYPE}")
    print(f"Testing electrodes: {ALL_ELECTRODES}")
    
    # Load data
    print("\n1. Loading data...")
    epochs = load_single_participant_session(PARTICIPANT, SESSION_TYPE)
    y = epochs.metadata["difficulty"].astype(int).to_numpy()
    
    results = []
    
    # Baseline: All electrodes
    print("\n2. Baseline performance (all electrodes)...")
    X_all, col_names_all, ep_filt_all = extract_features(epochs)
    baseline_mean, baseline_std, baseline_scores = train_and_evaluate_rf(X_all, y)
    
    print(f"Baseline accuracy: {baseline_mean:.3f} ± {baseline_std:.3f}")
    print(f"Channels used: {ep_filt_all.ch_names}")
    print(f"Total features: {X_all.shape[1]}")
    
    results.append({
        'condition': 'All_electrodes',
        'excluded_electrode': 'None',
        'channels_used': ep_filt_all.ch_names.copy(),
        'n_channels': len(ep_filt_all.ch_names),
        'n_features': X_all.shape[1],
        'accuracy_mean': baseline_mean,
        'accuracy_std': baseline_std,
        'accuracy_drop': 0.0,
        'cv_scores': baseline_scores
    })
    
    # Leave-one-out analysis
    print(f"\n3. Leave-one-out analysis...")
    
    for i, electrode in enumerate(ALL_ELECTRODES):
        print(f"\n--- Excluding {electrode} ({i+1}/{len(ALL_ELECTRODES)}) ---")
        
        try:
            # Extract features without this electrode
            X_excl, col_names_excl, ep_filt_excl = extract_features(epochs, exclude_channels=[electrode])
            
            if X_excl.shape[1] == 0:
                print(f"No features left after excluding {electrode}, skipping...")
                continue
                
            # Train and evaluate
            acc_mean, acc_std, cv_scores = train_and_evaluate_rf(X_excl, y)
            accuracy_drop = baseline_mean - acc_mean
            
            print(f"Accuracy without {electrode}: {acc_mean:.3f} ± {acc_std:.3f}")
            print(f"Accuracy drop: {accuracy_drop:.3f}")
            print(f"Channels used: {ep_filt_excl.ch_names}")
            print(f"Features: {X_excl.shape[1]}")
            
            results.append({
                'condition': f'Without_{electrode}',
                'excluded_electrode': electrode,
                'channels_used': ep_filt_excl.ch_names.copy(),
                'n_channels': len(ep_filt_excl.ch_names),
                'n_features': X_excl.shape[1],
                'accuracy_mean': acc_mean,
                'accuracy_std': acc_std,
                'accuracy_drop': accuracy_drop,
                'cv_scores': cv_scores
            })
            
        except Exception as e:
            print(f"Error processing {electrode}: {e}")
            continue
    
    # Create results DataFrame
    print("\n4. Results Summary...")
    results_df = pd.DataFrame(results)
    
    # Sort by accuracy drop (descending)
    results_df_sorted = results_df.sort_values('accuracy_drop', ascending=False)
    
    print("\n=== ELECTRODE IMPORTANCE RANKING ===")
    print("(Higher accuracy drop = more important electrode)")
    print()
    print(results_df_sorted[['excluded_electrode', 'n_channels', 'accuracy_mean', 'accuracy_drop']].to_string(index=False))
    
    # Identify most and least important electrodes
    baseline_row = results_df[results_df['excluded_electrode'] == 'None'].iloc[0]
    loo_results = results_df[results_df['excluded_electrode'] != 'None'].copy()
    
    if not loo_results.empty:
        most_important = loo_results.loc[loo_results['accuracy_drop'].idxmax()]
        least_important = loo_results.loc[loo_results['accuracy_drop'].idxmin()]
        
        print(f"\n=== KEY FINDINGS ===")
        print(f"Baseline accuracy (all electrodes): {baseline_row['accuracy_mean']:.3f} ± {baseline_row['accuracy_std']:.3f}")
        print(f"Most important electrode: {most_important['excluded_electrode']} (drop: {most_important['accuracy_drop']:.3f})")
        print(f"Least important electrode: {least_important['excluded_electrode']} (drop: {least_important['accuracy_drop']:.3f})")
        
        # Statistical significance check
        significant_drops = loo_results[loo_results['accuracy_drop'] > 2 * baseline_row['accuracy_std']]
        if not significant_drops.empty:
            print(f"\nElectrodes with significant accuracy drops (>2σ):")
            for _, row in significant_drops.iterrows():
                print(f"  - {row['excluded_electrode']}: {row['accuracy_drop']:.3f}")
        else:
            print(f"\nNo electrodes show significant accuracy drops (all drops < {2 * baseline_row['accuracy_std']:.3f})")
    
    # Save results
    print("\n5. Saving results...")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_path = output_dir / f"electrode_importance_{PARTICIPANT}_{SESSION_TYPE}.csv"
    results_df_sorted.to_csv(results_path, index=False)
    print(f"Detailed results saved to: {results_path}")
    
    # Save summary
    summary_df = results_df_sorted[['excluded_electrode', 'n_channels', 'accuracy_mean', 'accuracy_std', 'accuracy_drop']].copy()
    summary_path = output_dir / f"electrode_importance_summary_{PARTICIPANT}_{SESSION_TYPE}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    print(f"\n=== Analysis Complete ===")
    return results_df_sorted

if __name__ == "__main__":
    results = main()
