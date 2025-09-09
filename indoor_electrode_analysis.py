# --- Session-Specific Electrode Importance Analysis ---
"""
This script performs electrode importance analysis for specified session type (indoor/outdoor).
Simply change ANALYSIS_SESSION variable at t    # Baseline performance (all electrodes)
    print("→ Baseline performance (all electrodes)...")
    X_all, _, _ = extract_features(epochs)
    baseline_mean, baseline_std, _ = train_and_evaluate_rf(X_all, y)op to switch between environments.

Strategy:
1. Run electrode importance analysis for EACH participant's specified session type only
2. For each analysis, rank electrodes by accuracy drop when removed
3. Create a voting system across all specified session analyses
4. Generate comprehensive visualizations and summary statistics for the specified environment

This focuses specifically on:
- Single session type electrode importance only (indoor OR outdoor)
- No data mixing between participants or session types
- Session-specific electrode ranking consensus
- Clear identification of consistently important electrodes in specified conditions

USAGE: Change ANALYSIS_SESSION = "indoor" or "outdoor" at the top of the script
"""

import numpy as np, pandas as pd, mne, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import clone
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ========================================
# CONFIGURATION - CHANGE THIS WORD ONLY:
# ========================================
ANALYSIS_SESSION = "indoor"  # Change to "indoor" or "outdoor"

# Auto-generated configuration (DO NOT MODIFY)
# Updated electrode names to match the channel mapping from data_loading.py
# channel_mapping = {1: "Fz", 2: "C4", 3: "Cz", 4: "C3", 5: "Pz", 6: "PO8", 7: "Oz", 8: "PO7"}
ALL_ELECTRODES = ["Fz", "C4", "Cz", "C3", "Pz", "PO8", "Oz", "PO7"]
INCLUDE_0_BACK = False  # Working memory load analysis only
SESSION_TYPES = [ANALYSIS_SESSION]  # Automatically set based on ANALYSIS_SESSION

# Results storage
ALL_RESULTS = []
ELECTRODE_RANKINGS = defaultdict(list)  # electrode -> list of ranks across analyses
ELECTRODE_RANK_SUMS = defaultdict(int)  # electrode -> sum of ranks (higher = worse)
ACCURACY_COMPARISON_RESULTS = []  # 8 vs 4 electrode comparison results

def get_available_participants():
    """Scan for available processed data"""
    processed_dir = Path("results/processed")
    if not processed_dir.exists():
        return []
    
    participants = []
    for participant_dir in processed_dir.iterdir():
        if participant_dir.is_dir() and not participant_dir.name.endswith('.csv'):
            participants.append(participant_dir.name)
    
    return sorted(participants)

def get_participants_with_both_sessions():
    """Get participants who have both indoor and outdoor data"""
    processed_dir = Path("results/processed")
    if not processed_dir.exists():
        return []
    
    participants_with_both = []
    for participant_dir in processed_dir.iterdir():
        if participant_dir.is_dir() and not participant_dir.name.endswith('.csv'):
            participant = participant_dir.name
            indoor_file = participant_dir / "indoor_processed-epo.fif"
            outdoor_file = participant_dir / "outdoor_processed-epo.fif"
            
            if indoor_file.exists() and outdoor_file.exists():
                participants_with_both.append(participant)
    
    print(f"Found {len(participants_with_both)} participants with both indoor and outdoor data: {participants_with_both}")
    return sorted(participants_with_both)

def load_participant_session(participant, session_type):
    """Load epochs for a specific participant and session"""
    processed_dir = Path("results/processed")
    epo_file = processed_dir / participant / f"{session_type}_processed-epo.fif"

    if not epo_file.exists():
        print(f"⚠ Skipping {participant} {session_type}: file not found")
        return None

    try:
        epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
        
        # Configure analysis events based on INCLUDE_0_BACK setting
        if INCLUDE_0_BACK:
            analysis_events = ['0-back', '1-back', '2-back', '3-back']
            event_id_to_difficulty = {
                epochs.event_id['0-back']: 0,
                epochs.event_id['1-back']: 1,
                epochs.event_id['2-back']: 2,
                epochs.event_id['3-back']: 3
            }
        else:
            analysis_events = ['1-back', '2-back', '3-back']
            event_id_to_difficulty = {
                epochs.event_id['1-back']: 1,
                epochs.event_id['2-back']: 2,
                epochs.event_id['3-back']: 3
            }
        
        # Filter epochs
        epochs_filtered = epochs[analysis_events]
        
        if len(epochs_filtered) < 10:  # Minimum epochs threshold
            print(f"⚠ Skipping {participant} {session_type}: too few epochs ({len(epochs_filtered)})")
            return None
        
        # Extract difficulty labels
        difficulties = [event_id_to_difficulty[event_id] for event_id in epochs_filtered.events[:, 2]]
        
        # Create metadata
        metadata = pd.DataFrame({
            'difficulty': difficulties,
            'participant': [participant] * len(epochs_filtered),
            'session_type': [session_type] * len(epochs_filtered)
        })
        epochs_filtered.metadata = metadata
        
        print(f"✓ Loaded {participant} {session_type}: {len(epochs_filtered)} epochs")
        return epochs_filtered
        
    except Exception as e:
        print(f"⚠ Error loading {participant} {session_type}: {e}")
        return None

def extract_features(epochs_data, exclude_channels=None):
    """Extract bandpower features from epochs"""
    bands = {
        "theta": (4, 8),
        "alpha": (8, 13), 
        "beta": (13, 30),
        "gamma": (30, 40),
    }

    ep_filt = epochs_data.copy().filter(4.0, 40.0, picks="eeg")
    
    # Drop excluded channels
    if exclude_channels:
        channels_to_drop = [ch for ch in exclude_channels if ch in ep_filt.ch_names]
        if channels_to_drop:
            ep_filt.drop_channels(channels_to_drop)

    # Compute PSD
    try:
        psd = ep_filt.compute_psd(method="welch", fmin=1.0, fmax=40.0,
                                  n_fft=int(ep_filt.info["sfreq"] * 2),
                                  n_overlap=int(ep_filt.info["sfreq"] * 1),
                                  picks="eeg", verbose=False)
        psds, freqs = psd.get_data(return_freqs=True)
    except Exception:
        from mne.time_frequency import psd_welch
        psds, freqs = psd_welch(ep_filt, fmin=1.0, fmax=40.0,
                               n_fft=int(ep_filt.info["sfreq"] * 2),
                               n_overlap=int(ep_filt.info["sfreq"] * 1),
                               picks="eeg", average="mean", verbose=False)

    # Calculate bandpower features
    bin_mask = {b: (freqs >= lo) & (freqs < hi) for b, (lo, hi) in bands.items()}
    total_pow = psds.sum(axis=2) + 1e-12

    feat_list = []
    col_names = []
    for b, m in bin_mask.items():
        bp = psds[:, :, m].sum(axis=2)
        rel = bp / total_pow
        feat_list.append(rel)
        col_names += [f"{ch}_{b}" for ch in ep_filt.ch_names]

    X = np.concatenate(feat_list, axis=1)
    return X, col_names, ep_filt

def train_and_evaluate_rf(X, y, cv_folds=5):
    """Train Random Forest and return cross-validation scores"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply feature selection if more than 20 features
    if X_scaled.shape[1] > 20:
        selector = SelectKBest(score_func=f_classif, k=20)
        X_selected = selector.fit_transform(X_scaled, y)
    else:
        X_selected = X_scaled

    min_class_count = np.bincount(y).min()
    n_splits = min(cv_folds, min_class_count) if min_class_count > 1 else 2

    rf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                               min_samples_split=4, min_samples_leaf=6,
                               class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(rf, X_selected, y, cv=cv, scoring="accuracy", n_jobs=-1)
    
    return scores.mean(), scores.std(), scores

def train_and_evaluate_rf_with_models(X, y, cv_folds=5):
    """Train Random Forest with cross-validation and return trained models + individual fold scores"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply feature selection if more than 20 features
    if X_scaled.shape[1] > 20:
        selector = SelectKBest(score_func=f_classif, k=20)
        X_selected = selector.fit_transform(X_scaled, y)
    else:
        selector = None
        X_selected = X_scaled

    min_class_count = np.bincount(y).min()
    n_splits = min(cv_folds, min_class_count) if min_class_count > 1 else 2

    rf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                               min_samples_split=4, min_samples_leaf=6,
                               class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    # Store individual fold scores and trained models
    fold_scores = []
    trained_models = []
    
    for train_idx, val_idx in cv.split(X_selected, y):
        X_train_fold, X_val_fold = X_selected[train_idx], X_selected[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model on this fold
        rf_fold = clone(rf)
        rf_fold.fit(X_train_fold, y_train_fold)
        
        # Evaluate on validation set
        score = rf_fold.score(X_val_fold, y_val_fold)
        fold_scores.append(score)
        
        # Store the trained model and preprocessing objects
        trained_models.append({
            'model': rf_fold,
            'scaler': scaler,
            'selector': selector,
            'train_indices': train_idx
        })
    
    fold_scores = np.array(fold_scores)
    return fold_scores.mean(), fold_scores.std(), fold_scores, trained_models, scaler, selector

def test_models_on_outdoor_data(trained_models, outdoor_epochs, exclude_channels=None):
    """Test trained indoor models on outdoor data (no data leakage)"""
    if outdoor_epochs is None:
        return None, None, None
    
    y_outdoor = outdoor_epochs.metadata["difficulty"].astype(int).to_numpy()
    X_outdoor, _, _ = extract_features(outdoor_epochs, exclude_channels=exclude_channels)
    
    outdoor_scores = []
    
    for model_info in trained_models:
        scaler = model_info['scaler']
        selector = model_info['selector']
        model = model_info['model']
        
        # Apply same preprocessing as training
        X_outdoor_scaled = scaler.transform(X_outdoor)
        
        if selector is not None:
            X_outdoor_selected = selector.transform(X_outdoor_scaled)
        else:
            X_outdoor_selected = X_outdoor_scaled
        
        # Test model on outdoor data
        score = model.score(X_outdoor_selected, y_outdoor)
        outdoor_scores.append(score)
    
    outdoor_scores = np.array(outdoor_scores)
    return outdoor_scores.mean(), outdoor_scores.std(), outdoor_scores

def compare_8vs4_electrodes_with_ttest_and_outdoor_testing(participant, best_4_electrodes):
    """Enhanced comparison with proper cross-validation, t-test, and outdoor testing"""
    print(f"\n🔬 ENHANCED 8 vs 4 Electrode Analysis: {participant}")
    print("="*70)
    
    # Load indoor data for training
    indoor_epochs = load_participant_session(participant, "indoor")
    if indoor_epochs is None:
        print(f"⚠ No indoor data for {participant}")
        return None
    
    # Load outdoor data for testing
    outdoor_epochs = load_participant_session(participant, "outdoor")
    outdoor_available = outdoor_epochs is not None
    
    print(f"🏆 Best 4 electrodes: {best_4_electrodes}")
    print(f"📊 Outdoor data available: {'Yes' if outdoor_available else 'No'}")
    
    y_indoor = indoor_epochs.metadata["difficulty"].astype(int).to_numpy()
    
    # === PHASE 1: INDOOR CROSS-VALIDATION TRAINING ===
    print(f"\n📈 PHASE 1: Indoor Cross-Validation Training")
    print("-" * 50)
    
    # Train with all 8 electrodes on indoor data
    print("→ Training with all 8 electrodes (indoor CV)...")
    X_8_electrodes, _, _ = extract_features(indoor_epochs)
    acc_8_mean, acc_8_std, acc_8_scores, models_8, scaler_8, selector_8 = train_and_evaluate_rf_with_models(X_8_electrodes, y_indoor)
    
    # Train with best 4 electrodes on indoor data
    print(f"→ Training with best 4 electrodes (indoor CV)...")
    worst_4_electrodes = [el for el in ALL_ELECTRODES if el not in best_4_electrodes]
    X_4_electrodes, _, _ = extract_features(indoor_epochs, exclude_channels=worst_4_electrodes)
    acc_4_mean, acc_4_std, acc_4_scores, models_4, scaler_4, selector_4 = train_and_evaluate_rf_with_models(X_4_electrodes, y_indoor)
    
    print(f"  8 electrodes (indoor CV): {acc_8_mean:.3f} ± {acc_8_std:.3f}")
    print(f"  4 electrodes (indoor CV): {acc_4_mean:.3f} ± {acc_4_std:.3f}")
    print(f"  Individual 8-electrode scores: {[f'{s:.3f}' for s in acc_8_scores]}")
    print(f"  Individual 4-electrode scores: {[f'{s:.3f}' for s in acc_4_scores]}")
    
    # === PHASE 2: STATISTICAL COMPARISON (T-TEST) ===
    print(f"\n📊 PHASE 2: Statistical Comparison (Paired T-Test)")
    print("-" * 50)
    
    # Perform paired t-test between 8-electrode and 4-electrode CV scores
    t_stat, p_value = stats.ttest_rel(acc_8_scores, acc_4_scores)
    
    print(f"  Paired t-test results:")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significance level: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    if p_value < 0.05:
        better_config = "8 electrodes" if acc_8_mean > acc_4_mean else "4 electrodes"
        print(f"  🎯 Significant difference detected: {better_config} performs better (p < 0.05)")
    else:
        print(f"  🤝 No significant difference between configurations (p ≥ 0.05)")
    
    # === PHASE 3: OUTDOOR TESTING ===
    outdoor_8_mean = outdoor_4_mean = None
    outdoor_8_std = outdoor_4_std = None
    outdoor_8_scores = outdoor_4_scores = None
    
    if outdoor_available:
        print(f"\n🌳 PHASE 3: Outdoor Testing (No Data Leakage)")
        print("-" * 50)
        
        # Test 8-electrode models on outdoor data
        print("→ Testing 8-electrode models on outdoor data...")
        outdoor_8_mean, outdoor_8_std, outdoor_8_scores = test_models_on_outdoor_data(
            models_8, outdoor_epochs, exclude_channels=None
        )
        
        # Test 4-electrode models on outdoor data  
        print("→ Testing 4-electrode models on outdoor data...")
        outdoor_4_mean, outdoor_4_std, outdoor_4_scores = test_models_on_outdoor_data(
            models_4, outdoor_epochs, exclude_channels=worst_4_electrodes
        )
        
        print(f"  8 electrodes (outdoor test): {outdoor_8_mean:.3f} ± {outdoor_8_std:.3f}")
        print(f"  4 electrodes (outdoor test): {outdoor_4_mean:.3f} ± {outdoor_4_std:.3f}")
        print(f"  Individual 8-electrode outdoor scores: {[f'{s:.3f}' for s in outdoor_8_scores]}")
        print(f"  Individual 4-electrode outdoor scores: {[f'{s:.3f}' for s in outdoor_4_scores]}")
        
        # Performance drop from indoor to outdoor
        indoor_to_outdoor_drop_8 = acc_8_mean - outdoor_8_mean
        indoor_to_outdoor_drop_4 = acc_4_mean - outdoor_4_mean
        
        print(f"\n📉 Generalization Analysis:")
        print(f"  8-electrode indoor→outdoor drop: {indoor_to_outdoor_drop_8:.3f}")
        print(f"  4-electrode indoor→outdoor drop: {indoor_to_outdoor_drop_4:.3f}")
        
        generalization_ratio_8 = outdoor_8_mean / acc_8_mean if acc_8_mean > 0 else 0
        generalization_ratio_4 = outdoor_4_mean / acc_4_mean if acc_4_mean > 0 else 0
        
        print(f"  8-electrode generalization: {generalization_ratio_8*100:.1f}% of indoor performance")
        print(f"  4-electrode generalization: {generalization_ratio_4*100:.1f}% of indoor performance")
    else:
        print(f"\n🌳 PHASE 3: Outdoor Testing - Skipped (no outdoor data)")
    
    # Return comprehensive results
    results = {
        'participant': participant,
        'best_4_electrodes': ', '.join(best_4_electrodes),
        
        # Indoor cross-validation results
        'indoor_cv_8_mean': acc_8_mean,
        'indoor_cv_8_std': acc_8_std, 
        'indoor_cv_8_scores': acc_8_scores,
        'indoor_cv_4_mean': acc_4_mean,
        'indoor_cv_4_std': acc_4_std,
        'indoor_cv_4_scores': acc_4_scores,
        
        # Statistical test results
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_difference': p_value < 0.05,
        
        # Outdoor test results (if available)
        'outdoor_test_8_mean': outdoor_8_mean,
        'outdoor_test_8_std': outdoor_8_std,
        'outdoor_test_8_scores': outdoor_8_scores,
        'outdoor_test_4_mean': outdoor_4_mean, 
        'outdoor_test_4_std': outdoor_4_std,
        'outdoor_test_4_scores': outdoor_4_scores,
        'outdoor_available': outdoor_available,
        
        # Performance analysis
        'indoor_difference_8_minus_4': acc_8_mean - acc_4_mean,
        'outdoor_difference_8_minus_4': (outdoor_8_mean - outdoor_4_mean) if outdoor_available else None,
        'generalization_ratio_8': (outdoor_8_mean / acc_8_mean) if outdoor_available and acc_8_mean > 0 else None,
        'generalization_ratio_4': (outdoor_4_mean / acc_4_mean) if outdoor_available and acc_4_mean > 0 else None
    }
    
    return results

def analyze_single_participant_session(participant, session_type):
    """Run electrode importance analysis for one participant-session combination"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {participant.upper()} - {session_type.upper()}")
    print(f"{'='*60}")
    
    # Load data
    epochs = load_participant_session(participant, session_type)
    if epochs is None:
        return None
    
    y = epochs.metadata["difficulty"].astype(int).to_numpy()
    results = []
    
    # Baseline performance (all electrodes) - FIT feature selector here
    print("→ Baseline performance (all electrodes)...")
    X_all, _, ep_filt_all = extract_features(epochs)
    baseline_mean, baseline_std, _ = train_and_evaluate_rf(X_all, y)
    
    results.append({
        'participant': participant,
        'session_type': session_type,
        'condition': 'baseline',
        'excluded_electrode': 'None',
        'accuracy_mean': baseline_mean,
        'accuracy_std': baseline_std,
        'accuracy_drop': 0.0
    })
    
    print(f"  Baseline: {baseline_mean:.3f} ± {baseline_std:.3f}")
    
    # Leave-one-out analysis
    print("→ Leave-one-out electrode analysis...")
    electrode_drops = {}
    
    for electrode in ALL_ELECTRODES:
        try:
            X_excl, _, _ = extract_features(epochs, exclude_channels=[electrode])
            if X_excl.shape[1] == 0:
                continue
                
            # Train and evaluate without this electrode
            acc_mean, acc_std, _ = train_and_evaluate_rf(X_excl, y)
            accuracy_drop = baseline_mean - acc_mean
            electrode_drops[electrode] = accuracy_drop
            
            results.append({
                'participant': participant,
                'session_type': session_type,
                'condition': 'leave_one_out',
                'excluded_electrode': electrode,
                'accuracy_mean': acc_mean,
                'accuracy_std': acc_std,
                'accuracy_drop': accuracy_drop
            })
            
            print(f"  {electrode}: drop = {accuracy_drop:.3f}")
            
        except Exception as e:
            print(f"  {electrode}: ERROR - {e}")
            continue
    
    # Rank electrodes by importance (accuracy drop)
    if electrode_drops:
        sorted_electrodes = sorted(electrode_drops.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 ELECTRODE RANKING for {participant} {session_type}:")
        for rank, (electrode, drop) in enumerate(sorted_electrodes, 1):
            print(f"  {rank}. {electrode}: {drop:.3f}")
            ELECTRODE_RANKINGS[electrode].append(rank)
            ELECTRODE_RANK_SUMS[electrode] += rank  # Add rank to sum (higher sum = worse electrode)
        
        # Most important electrode
        most_important = sorted_electrodes[0][0]
        print(f"🏆 Most important: {most_important}")
    
    return results

def compare_8vs4_electrodes(participant, session_type, best_4_electrodes):
    """Compare accuracy using all 8 electrodes vs only the best 4 electrodes"""
    print(f"\n🔬 8 vs 4 Electrode Comparison: {participant} {session_type}")
    
    # Load data for this participant-session
    epochs = load_participant_session(participant, session_type)
    if epochs is None:
        return None
    
    # Extract labels - use same labeling scheme as main analysis
    y = epochs.metadata["difficulty"].astype(int).to_numpy()
    
    # Test with all 8 electrodes
    print("→ Testing with all 8 electrodes...")
    X_8_electrodes, _, _ = extract_features(epochs)
    acc_8_mean, acc_8_std, _ = train_and_evaluate_rf(X_8_electrodes, y)
    
    # Test with best 4 electrodes only
    print(f"→ Testing with best 4 electrodes: {best_4_electrodes}")
    worst_4_electrodes = [el for el in ALL_ELECTRODES if el not in best_4_electrodes]
    X_4_electrodes, _, _ = extract_features(epochs, exclude_channels=worst_4_electrodes)
    acc_4_mean, acc_4_std, _ = train_and_evaluate_rf(X_4_electrodes, y)
    
    # Calculate difference
    accuracy_difference = acc_8_mean - acc_4_mean
    
    comparison_result = {
        'participant': participant,
        'session_type': session_type,
        'accuracy_8_electrodes': acc_8_mean,
        'accuracy_8_std': acc_8_std,
        'accuracy_4_electrodes': acc_4_mean,
        'accuracy_4_std': acc_4_std,
        'accuracy_difference': accuracy_difference,
        'best_4_electrodes': ', '.join(best_4_electrodes),
        'performance_ratio': acc_4_mean / acc_8_mean if acc_8_mean > 0 else 0
    }
    
    print(f"  8 electrodes: {acc_8_mean:.3f} ± {acc_8_std:.3f}")
    print(f"  4 electrodes: {acc_4_mean:.3f} ± {acc_4_std:.3f}")
    print(f"  Difference: {accuracy_difference:.3f} (8-electrode advantage)")
    print(f"  4-electrode efficiency: {acc_4_mean/acc_8_mean*100:.1f}% of 8-electrode performance")
    
    return comparison_result

def create_electrode_comparison_visualization(comparison_results, session_type="indoor"):
    """Create visualization comparing 8 vs 4 electrode performance"""
    
    if not comparison_results:
        print("❌ No comparison results to visualize!")
        return None
    
    df = pd.DataFrame(comparison_results)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    np.random.seed(RANDOM_SEED)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    session_title = session_type.upper()
    fig.suptitle(f'8 vs 4 Electrode Performance Comparison\n{session_title} SESSIONS', 
                 fontsize=20, fontweight='bold', color='darkblue')
    
    # 1. Individual participant comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    x_pos = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, df['accuracy_8_electrodes'], width, 
                    label='8 Electrodes', color='steelblue', alpha=0.8,
                    yerr=df['accuracy_8_std'], capsize=5)
    bars2 = ax1.bar(x_pos + width/2, df['accuracy_4_electrodes'], width,
                    label='4 Best Electrodes', color='lightcoral', alpha=0.8,
                    yerr=df['accuracy_4_std'], capsize=5)
    
    ax1.set_xlabel('Participants', fontweight='bold')
    ax1.set_ylabel('Classification Accuracy', fontweight='bold')
    ax1.set_title(f'{session_title}: Individual Participant Comparison', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([p.replace('sub-', '') for p in df['participant']], rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Performance ratio analysis
    ax2 = fig.add_subplot(gs[0, 1])
    
    ratios = df['performance_ratio'] * 100  # Convert to percentage
    bars = ax2.bar(range(len(df)), ratios, color='darkgreen', alpha=0.7)
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
    ax2.axhline(y=ratios.mean(), color='orange', linestyle='-', alpha=0.8, 
                label=f'Average: {ratios.mean():.1f}%')
    
    ax2.set_xlabel('Participants', fontweight='bold')
    ax2.set_ylabel('4-Electrode Performance\n(% of 8-Electrode)', fontweight='bold')
    ax2.set_title(f'{session_title}: Efficiency of 4 Best Electrodes', fontweight='bold')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([p.replace('sub-', '') for p in df['participant']], rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        ax2.text(bar.get_x() + bar.get_width()/2., ratio + 1,
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=12)
    
    # 3. Statistical summary
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Box plot comparison
    data_8 = df['accuracy_8_electrodes'].values
    data_4 = df['accuracy_4_electrodes'].values
    
    bp = ax3.boxplot([data_8, data_4], labels=['8 Electrodes', '4 Best Electrodes'], 
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('lightcoral')
    bp['boxes'][1].set_alpha(0.7)
    
    ax3.set_ylabel('Classification Accuracy', fontweight='bold')
    ax3.set_title(f'{session_title}: Statistical Distribution Comparison', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add mean lines
    ax3.axhline(y=data_8.mean(), color='steelblue', linestyle='--', alpha=0.7)
    ax3.axhline(y=data_4.mean(), color='lightcoral', linestyle='--', alpha=0.7)
    
    # 4. Difference analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    differences = df['accuracy_difference'].values
    colors = ['green' if d >= 0 else 'red' for d in differences]
    bars = ax4.bar(range(len(df)), differences, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax4.axhline(y=differences.mean(), color='purple', linestyle='--', alpha=0.8,
                label=f'Average: {differences.mean():.3f}')
    
    ax4.set_xlabel('Participants', fontweight='bold')
    ax4.set_ylabel('Accuracy Difference\n(8-electrode - 4-electrode)', fontweight='bold')
    ax4.set_title(f'{session_title}: Performance Difference Analysis', fontweight='bold')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels([p.replace('sub-', '') for p in df['participant']], rotation=45)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        y_pos = diff + 0.002 if diff >= 0 else diff - 0.005
        ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{diff:.3f}', ha='center', va='bottom' if diff >= 0 else 'top', fontsize=12)
    
    # Save the comparison plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"electrode_comparison_8vs4_{session_type}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 8 vs 4 electrode comparison plot saved to: {plot_path}")
    
    plt.show()
    return fig

def create_session_visualization(all_results_df, session_type="indoor"):
    """Create visualization for session analysis results"""
    
    if all_results_df is None or all_results_df.empty:
        print(f"❌ No {session_type} analysis results to visualize!")
        return None
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    np.random.seed(RANDOM_SEED)
    
    # Create figure with electrode importance visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter data to only leave-one-out results (not baseline)
    loo_data = all_results_df[all_results_df['condition'] == 'leave_one_out'].copy()
    
    if loo_data.empty:
        ax.text(0.5, 0.5, f'No leave-one-out data\navailable for {session_type}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=18, color='gray')
        ax.set_title(f'{session_type.title()} Session - Electrode Performance Analysis', 
                     fontsize=18, fontweight='bold')
        return fig
    
    # Group by excluded electrode and calculate mean accuracy drop (higher drop = more important)
    electrode_drops = loo_data.groupby('excluded_electrode')['accuracy_drop'].mean().sort_values(ascending=False)
    
    # Create bar plot
    bars = ax.bar(range(len(electrode_drops)), electrode_drops.values, 
                  color=plt.cm.viridis(np.linspace(0, 1, len(electrode_drops))))
    
    # Customize the plot
    ax.set_xlabel('Electrode', fontsize=16, fontweight='bold')
    ax.set_ylabel('Mean Accuracy Drop\n(Higher = More Important)', fontsize=16, fontweight='bold')
    ax.set_title(f'{session_type.title()} Session - Electrode Importance Analysis\n(Accuracy Drop When Removed)', 
                 fontsize=18, fontweight='bold')
    ax.set_xticks(range(len(electrode_drops)))
    ax.set_xticklabels(electrode_drops.index, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (electrode, drop) in enumerate(electrode_drops.items()):
        ax.text(i, drop + 0.001, f'{drop:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"{session_type}_electrode_analysis_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 {session_type.title()} session visualization saved to: {plot_path}")
    
    plt.show()
    
    return plot_path

def create_rank_sum_visualization(rank_sum_data, session_type="indoor"):
    """Create visualization for electrode rank sum analysis"""
    
    if not rank_sum_data:
        print(f"❌ No rank sum data to visualize for {session_type}!")
        return None
    
    # Sort by rank sum (lower is better)
    rank_sum_sorted = sorted(rank_sum_data.items(), key=lambda x: x[1])
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    np.random.seed(RANDOM_SEED)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    electrodes = [item[0] for item in rank_sum_sorted]
    rank_sums = [item[1] for item in rank_sum_sorted]
    
    # Create color gradient (green for best, red for worst)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(electrodes)))
    
    # Create bar plot
    bars = ax.bar(range(len(electrodes)), rank_sums, color=colors, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Electrode', fontsize=16, fontweight='bold')
    ax.set_ylabel('Rank Sum\n(Lower = Better Performance)', fontsize=16, fontweight='bold')
    ax.set_title(f'{session_type.title()} Session - Electrode Ranking Summary\n(Based on Cumulative Rankings Across All Participants)', 
                 fontsize=18, fontweight='bold')
    ax.set_xticks(range(len(electrodes)))
    ax.set_xticklabels(electrodes, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (electrode, rank_sum) in enumerate(zip(electrodes, rank_sums)):
        ax.text(i, rank_sum + max(rank_sums) * 0.01, f'{rank_sum}', 
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Add rank position labels
    for i, (electrode, rank_sum) in enumerate(zip(electrodes, rank_sums)):
        position = i + 1
        ax.text(i, rank_sum * 0.5, f'#{position}', 
                ha='center', va='center', fontweight='bold', 
                fontsize=16, color='white' if rank_sum > max(rank_sums) * 0.7 else 'black')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"{session_type}_electrode_rank_sum_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 {session_type.title()} rank sum visualization saved to: {plot_path}")
    
    plt.show()
    
    return plot_path

def create_enhanced_comparison_visualization(enhanced_results):
    """Create visualization for the enhanced comparison with t-tests and outdoor testing"""
    
    if not enhanced_results:
        print("❌ No enhanced comparison results to visualize!")
        return None
    
    df = pd.DataFrame(enhanced_results)
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("Set2")
    np.random.seed(RANDOM_SEED)
    
    # Create figure with 2x3 layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Enhanced 8 vs 4 Electrode Analysis\nIndoor Training → Statistical Testing → Outdoor Generalization', 
                 fontsize=20, fontweight='bold', color='darkblue')
    
    # 1. Indoor Cross-Validation Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    participants = [p.replace('sub-', '') for p in df['participant']]
    x_pos = np.arange(len(participants))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, df['indoor_cv_8_mean'], width,
                    label='8 Electrodes', color='steelblue', alpha=0.8,
                    yerr=df['indoor_cv_8_std'], capsize=3)
    bars2 = ax1.bar(x_pos + width/2, df['indoor_cv_4_mean'], width,
                    label='4 Best Electrodes', color='lightcoral', alpha=0.8,
                    yerr=df['indoor_cv_4_std'], capsize=3)
    
    ax1.set_xlabel('Participants', fontweight='bold')
    ax1.set_ylabel('Indoor CV Accuracy', fontweight='bold')
    ax1.set_title('Indoor Cross-Validation Training', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(participants, rotation=45)
    ax1.legend(loc='lower left')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Statistical Significance (P-values)
    ax2 = fig.add_subplot(gs[0, 1])
    
    p_values = df['p_value'].values
    colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
    bars = ax2.bar(range(len(participants)), -np.log10(p_values), color=colors, alpha=0.7)
    
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.8, label='p = 0.05')
    ax2.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.8, label='p = 0.01')
    
    ax2.set_xlabel('Participants', fontweight='bold')
    ax2.set_ylabel('-log10(p-value)', fontweight='bold')
    ax2.set_title('Statistical Significance (T-Test)', fontweight='bold')
    ax2.set_xticks(range(len(participants)))
    ax2.set_xticklabels(participants, rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add significance annotations
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        sig_label = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                sig_label, ha='center', va='bottom', fontweight='bold')
    
    # 3. Outdoor Testing Results
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Filter participants with outdoor data
    outdoor_df = df[df['outdoor_available'] == True]
    if len(outdoor_df) > 0:
        outdoor_participants = [p.replace('sub-', '') for p in outdoor_df['participant']]
        x_pos_outdoor = np.arange(len(outdoor_participants))
        
        bars1 = ax3.bar(x_pos_outdoor - width/2, outdoor_df['outdoor_test_8_mean'], width,
                        label='8 Electrodes', color='darkblue', alpha=0.8,
                        yerr=outdoor_df['outdoor_test_8_std'], capsize=3)
        bars2 = ax3.bar(x_pos_outdoor + width/2, outdoor_df['outdoor_test_4_mean'], width,
                        label='4 Best Electrodes', color='darkred', alpha=0.8,
                        yerr=outdoor_df['outdoor_test_4_std'], capsize=3)
        
        ax3.set_xlabel('Participants', fontweight='bold')
        ax3.set_ylabel('Outdoor Test Accuracy', fontweight='bold')
        ax3.set_title('Outdoor Generalization Testing', fontweight='bold')
        ax3.set_xticks(x_pos_outdoor)
        ax3.set_xticklabels(outdoor_participants, rotation=45)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Outdoor Data\nAvailable', ha='center', va='center',
                transform=ax3.transAxes, fontsize=18, color='gray')
        ax3.set_title('Outdoor Generalization Testing', fontweight='bold')
    
    # 4. Generalization Analysis (Indoor vs Outdoor)
    ax4 = fig.add_subplot(gs[1, 0])
    
    if len(outdoor_df) > 0:
        # Plot indoor vs outdoor for both configurations
        for i, (_, row) in enumerate(outdoor_df.iterrows()):
            ax4.plot([0, 1], [row['indoor_cv_8_mean'], row['outdoor_test_8_mean']], 
                    'o-', color='steelblue', alpha=0.7, linewidth=2)
            ax4.plot([2, 3], [row['indoor_cv_4_mean'], row['outdoor_test_4_mean']], 
                    'o-', color='lightcoral', alpha=0.7, linewidth=2)
        
        ax4.set_xticks([0, 1, 2, 3])
        ax4.set_xticklabels(['Indoor\n8-Elec', 'Outdoor\n8-Elec', 'Indoor\n4-Elec', 'Outdoor\n4-Elec'])
        ax4.set_ylabel('Accuracy', fontweight='bold')
        ax4.set_title('Indoor → Outdoor Generalization', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Outdoor Data\nfor Generalization\nAnalysis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=16, color='gray')
        ax4.set_title('Indoor → Outdoor Generalization', fontweight='bold')
    
    # 5. Performance Retention Analysis
    ax5 = fig.add_subplot(gs[1, 1])
    
    if len(outdoor_df) > 0:
        retention_8 = outdoor_df['generalization_ratio_8'] * 100
        retention_4 = outdoor_df['generalization_ratio_4'] * 100
        
        x_pos_ret = np.arange(len(outdoor_participants))
        bars1 = ax5.bar(x_pos_ret - width/2, retention_8, width,
                        label='8 Electrodes', color='steelblue', alpha=0.8)
        bars2 = ax5.bar(x_pos_ret + width/2, retention_4, width,
                        label='4 Best Electrodes', color='lightcoral', alpha=0.8)
        
        ax5.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Perfect Retention')
        ax5.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% Retention')
        
        ax5.set_xlabel('Participants', fontweight='bold')
        ax5.set_ylabel('Performance Retention (%)', fontweight='bold')
        ax5.set_title('Outdoor Performance Retention', fontweight='bold')
        ax5.set_xticks(x_pos_ret)
        ax5.set_xticklabels(outdoor_participants, rotation=45)
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No Outdoor Data\nfor Retention\nAnalysis', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=16, color='gray')
        ax5.set_title('Outdoor Performance Retention', fontweight='bold')
    
    # 6. Summary Statistics Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Calculate summary statistics
    indoor_8_mean = df['indoor_cv_8_mean'].mean()
    indoor_4_mean = df['indoor_cv_4_mean'].mean()
    sig_count = sum(df['significant_difference'])
    
    summary_text = f"""
    SUMMARY STATISTICS
    
    Indoor Cross-Validation:
    • 8 electrodes: {indoor_8_mean:.3f} ± {df['indoor_cv_8_std'].mean():.3f}
    • 4 electrodes: {indoor_4_mean:.3f} ± {df['indoor_cv_4_std'].mean():.3f}
    • Significant differences: {sig_count}/{len(df)}
    
    """
    
    if len(outdoor_df) > 0:
        outdoor_8_mean = outdoor_df['outdoor_test_8_mean'].mean()
        outdoor_4_mean = outdoor_df['outdoor_test_4_mean'].mean()
        avg_retention_8 = outdoor_df['generalization_ratio_8'].mean() * 100
        avg_retention_4 = outdoor_df['generalization_ratio_4'].mean() * 100
        
        summary_text += f"""Outdoor Testing:
    • 8 electrodes: {outdoor_8_mean:.3f} ± {outdoor_df['outdoor_test_8_std'].mean():.3f}
    • 4 electrodes: {outdoor_4_mean:.3f} ± {outdoor_df['outdoor_test_4_std'].mean():.3f}
    • 8-elec retention: {avg_retention_8:.1f}%
    • 4-elec retention: {avg_retention_4:.1f}%
    • Participants tested: {len(outdoor_df)}
    """
    else:
        summary_text += "Outdoor Testing: No data available"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=14,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Save plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "enhanced_electrode_comparison_indoor_training_outdoor_testing.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Enhanced comparison plot saved to: {plot_path}")
    
    plt.show()
    return fig

def main():
    """Run electrode importance analysis for specified session types"""
    # Clear global variables to prevent data accumulation from previous runs
    global ALL_RESULTS, ELECTRODE_RANKINGS, ELECTRODE_RANK_SUMS, ACCURACY_COMPARISON_RESULTS
    ALL_RESULTS.clear()
    ELECTRODE_RANKINGS.clear()
    ELECTRODE_RANK_SUMS.clear()
    ACCURACY_COMPARISON_RESULTS.clear()
    
    session_name = SESSION_TYPES[0] if len(SESSION_TYPES) == 1 else "mixed"
    session_title = session_name.upper()
    
    print(f"🔬 {session_title}-ONLY ELECTRODE IMPORTANCE ANALYSIS")
    print("=" * 80)
    print("Strategy: Individual analysis + voting for specified sessions only")
    print(f"Analysis mode: {'Attention + Working Memory' if INCLUDE_0_BACK else 'Working Memory Load Only'}")
    print(f"Session filter: {session_title} ONLY")
    print("=" * 80)
    
    # Get available participants
    participants = get_available_participants()
    print(f"\n📋 Found {len(participants)} participants: {participants}")
    print(f"📋 Session types: {SESSION_TYPES} ({session_title} only)")
    
    # Check which participants have both indoor and outdoor data
    participants_with_both = get_participants_with_both_sessions()
    participants_indoor_only = []
    participants_outdoor_only = []
    
    processed_dir = Path("results/processed")
    for participant in participants:
        participant_dir = processed_dir / participant
        has_indoor = (participant_dir / "indoor_processed-epo.fif").exists()
        has_outdoor = (participant_dir / "outdoor_processed-epo.fif").exists()
        
        if has_indoor and not has_outdoor:
            participants_indoor_only.append(participant)
        elif has_outdoor and not has_indoor:
            participants_outdoor_only.append(participant)
    
    print(f"\n📊 DATA AVAILABILITY SUMMARY:")
    print(f"  Participants with BOTH indoor & outdoor: {len(participants_with_both)} {participants_with_both}")
    if participants_indoor_only:
        print(f"  Participants with INDOOR only: {len(participants_indoor_only)} {participants_indoor_only}")
    if participants_outdoor_only:
        print(f"  Participants with OUTDOOR only: {len(participants_outdoor_only)} {participants_outdoor_only}")
    
    total_planned = len(participants) * len(SESSION_TYPES)
    print(f"\n📋 Total analyses planned: {total_planned}")
    
    # Run analysis for each participant-session combination
    completed_analyses = 0
    for participant in participants:
        for session_type in SESSION_TYPES:
            results = analyze_single_participant_session(participant, session_type)
            if results:
                ALL_RESULTS.extend(results)
                completed_analyses += 1
    
    if not ALL_RESULTS:
        print("❌ No analyses completed successfully!")
        return
    
    print(f"\n✅ COMPLETED: {completed_analyses}/{total_planned} {session_name} analyses")
    
    # Convert to DataFrame
    all_results_df = pd.DataFrame(ALL_RESULTS)
    
    # Average ranking analysis
    print(f"\n📊 AVERAGE RANKING ANALYSIS ({session_title} Sessions):")
    print("=" * 60)
    avg_rankings = {}
    for electrode, ranks in ELECTRODE_RANKINGS.items():
        avg_rank = np.mean(ranks)
        std_rank = np.std(ranks)
        avg_rankings[electrode] = (avg_rank, std_rank)
    
    rank_sorted = sorted(avg_rankings.items(), key=lambda x: x[1][0])
    for rank, (electrode, (avg_rank, std_rank)) in enumerate(rank_sorted, 1):
        print(f"{rank}. {electrode}: avg rank {avg_rank:.1f} (±{std_rank:.1f})")
    
    # Rank sum analysis (NEW)
    print(f"\n🔢 RANK SUM ANALYSIS ({session_title} Sessions):")
    print("=" * 60)
    print("Lower sum = BETTER electrode (consistently high ranks)")
    print("Higher sum = WORSE electrode (consistently low ranks)")
    rank_sum_sorted = sorted(ELECTRODE_RANK_SUMS.items(), key=lambda x: x[1])
    for position, (electrode, total_rank) in enumerate(rank_sum_sorted, 1):
        avg_rank_for_electrode = total_rank / completed_analyses
        print(f"{position}. {electrode}: sum={total_rank} (avg={avg_rank_for_electrode:.1f})")
    
    # Identify best and worst
    best_electrode = rank_sum_sorted[0] if rank_sum_sorted else ('N/A', 0)
    worst_electrode = rank_sum_sorted[-1] if rank_sum_sorted else ('N/A', 0)
    print(f"\n🥇 BEST electrode (lowest sum): {best_electrode[0]} (sum={best_electrode[1]})")
    print(f"🥉 WORST electrode (highest sum): {worst_electrode[0]} (sum={worst_electrode[1]})")
    
    # Save comprehensive results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Determine session type for file naming
    session_name = SESSION_TYPES[0] if len(SESSION_TYPES) == 1 else "mixed"
    
    # Save detailed results
    detailed_path = output_dir / f"electrode_analysis_{session_name}_only.csv"
    all_results_df.to_csv(detailed_path, index=False)
    print(f"\n💾 Detailed {session_name} results saved to: {detailed_path}")
    
    # Save ranking summary  
    ranking_summary = pd.DataFrame([
        {'electrode': electrode, 'avg_rank': avg_rank, 'std_rank': std_rank, 'consistency_score': 1/std_rank if std_rank > 0 else float('inf')}
        for electrode, (avg_rank, std_rank) in avg_rankings.items()
    ]).sort_values('avg_rank')
    ranking_path = output_dir / f"electrode_ranking_{session_name}_only.csv"
    ranking_summary.to_csv(ranking_path, index=False)
    print(f"💾 {session_name.title()} ranking summary saved to: {ranking_path}")
    
    # Save rank sum summary (NEW)
    rank_sum_summary = pd.DataFrame([
        {
            'electrode': electrode, 
            'rank_sum': rank_sum, 
            'avg_rank': rank_sum / completed_analyses,
            'performance_order': position
        }
        for position, (electrode, rank_sum) in enumerate(rank_sum_sorted, 1)
    ])
    rank_sum_path = output_dir / f"electrode_rank_sums_{session_name}_only.csv"
    rank_sum_summary.to_csv(rank_sum_path, index=False)
    print(f"💾 {session_name.title()} rank sum analysis saved to: {rank_sum_path}")
    
    # Generate comprehensive visualization
    session_name = SESSION_TYPES[0] if len(SESSION_TYPES) == 1 else "mixed"
    print(f"\n🎨 Generating {session_name}-only visualization...")
    create_session_visualization(all_results_df, session_name)
    
    # Generate rank sum visualization
    print(f"\n🎨 Generating {session_name} rank sum visualization...")
    create_rank_sum_visualization(ELECTRODE_RANK_SUMS, session_name)
    
    # ENHANCED 8 vs 4 Electrode Comparison with T-Tests and Outdoor Testing
    print(f"\n🔬 ENHANCED 8 vs 4 ELECTRODE ANALYSIS WITH STATISTICAL TESTING")
    print("=" * 80)
    print("Strategy: Indoor CV training → T-test comparison → Outdoor testing")
    print("No data leakage: Train on indoor, test on outdoor")
    print("=" * 80)
    
    # Identify best 4 electrodes based on average ranking
    best_4_electrodes = [electrode for electrode, _ in rank_sorted[:4]]
    print(f"🏆 Best 4 electrodes identified: {best_4_electrodes}")
    
    # === ENHANCED ANALYSIS: Only participants with both indoor and outdoor data ===
    print(f"\n🔬 ENHANCED ANALYSIS: T-test and Outdoor Testing")
    print("=" * 60)
    print("For valid indoor→outdoor generalization testing, we need participants with BOTH session types")
    
    participants_with_both = get_participants_with_both_sessions()
    
    if not participants_with_both:
        print("⚠ No participants found with both indoor and outdoor data!")
        print("⚠ Skipping enhanced analysis...")
        enhanced_comparison_results = []
        enhanced_completed = 0
    else:
        print(f"✅ Found {len(participants_with_both)} participants suitable for enhanced analysis")
        
        enhanced_comparison_results = []
        enhanced_completed = 0
        
        for participant in participants_with_both:
            result = compare_8vs4_electrodes_with_ttest_and_outdoor_testing(participant, best_4_electrodes)
            if result:
                enhanced_comparison_results.append(result)
                enhanced_completed += 1
        
        print(f"\n✅ COMPLETED: {enhanced_completed}/{len(participants_with_both)} enhanced comparisons")
    
    # Generate enhanced visualization
    if enhanced_comparison_results:
        print(f"\n🎨 Generating enhanced comparison visualization...")
        create_enhanced_comparison_visualization(enhanced_comparison_results)
        
        # Save enhanced results to CSV
        enhanced_df = pd.DataFrame(enhanced_comparison_results)
        enhanced_path = output_dir / f"enhanced_electrode_comparison_{session_name}.csv"
        enhanced_df.to_csv(enhanced_path, index=False)
        print(f"💾 Enhanced comparison results saved to: {enhanced_path}")
        
        # Print comprehensive summary
        print(f"\n📊 ENHANCED COMPARISON SUMMARY:")
        print("=" * 60)
        print(f"📋 Participants included: {len(enhanced_df)} (only those with BOTH indoor & outdoor data)")
        participants_included = [result['participant'] for result in enhanced_comparison_results]
        print(f"📋 Included participants: {participants_included}")
        
        # Indoor CV summary
        print(f"\n🏠 INDOOR CROSS-VALIDATION RESULTS:")
        avg_indoor_8 = enhanced_df['indoor_cv_8_mean'].mean()
        avg_indoor_4 = enhanced_df['indoor_cv_4_mean'].mean()
        print(f"  8 electrodes: {avg_indoor_8:.3f} ± {enhanced_df['indoor_cv_8_std'].mean():.3f}")
        print(f"  4 electrodes: {avg_indoor_4:.3f} ± {enhanced_df['indoor_cv_4_std'].mean():.3f}")
        
        # Statistical significance summary
        sig_count = sum(enhanced_df['significant_difference'])
        print(f"\n📊 STATISTICAL SIGNIFICANCE (T-TESTS):")
        print(f"  Significant differences found: {sig_count}/{len(enhanced_df)} participants")
        print(f"  Average p-value: {enhanced_df['p_value'].mean():.4f}")
        
        if sig_count > 0:
            sig_participants = enhanced_df[enhanced_df['significant_difference'] == True]
            print(f"  Participants with significant differences:")
            for _, row in sig_participants.iterrows():
                better = "8-elec" if row['indoor_difference_8_minus_4'] > 0 else "4-elec"
                print(f"    {row['participant']}: {better} better (p={row['p_value']:.4f})")
        
        # Outdoor testing summary
        outdoor_available = enhanced_df['outdoor_available'].sum()
        if outdoor_available > 0:
            outdoor_df = enhanced_df[enhanced_df['outdoor_available'] == True]
            print(f"\n🌳 OUTDOOR TESTING RESULTS ({outdoor_available} participants):")
            avg_outdoor_8 = outdoor_df['outdoor_test_8_mean'].mean()
            avg_outdoor_4 = outdoor_df['outdoor_test_4_mean'].mean()
            print(f"  8 electrodes: {avg_outdoor_8:.3f} ± {outdoor_df['outdoor_test_8_std'].mean():.3f}")
            print(f"  4 electrodes: {avg_outdoor_4:.3f} ± {outdoor_df['outdoor_test_4_std'].mean():.3f}")
            
            avg_retention_8 = outdoor_df['generalization_ratio_8'].mean() * 100
            avg_retention_4 = outdoor_df['generalization_ratio_4'].mean() * 100
            print(f"  8-electrode retention: {avg_retention_8:.1f}% of indoor performance")
            print(f"  4-electrode retention: {avg_retention_4:.1f}% of indoor performance")
            
            # Generalization comparison
            better_generalizer = "8 electrodes" if avg_retention_8 > avg_retention_4 else "4 electrodes"
            print(f"  Better generalization: {better_generalizer}")
        else:
            print(f"\n🌳 OUTDOOR TESTING: No outdoor data available")
    
    # 8 vs 4 Electrode Comparison Analysis (Keep original for comparison)
    print(f"\n🔬 STARTING ORIGINAL 8 vs 4 ELECTRODE COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Run 8 vs 4 comparison for each participant
    comparison_completed = 0
    for participant in participants:
        for session_type in SESSION_TYPES:
            comparison_result = compare_8vs4_electrodes(participant, session_type, best_4_electrodes)
            if comparison_result:
                ACCURACY_COMPARISON_RESULTS.append(comparison_result)
                comparison_completed += 1
    
    print(f"\n✅ COMPLETED: {comparison_completed}/{total_planned} original 8 vs 4 comparisons")
    
    # Generate 8 vs 4 comparison visualization
    if ACCURACY_COMPARISON_RESULTS:
        print(f"\n🎨 Generating original 8 vs 4 electrode comparison visualization...")
        create_electrode_comparison_visualization(ACCURACY_COMPARISON_RESULTS, session_name)
        
        # Save comparison results to CSV
        comparison_df = pd.DataFrame(ACCURACY_COMPARISON_RESULTS)
        comparison_path = output_dir / f"electrode_comparison_8vs4_{session_name}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"💾 8 vs 4 comparison results saved to: {comparison_path}")
        
        # Print summary statistics
        print(f"\n📊 8 vs 4 ELECTRODE COMPARISON SUMMARY ({session_title}):")
        print("=" * 60)
        avg_8_acc = comparison_df['accuracy_8_electrodes'].mean()
        avg_4_acc = comparison_df['accuracy_4_electrodes'].mean()
        avg_diff = comparison_df['accuracy_difference'].mean()
        avg_ratio = comparison_df['performance_ratio'].mean() * 100
        
        print(f"Average 8-electrode accuracy: {avg_8_acc:.3f}")
        print(f"Average 4-electrode accuracy: {avg_4_acc:.3f}")
        print(f"Average difference (8-4): {avg_diff:.3f}")
        print(f"4-electrode efficiency: {avg_ratio:.1f}% of 8-electrode performance")
        
        # Count participants where 4 electrodes perform nearly as well (within 5%)
        efficient_count = sum(1 for ratio in comparison_df['performance_ratio'] if ratio >= 0.95)
        print(f"Participants with ≥95% efficiency using 4 electrodes: {efficient_count}/{len(comparison_df)}")
        
        # Identify best performing participant with 4 electrodes
        best_4_participant = comparison_df.loc[comparison_df['accuracy_4_electrodes'].idxmax()]
        print(f"Best 4-electrode performance: {best_4_participant['participant']} ({best_4_participant['accuracy_4_electrodes']:.3f})")
    else:
        print("❌ No 8 vs 4 comparison results to analyze!")
    
    # Final recommendations
    print(f"\n🎯 {session_title} SESSION RECOMMENDATIONS:")
    print("=" * 60)
    if rank_sorted:
        # Get top electrode by average ranking
        top_electrode_by_rank = rank_sorted[0]
        print(f"🏆 Best by average ranking ({session_title}): {top_electrode_by_rank[0][0]} (avg rank: {top_electrode_by_rank[1][0]:.1f})")
        
        if len(rank_sorted) >= 3:
            top_3 = [electrode for electrode, _ in rank_sorted[:3]]
            print(f"📈 Top 3 by average ranking ({session_title}): {', '.join(top_3)}")
            
            # Find most consistent (lowest std in ranking)
            most_consistent = min(avg_rankings.items(), key=lambda x: x[1][1])
            print(f"🎯 Most consistent performer ({session_title}): {most_consistent[0]} (rank std: {most_consistent[1][1]:.2f})")
        
        # Add rank sum recommendations
        if rank_sum_sorted:
            print(f"🥇 BEST electrode by rank sum: {best_electrode[0]} (sum={best_electrode[1]})")
            print(f"🥉 WORST electrode by rank sum: {worst_electrode[0]} (sum={worst_electrode[1]})")
    
    print(f"\n✅ {session_title}-ONLY ANALYSIS COMPLETE!")
    print(f"📊 Results: {completed_analyses} {session_name} analyses → ranking → rank sums → visualization")
    print(f"📊 Enhanced Analysis: {len(enhanced_comparison_results)} enhanced comparisons → statistical testing → outdoor validation")
    
    return all_results_df, avg_rankings, enhanced_comparison_results

if __name__ == "__main__":
    results_df, rankings, enhanced_comparison_results = main()
