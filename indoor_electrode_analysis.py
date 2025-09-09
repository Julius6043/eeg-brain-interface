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
ALL_ELECTRODES = ["EEG1", "EEG2", "EEG3", "EEG4", "EEG5", "EEG6", "EEG7", "EEG8"]
INCLUDE_0_BACK = False  # Working memory load analysis only
SESSION_TYPES = [ANALYSIS_SESSION]  # Automatically set based on ANALYSIS_SESSION

# Results storage
ALL_RESULTS = []
ELECTRODE_RANKINGS = defaultdict(list)  # electrode -> list of ranks across analyses
ELECTRODE_RANK_SUMS = defaultdict(int)  # electrode -> sum of ranks (higher = worse)
ACCURACY_COMPARISON_RESULTS = []  # 8 vs 4 electrode comparison results
INDOOR_OUTDOOR_COMPARISON_RESULTS = []  # Indoor vs outdoor comparison results

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
    baseline_mean, baseline_std, _, feature_selector = train_and_evaluate_rf(X_all, y)
    
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

def compare_indoor_vs_outdoor_4electrodes(participant, best_4_electrodes):
    """Compare accuracy of same 4 electrodes between indoor and outdoor environments"""
    print(f"\n🏠🌳 Indoor vs Outdoor Comparison (4 electrodes): {participant}")
    
    # Load both indoor and outdoor data for this participant
    indoor_epochs = load_participant_session(participant, "indoor")
    outdoor_epochs = load_participant_session(participant, "outdoor")
    
    # Check if both sessions are available
    if indoor_epochs is None and outdoor_epochs is None:
        print(f"⚠ No data available for {participant}")
        return None
    elif indoor_epochs is None:
        print(f"⚠ No indoor data for {participant}")
        return None
    elif outdoor_epochs is None:
        print(f"⚠ No outdoor data for {participant}")
        return None
    
    # Prepare feature extraction with only best 4 electrodes
    worst_4_electrodes = [el for el in ALL_ELECTRODES if el not in best_4_electrodes]
    print(f"→ Using electrodes: {best_4_electrodes}")
    
    # Indoor performance
    print("→ Testing indoor performance...")
    y_indoor = indoor_epochs.metadata["difficulty"].astype(int).to_numpy()
    X_indoor, _, _ = extract_features(indoor_epochs, exclude_channels=worst_4_electrodes)
    acc_indoor_mean, acc_indoor_std, _ = train_and_evaluate_rf(X_indoor, y_indoor)
    
    # Outdoor performance
    print("→ Testing outdoor performance...")
    y_outdoor = outdoor_epochs.metadata["difficulty"].astype(int).to_numpy()
    X_outdoor, _, _ = extract_features(outdoor_epochs, exclude_channels=worst_4_electrodes)
    acc_outdoor_mean, acc_outdoor_std, _ = train_and_evaluate_rf(X_outdoor, y_outdoor)
    
    # Calculate difference (indoor - outdoor)
    accuracy_difference = acc_indoor_mean - acc_outdoor_mean
    performance_ratio = acc_outdoor_mean / acc_indoor_mean if acc_indoor_mean > 0 else 0
    
    comparison_result = {
        'participant': participant,
        'best_4_electrodes': ', '.join(best_4_electrodes),
        'accuracy_indoor': acc_indoor_mean,
        'accuracy_indoor_std': acc_indoor_std,
        'accuracy_outdoor': acc_outdoor_mean,
        'accuracy_outdoor_std': acc_outdoor_std,
        'accuracy_difference_indoor_outdoor': accuracy_difference,
        'performance_ratio_outdoor_indoor': performance_ratio,
        'indoor_epochs': len(indoor_epochs),
        'outdoor_epochs': len(outdoor_epochs)
    }
    
    print(f"  🏠 Indoor accuracy: {acc_indoor_mean:.3f} ± {acc_indoor_std:.3f}")
    print(f"  🌳 Outdoor accuracy: {acc_outdoor_mean:.3f} ± {acc_outdoor_std:.3f}")
    print(f"  📊 Difference (I-O): {accuracy_difference:.3f}")
    print(f"  📈 Outdoor/Indoor ratio: {performance_ratio:.3f}")
    
    return comparison_result

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
                 fontsize=16, fontweight='bold', color='darkblue')
    
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
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
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
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=8)
    
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
                f'{diff:.3f}', ha='center', va='bottom' if diff >= 0 else 'top', fontsize=8)
    
    # Save the comparison plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"electrode_comparison_8vs4_{session_type}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 8 vs 4 electrode comparison plot saved to: {plot_path}")
    
    plt.show()
    return fig

def create_indoor_outdoor_comparison_visualization(comparison_results):
    """Create visualization comparing indoor vs outdoor performance with same 4 electrodes"""
    
    if not comparison_results:
        print("❌ No indoor vs outdoor comparison results to visualize!")
        return None
    
    df = pd.DataFrame(comparison_results)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    np.random.seed(RANDOM_SEED)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Indoor vs Outdoor Performance Comparison\n4 Best Electrodes Only', 
                 fontsize=16, fontweight='bold', color='darkblue')
    
    # 1. Individual participant comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    x_pos = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, df['accuracy_indoor'], width, 
                    label='Indoor', color='lightblue', alpha=0.8,
                    yerr=df['accuracy_indoor_std'], capsize=5)
    bars2 = ax1.bar(x_pos + width/2, df['accuracy_outdoor'], width,
                    label='Outdoor', color='lightgreen', alpha=0.8,
                    yerr=df['accuracy_outdoor_std'], capsize=5)
    
    ax1.set_xlabel('Participants', fontweight='bold')
    ax1.set_ylabel('Classification Accuracy', fontweight='bold')
    ax1.set_title('Indoor vs Outdoor Accuracy by Participant', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([p.replace('sub-', '') for p in df['participant']], rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars1, df['accuracy_indoor']):
        ax1.text(bar.get_x() + bar.get_width()/2., acc + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    for bar, acc in zip(bars2, df['accuracy_outdoor']):
        ax1.text(bar.get_x() + bar.get_width()/2., acc + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Performance ratio (Outdoor/Indoor)
    ax2 = fig.add_subplot(gs[0, 1])
    
    ratio_colors = ['green' if r >= 1.0 else 'orange' if r >= 0.9 else 'red' 
                    for r in df['performance_ratio_outdoor_indoor']]
    bars = ax2.bar(range(len(df)), df['performance_ratio_outdoor_indoor'], 
                   color=ratio_colors, alpha=0.7)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.8, label='Equal Performance')
    ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.6, label='90% Performance')
    
    ax2.set_xlabel('Participants', fontweight='bold')
    ax2.set_ylabel('Performance Ratio\n(Outdoor/Indoor)', fontweight='bold')
    ax2.set_title('Outdoor vs Indoor Performance Ratio', fontweight='bold')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([p.replace('sub-', '') for p in df['participant']], rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars, df['performance_ratio_outdoor_indoor'])):
        ax2.text(bar.get_x() + bar.get_width()/2., ratio + 0.02,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Statistical distribution comparison
    ax3 = fig.add_subplot(gs[1, 0])
    
    data_indoor = df['accuracy_indoor'].values
    data_outdoor = df['accuracy_outdoor'].values
    
    bp = ax3.boxplot([data_indoor, data_outdoor], 
                     labels=['Indoor', 'Outdoor'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('lightgreen')
    bp['boxes'][1].set_alpha(0.7)
    
    ax3.set_ylabel('Classification Accuracy', fontweight='bold')
    ax3.set_title('Statistical Distribution Comparison', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add mean lines
    ax3.axhline(y=data_indoor.mean(), color='blue', linestyle='--', alpha=0.7)
    ax3.axhline(y=data_outdoor.mean(), color='green', linestyle='--', alpha=0.7)
    
    # 4. Difference analysis (Indoor - Outdoor)
    ax4 = fig.add_subplot(gs[1, 1])
    
    differences = df['accuracy_difference_indoor_outdoor'].values
    colors = ['blue' if d >= 0 else 'green' for d in differences]
    bars = ax4.bar(range(len(df)), differences, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax4.axhline(y=differences.mean(), color='purple', linestyle='--', alpha=0.8,
                label=f'Average: {differences.mean():.3f}')
    
    ax4.set_xlabel('Participants', fontweight='bold')
    ax4.set_ylabel('Accuracy Difference\n(Indoor - Outdoor)', fontweight='bold')
    ax4.set_title('Indoor vs Outdoor Difference Analysis', fontweight='bold')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels([p.replace('sub-', '') for p in df['participant']], rotation=45)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        y_pos = diff + 0.002 if diff >= 0 else diff - 0.005
        ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{diff:.3f}', ha='center', va='bottom' if diff >= 0 else 'top', fontsize=8)
    
    # Save the comparison plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"indoor_outdoor_4electrodes_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Indoor vs Outdoor comparison plot saved to: {plot_path}")
    
    plt.show()
    return fig

def create_session_visualization(all_results_df, session_type="indoor"):
    """Create comprehensive visualization of electrode importance for specified session type"""
    
    # Set up the plotting style with reproducible seed
    plt.style.use('default')
    sns.set_palette("Set2")
    np.random.seed(RANDOM_SEED)  # Ensure reproducible color assignments
    
    # Create figure with subplots - 3x2 layout without voting
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    session_title = session_type.upper()
    fig.suptitle(f'EEG Electrode Importance Analysis\n{session_title} SESSIONS ONLY', 
                 fontsize=18, fontweight='bold', color='darkblue')
    
    # 1. Average Ranking Across All Analyses
    ax1 = fig.add_subplot(gs[0, 0])
    avg_ranks = {electrode: np.mean(ranks) for electrode, ranks in ELECTRODE_RANKINGS.items()}
    rank_data = pd.Series(avg_ranks).sort_values(ascending=True)  # Lower rank = more important
    
    bars1 = ax1.bar(rank_data.index, rank_data.values, color='steelblue', alpha=0.8, edgecolor='darkblue')
    ax1.set_ylabel('Average Ranking (Lower = More Important)', fontweight='bold')
    ax1.set_title(f'{session_title} Sessions: Average Electrode Rankings', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, rank_data.values):
        ax1.text(bar.get_x() + bar.get_width()/2, value + 0.1, 
                f'{value:.1f}', ha='center', fontweight='bold')
    
    # 2. Rank Sum Analysis (Higher = Worse)
    ax2 = fig.add_subplot(gs[0, 1])
    rank_sum_data = pd.Series(ELECTRODE_RANK_SUMS).sort_values(ascending=True)  # Lower sum = better
    
    bars2 = ax2.bar(rank_sum_data.index, rank_sum_data.values, color='darkred', alpha=0.7, edgecolor='darkblue')
    ax2.set_ylabel('Cumulative Rank Sum\n(Lower = Better)', fontweight='bold')
    ax2.set_title(f'{session_title} Sessions: Rank Sum Analysis\n(Lower Sum = Better Electrode)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels for rank sum
    for bar, value in zip(bars2, rank_sum_data.values):
        ax2.text(bar.get_x() + bar.get_width()/2, value + 0.5, 
                str(int(value)), ha='center', fontweight='bold')
    
    # 3. Ranking Consistency (Standard Deviation)
    ax3 = fig.add_subplot(gs[1, 0])
    rank_stds = {electrode: np.std(ranks) for electrode, ranks in ELECTRODE_RANKINGS.items()}
    std_data = pd.Series(rank_stds).sort_values(ascending=True)
    
    bars3 = ax3.bar(std_data.index, std_data.values, color='lightsteelblue', alpha=0.8, edgecolor='darkblue')
    ax3.set_ylabel('Ranking Standard Deviation\n(Lower = More Consistent)', fontweight='bold')
    ax3.set_title(f'{session_title} Sessions: Ranking Consistency', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Create data for remaining plots
    loo_data = all_results_df[all_results_df['condition'] == 'leave_one_out'].copy()
    loo_data['participant_id'] = loo_data['participant']
    
    # 4. Distribution of accuracy drops by electrode - spanning full width  
    ax4 = fig.add_subplot(gs[2, :])
    
    # Box plot of accuracy drops
    electrode_drops_list = []
    electrode_labels = []
    
    for electrode in ALL_ELECTRODES:
        drops = loo_data[loo_data['excluded_electrode'] == electrode]['accuracy_drop'].values
        if len(drops) > 0:
            electrode_drops_list.append(drops)
            electrode_labels.append(electrode)
    
    # Generate colors for box plots
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(electrode_labels)))
    bp = ax4.boxplot(electrode_drops_list, labels=electrode_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Accuracy Drop Distribution', fontweight='bold')
    ax4.set_title(f'{session_title} Sessions: Electrode Importance Distributions', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # Save the comprehensive plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"electrode_importance_{session_type}_only_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 {session_title}-only analysis plot saved to: {plot_path}")
    
    plt.show()
    return fig

def main():
    """Run electrode importance analysis for specified session types"""
    # Clear global variables to prevent data accumulation from previous runs
    global ALL_RESULTS, ELECTRODE_RANKINGS, ELECTRODE_RANK_SUMS, ACCURACY_COMPARISON_RESULTS, INDOOR_OUTDOOR_COMPARISON_RESULTS
    ALL_RESULTS.clear()
    ELECTRODE_RANKINGS.clear()
    ELECTRODE_RANK_SUMS.clear()
    ACCURACY_COMPARISON_RESULTS.clear()
    INDOOR_OUTDOOR_COMPARISON_RESULTS.clear()
    
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
    
    total_planned = len(participants) * len(SESSION_TYPES)
    print(f"📋 Total analyses planned: {total_planned}")
    
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
    
    # 8 vs 4 Electrode Comparison Analysis
    print(f"\n🔬 STARTING 8 vs 4 ELECTRODE COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Identify best 4 electrodes based on average ranking
    best_4_electrodes = [electrode for electrode, _ in rank_sorted[:4]]
    print(f"🏆 Best 4 electrodes identified: {best_4_electrodes}")
    
    # Run 8 vs 4 comparison for each participant
    comparison_completed = 0
    for participant in participants:
        for session_type in SESSION_TYPES:
            comparison_result = compare_8vs4_electrodes(participant, session_type, best_4_electrodes)
            if comparison_result:
                ACCURACY_COMPARISON_RESULTS.append(comparison_result)
                comparison_completed += 1
    
    print(f"\n✅ COMPLETED: {comparison_completed}/{total_planned} 8 vs 4 comparisons")
    
    # Generate 8 vs 4 comparison visualization
    if ACCURACY_COMPARISON_RESULTS:
        print(f"\n🎨 Generating 8 vs 4 electrode comparison visualization...")
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
    
    # Indoor vs Outdoor Comparison Analysis (4 electrodes)
    print(f"\n🏠🌳 STARTING INDOOR vs OUTDOOR COMPARISON ANALYSIS (4 electrodes)")
    print("=" * 60)
    print(f"🏆 Using best 4 electrodes: {best_4_electrodes}")
    
    # Storage for indoor vs outdoor comparison results
    INDOOR_OUTDOOR_COMPARISON_RESULTS = []
    
    # Run indoor vs outdoor comparison for each participant
    indoor_outdoor_completed = 0
    for participant in participants:
        comparison_result = compare_indoor_vs_outdoor_4electrodes(participant, best_4_electrodes)
        if comparison_result:
            INDOOR_OUTDOOR_COMPARISON_RESULTS.append(comparison_result)
            indoor_outdoor_completed += 1
    
    print(f"\n✅ COMPLETED: {indoor_outdoor_completed}/{len(participants)} indoor vs outdoor comparisons")
    
    # Generate indoor vs outdoor comparison visualization and analysis
    if INDOOR_OUTDOOR_COMPARISON_RESULTS:
        print(f"\n🎨 Generating indoor vs outdoor comparison visualization...")
        create_indoor_outdoor_comparison_visualization(INDOOR_OUTDOOR_COMPARISON_RESULTS)
        
        # Save comparison results to CSV
        indoor_outdoor_df = pd.DataFrame(INDOOR_OUTDOOR_COMPARISON_RESULTS)
        indoor_outdoor_path = output_dir / f"indoor_outdoor_4electrodes_comparison.csv"
        indoor_outdoor_df.to_csv(indoor_outdoor_path, index=False)
        print(f"💾 Indoor vs Outdoor comparison results saved to: {indoor_outdoor_path}")
        
        # Print summary statistics
        print(f"\n📊 INDOOR vs OUTDOOR COMPARISON SUMMARY (4 electrodes):")
        print("=" * 60)
        avg_indoor_acc = indoor_outdoor_df['accuracy_indoor'].mean()
        avg_outdoor_acc = indoor_outdoor_df['accuracy_outdoor'].mean()
        avg_diff = indoor_outdoor_df['accuracy_difference_indoor_outdoor'].mean()
        avg_ratio = indoor_outdoor_df['performance_ratio_outdoor_indoor'].mean() * 100
        
        print(f"Average indoor accuracy: {avg_indoor_acc:.3f}")
        print(f"Average outdoor accuracy: {avg_outdoor_acc:.3f}")
        print(f"Average difference (Indoor-Outdoor): {avg_diff:.3f}")
        print(f"Outdoor performance: {avg_ratio:.1f}% of indoor performance")
        
        # Count participants where outdoor performs nearly as well as indoor (within 10%)
        outdoor_efficient_count = sum(1 for ratio in indoor_outdoor_df['performance_ratio_outdoor_indoor'] if ratio >= 0.90)
        print(f"Participants with outdoor ≥90% of indoor performance: {outdoor_efficient_count}/{len(indoor_outdoor_df)}")
        
        # Identify best and worst outdoor performers relative to indoor
        best_outdoor_ratio = indoor_outdoor_df.loc[indoor_outdoor_df['performance_ratio_outdoor_indoor'].idxmax()]
        worst_outdoor_ratio = indoor_outdoor_df.loc[indoor_outdoor_df['performance_ratio_outdoor_indoor'].idxmin()]
        print(f"Best outdoor/indoor ratio: {best_outdoor_ratio['participant']} ({best_outdoor_ratio['performance_ratio_outdoor_indoor']:.3f})")
        print(f"Worst outdoor/indoor ratio: {worst_outdoor_ratio['participant']} ({worst_outdoor_ratio['performance_ratio_outdoor_indoor']:.3f})")
        
        # Environment preference analysis
        indoor_better_count = sum(1 for diff in indoor_outdoor_df['accuracy_difference_indoor_outdoor'] if diff > 0)
        outdoor_better_count = sum(1 for diff in indoor_outdoor_df['accuracy_difference_indoor_outdoor'] if diff < 0)
        print(f"Participants performing better indoors: {indoor_better_count}/{len(indoor_outdoor_df)}")
        print(f"Participants performing better outdoors: {outdoor_better_count}/{len(indoor_outdoor_df)}")
    else:
        print("❌ No indoor vs outdoor comparison results to analyze!")
    
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
    print(f"📊 Comparison: {comparison_completed} 8 vs 4 electrode comparisons → visualization")
    print(f"📊 Environment: {indoor_outdoor_completed} indoor vs outdoor comparisons → visualization")
    
    return all_results_df, avg_rankings, ACCURACY_COMPARISON_RESULTS, INDOOR_OUTDOOR_COMPARISON_RESULTS

if __name__ == "__main__":
    results_df, rankings, comparison_results, indoor_outdoor_results = main()
