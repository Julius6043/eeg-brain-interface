#!/usr/bin/env python3
"""
EEG Electrode Comparison: 8 Electrodes vs Best 4 Electrodes (Indoor Analysis)

This script compares classification accuracy between:
1. All 8 electrodes (baseline)
2. Top 4 electrodes based on indoor electrode importance ranking

Based on indoor analysis results:
- EEG7: avg rank 2.9 (±2.0) - #1 overall
- EEG8: avg rank 3.6 (±1.6) - #2 overall  
- EEG4: avg rank 3.6 (±2.3) - #3 overall
- EEG1: avg rank 4.5 (±2.1) - #4 overall

Best 4 electrodes: [EEG7, EEG8, EEG4, EEG1]
"""

import numpy as np, pandas as pd, mne
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configuration
ALL_ELECTRODES = ["EEG1", "EEG2", "EEG3", "EEG4", "EEG5", "EEG6", "EEG7", "EEG8"]
BEST_4_ELECTRODES = ["EEG7", "EEG8", "EEG4", "EEG1"]  # Based on indoor ranking results
INCLUDE_0_BACK = False  # Working memory load analysis only
SESSION_TYPES = ["indoor"]  # Indoor only

# Results storage
comparison_results = []

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

def extract_features(epochs_data, electrode_subset=None):
    """Extract bandpower features from epochs with optional electrode subset"""
    bands = {
        "theta": (4, 8),
        "alpha": (8, 13), 
        "beta": (13, 30),
        "gamma": (30, 40),
    }

    ep_filt = epochs_data.copy().filter(4.0, 40.0, picks="eeg")
    
    # Select specific electrodes if subset provided
    if electrode_subset:
        available_channels = [ch for ch in electrode_subset if ch in ep_filt.ch_names]
        if len(available_channels) != len(electrode_subset):
            missing = set(electrode_subset) - set(available_channels)
            print(f"⚠ Missing electrodes: {missing}")
        ep_filt.pick_channels(available_channels)
        print(f"  → Using {len(available_channels)} electrodes: {available_channels}")

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

def compare_electrode_sets(participant, session_type):
    """Compare 8 electrodes vs best 4 electrodes for a participant"""
    print(f"\n{'='*60}")
    print(f"COMPARING: {participant.upper()} - {session_type.upper()}")
    print(f"{'='*60}")
    
    # Load data
    epochs = load_participant_session(participant, session_type)
    if epochs is None:
        return None
    
    y = epochs.metadata["difficulty"].astype(int).to_numpy()
    results = []
    
    # Test 1: All 8 electrodes
    print("🔹 Testing ALL 8 electrodes...")
    X_all, _, _ = extract_features(epochs, electrode_subset=ALL_ELECTRODES)
    acc_8_mean, acc_8_std, scores_8 = train_and_evaluate_rf(X_all, y)
    
    results.append({
        'participant': participant,
        'session_type': session_type,
        'electrode_set': '8_electrodes',
        'electrodes_used': ALL_ELECTRODES,
        'n_electrodes': 8,
        'accuracy_mean': acc_8_mean,
        'accuracy_std': acc_8_std,
        'scores': scores_8.tolist()
    })
    
    print(f"  → 8 electrodes: {acc_8_mean:.3f} ± {acc_8_std:.3f}")
    
    # Test 2: Best 4 electrodes
    print("🔸 Testing BEST 4 electrodes...")
    X_best4, _, _ = extract_features(epochs, electrode_subset=BEST_4_ELECTRODES)
    acc_4_mean, acc_4_std, scores_4 = train_and_evaluate_rf(X_best4, y)
    
    results.append({
        'participant': participant,
        'session_type': session_type,
        'electrode_set': '4_electrodes_best',
        'electrodes_used': BEST_4_ELECTRODES,
        'n_electrodes': 4,
        'accuracy_mean': acc_4_mean,
        'accuracy_std': acc_4_std,
        'scores': scores_4.tolist()
    })
    
    print(f"  → 4 electrodes: {acc_4_mean:.3f} ± {acc_4_std:.3f}")
    
    # Calculate difference
    accuracy_diff = acc_8_mean - acc_4_mean
    print(f"  📊 Difference: {accuracy_diff:.3f} (8-electrode advantage)")
    
    if accuracy_diff < 0.02:  # Less than 2% difference
        print(f"  ✅ Similar performance! 4 electrodes achieve {((acc_4_mean/acc_8_mean)*100):.1f}% of 8-electrode accuracy")
    else:
        print(f"  ⚠ Notable difference: 4 electrodes achieve {((acc_4_mean/acc_8_mean)*100):.1f}% of 8-electrode accuracy")
    
    return results

def create_comparison_visualization(results_df):
    """Create comprehensive visualization comparing 8 vs 4 electrodes"""
    
    plt.style.use('default')
    np.random.seed(RANDOM_SEED)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EEG Electrode Comparison: 8 Electrodes vs Best 4 Electrodes\n(Indoor Sessions Only)', 
                 fontsize=16, fontweight='bold', color='darkblue')
    
    # Prepare data
    df_8 = results_df[results_df['electrode_set'] == '8_electrodes'].copy()
    df_4 = results_df[results_df['electrode_set'] == '4_electrodes_best'].copy()
    
    # 1. Side-by-side accuracy comparison
    ax1 = axes[0, 0]
    participants = df_8['participant'].values
    x_pos = np.arange(len(participants))
    
    width = 0.35
    bars1 = ax1.bar(x_pos - width/2, df_8['accuracy_mean'], width, 
                    yerr=df_8['accuracy_std'], label='8 Electrodes', 
                    color='steelblue', alpha=0.8, capsize=5)
    bars2 = ax1.bar(x_pos + width/2, df_4['accuracy_mean'], width,
                    yerr=df_4['accuracy_std'], label='Best 4 Electrodes',
                    color='orange', alpha=0.8, capsize=5)
    
    ax1.set_xlabel('Participants', fontweight='bold')
    ax1.set_ylabel('Classification Accuracy', fontweight='bold')
    ax1.set_title('Accuracy Comparison by Participant', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(participants, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Paired difference plot
    ax2 = axes[0, 1]
    differences = df_8['accuracy_mean'].values - df_4['accuracy_mean'].values
    colors = ['green' if d <= 0.02 else 'red' if d > 0.05 else 'orange' for d in differences]
    
    bars = ax2.bar(participants, differences, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=0.02, color='green', linestyle='--', alpha=0.5, label='±2% threshold')
    ax2.axhline(y=-0.02, color='green', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Participants', fontweight='bold')
    ax2.set_ylabel('Accuracy Difference\n(8 electrodes - 4 electrodes)', fontweight='bold')
    ax2.set_title('Performance Difference', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Average performance summary
    ax3 = axes[0, 2]
    avg_8 = df_8['accuracy_mean'].mean()
    avg_4 = df_4['accuracy_mean'].mean()
    std_8 = df_8['accuracy_mean'].std()
    std_4 = df_4['accuracy_mean'].std()
    
    bars = ax3.bar(['8 Electrodes', 'Best 4 Electrodes'], [avg_8, avg_4],
                   yerr=[std_8, std_4], color=['steelblue', 'orange'], 
                   alpha=0.8, capsize=10)
    
    ax3.set_ylabel('Average Accuracy', fontweight='bold')
    ax3.set_title('Overall Performance Comparison', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val, std in zip(bars, [avg_8, avg_4], [std_8, std_4]):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{val:.3f}\n±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Distribution comparison (box plots)
    ax4 = axes[1, 0]
    data_to_plot = [df_8['accuracy_mean'].values, df_4['accuracy_mean'].values]
    bp = ax4.boxplot(data_to_plot, labels=['8 Electrodes', 'Best 4 Electrodes'], 
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('orange')
    
    ax4.set_ylabel('Accuracy Distribution', fontweight='bold')
    ax4.set_title('Accuracy Distribution Comparison', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Efficiency analysis (accuracy per electrode)
    ax5 = axes[1, 1]
    efficiency_8 = df_8['accuracy_mean'] / 8
    efficiency_4 = df_4['accuracy_mean'] / 4
    
    x_pos = np.arange(len(participants))
    bars1 = ax5.bar(x_pos - width/2, efficiency_8, width, 
                    label='8 Electrodes', color='steelblue', alpha=0.8)
    bars2 = ax5.bar(x_pos + width/2, efficiency_4, width,
                    label='Best 4 Electrodes', color='orange', alpha=0.8)
    
    ax5.set_xlabel('Participants', fontweight='bold')
    ax5.set_ylabel('Accuracy per Electrode', fontweight='bold')
    ax5.set_title('Electrode Efficiency Analysis', fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(participants, rotation=45)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Summary statistics table
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')
    
    # Calculate summary statistics
    retention_rate = (avg_4 / avg_8) * 100
    avg_diff = differences.mean()
    significant_drops = sum(1 for d in differences if d > 0.05)
    similar_performance = sum(1 for d in differences if abs(d) <= 0.02)
    
    summary_stats = [
        ['Metric', '8 Electrodes', 'Best 4 Electrodes'],
        ['Average Accuracy', f'{avg_8:.3f}', f'{avg_4:.3f}'],
        ['Std Deviation', f'{std_8:.3f}', f'{std_4:.3f}'],
        ['Best Performance', f'{df_8["accuracy_mean"].max():.3f}', f'{df_4["accuracy_mean"].max():.3f}'],
        ['Worst Performance', f'{df_8["accuracy_mean"].min():.3f}', f'{df_4["accuracy_mean"].min():.3f}'],
        ['', '', ''],
        ['Performance Retention', '', f'{retention_rate:.1f}%'],
        ['Average Difference', '', f'{avg_diff:.3f}'],
        ['Similar Performance (±2%)', '', f'{similar_performance}/{len(participants)}'],
        ['Significant Drops (>5%)', '', f'{significant_drops}/{len(participants)}'],
        ['', '', ''],
        ['Best 4 Electrodes', '', 'EEG7, EEG8, EEG4, EEG1']
    ]
    
    table = ax6.table(cellText=summary_stats[1:], colLabels=summary_stats[0], 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(summary_stats)):
        table[(i, 0)].set_facecolor('#E6F3FF')
        table[(i, 1)].set_facecolor('#F0F8FF')
        table[(i, 2)].set_facecolor('#F0F8FF')
    
    ax6.set_title('Comparison Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "electrode_comparison_8vs4_indoor.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Comparison plot saved to: {plot_path}")
    
    plt.show()
    return fig

def main():
    """Run electrode comparison analysis"""
    print("🔋 EEG ELECTRODE COMPARISON: 8 vs Best 4 Electrodes (Indoor)")
    print("=" * 80)
    print("Comparing classification accuracy between:")
    print("  🔹 All 8 electrodes (baseline)")
    print(f"  🔸 Best 4 electrodes: {BEST_4_ELECTRODES}")
    print("=" * 80)
    
    participants = get_available_participants()
    print(f"📋 Found {len(participants)} participants: {participants}")
    
    all_results = []
    
    for participant in participants:
        for session_type in SESSION_TYPES:
            results = compare_electrode_sets(participant, session_type)
            if results:
                all_results.extend(results)
                comparison_results.extend(results)
    
    if not all_results:
        print("❌ No valid data found for comparison!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "electrode_comparison_8vs4_detailed.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Detailed results saved to: {csv_path}")
    
    # Print summary
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 50)
    
    df_8 = results_df[results_df['electrode_set'] == '8_electrodes']
    df_4 = results_df[results_df['electrode_set'] == '4_electrodes_best']
    
    avg_8 = df_8['accuracy_mean'].mean()
    avg_4 = df_4['accuracy_mean'].mean()
    retention = (avg_4 / avg_8) * 100
    
    print(f"8 Electrodes Average: {avg_8:.3f} ± {df_8['accuracy_mean'].std():.3f}")
    print(f"4 Electrodes Average: {avg_4:.3f} ± {df_4['accuracy_mean'].std():.3f}")
    print(f"Performance Retention: {retention:.1f}%")
    print(f"Average Difference: {avg_8 - avg_4:.3f}")
    
    differences = df_8['accuracy_mean'].values - df_4['accuracy_mean'].values
    similar_count = sum(1 for d in differences if abs(d) <= 0.02)
    print(f"Similar Performance (±2%): {similar_count}/{len(differences)} participants")
    
    if retention >= 95:
        print("✅ EXCELLENT: 4 electrodes maintain >95% performance!")
    elif retention >= 90:
        print("✅ GOOD: 4 electrodes maintain >90% performance!")
    else:
        print("⚠ SIGNIFICANT: Notable performance drop with 4 electrodes")
    
    # Create visualization
    print(f"\n🎨 Generating comparison visualization...")
    create_comparison_visualization(results_df)
    
    print(f"\n✅ ELECTRODE COMPARISON COMPLETE!")

if __name__ == "__main__":
    main()
