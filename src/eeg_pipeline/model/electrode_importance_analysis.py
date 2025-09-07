# --- Electrode Importance Analysis: Leave-One-Out per Electrode ---
import numpy as np, pandas as pd, mne, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Konfiguration
PARTICIPANT = "julian"  # Wähle einen Teilnehmer
SESSION_TYPE = "outdoor"  # Session type
ALL_ELECTRODES = ["EEG1", "EEG2", "EEG3", "EEG4", "EEG5", "EEG6", "EEG7", "EEG8"]

# === ANALYSIS CONFIGURATION ===
INCLUDE_0_BACK = False  # Set to True to include 0-back condition for attention vs WM analysis

def load_single_participant_session(participant, session_type):
    """Load epochs for a single participant and session"""
    processed_dir = Path("results/processed")
    epo_file = processed_dir / participant / f"{session_type}_processed-epo.fif"

    if not epo_file.exists():
        raise FileNotFoundError(f"Epoched file not found: {epo_file}")

    print(f"Loading data from: {epo_file}")
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)

    # The difficulty labels are already assigned in epoching.py:
    # 'baseline': 0, '0-back': 1, '1-back': 2, '2-back': 3, '3-back': 4
    
    # Configure which conditions to analyze based on INCLUDE_0_BACK setting
    if INCLUDE_0_BACK:
        # Include 0-back for attention vs working memory analysis
        analysis_events = ['0-back', '1-back', '2-back', '3-back']
        print("Analysis mode: Attention (0-back) vs Working Memory (1,2,3-back)")
        event_id_to_difficulty = {
            epochs.event_id['0-back']: 0,  # 1 -> 0 (attention)
            epochs.event_id['1-back']: 1,  # 2 -> 1 (low WM)
            epochs.event_id['2-back']: 2,  # 3 -> 2 (medium WM)
            epochs.event_id['3-back']: 3   # 4 -> 3 (high WM)
        }
    else:
        # Traditional working memory load analysis only
        analysis_events = ['1-back', '2-back', '3-back']
        print("Analysis mode: Working Memory Load only (1,2,3-back)")
        event_id_to_difficulty = {
            epochs.event_id['1-back']: 1,  # 2 -> 1
            epochs.event_id['2-back']: 2,  # 3 -> 2  
            epochs.event_id['3-back']: 3   # 4 -> 3
        }
    
    # Filter epochs to include only analysis conditions
    epochs_filtered = epochs[analysis_events]
    
    if len(epochs_filtered) == 0:
        raise ValueError(f"No analysis epochs found in {epo_file}")
    
    # Extract difficulty labels directly from event IDs
    difficulties = [event_id_to_difficulty[event_id] for event_id in epochs_filtered.events[:, 2]]

    # Create metadata
    metadata = pd.DataFrame({
        'difficulty': difficulties,
        'participant': [participant] * len(epochs_filtered),
        'session_type': [session_type] * len(epochs_filtered)
    })
    epochs_filtered.metadata = metadata

    print(f"Loaded {len(epochs_filtered)} analysis epochs from {participant} ({session_type})")
    print(f"Event distribution: {dict(zip(*np.unique(difficulties, return_counts=True)))}")

    return epochs_filtered

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

def plot_electrode_importance(results_df, participant, session_type, baseline_accuracy):
    """Create comprehensive visualization of electrode importance ranking"""
    
    # Filter out baseline and sort by accuracy drop
    loo_results = results_df[results_df['excluded_electrode'] != 'None'].copy()
    loo_results = loo_results.sort_values('accuracy_drop', ascending=True)  # Ascending for better plot layout
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Electrode Importance Analysis: {participant.title()} ({session_type})', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Horizontal bar chart of accuracy drops
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(loo_results)))
    bars = ax1.barh(loo_results['excluded_electrode'], loo_results['accuracy_drop'], 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Accuracy Drop When Excluded', fontweight='bold')
    ax1.set_ylabel('Excluded Electrode', fontweight='bold')
    ax1.set_title('Electrode Importance Ranking\n(Higher drop = More important)', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, loo_results['accuracy_drop'])):
        ax1.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontweight='bold', fontsize=10)
    
    # Plot 2: Accuracy comparison (baseline vs without each electrode)
    electrodes = loo_results['excluded_electrode'].tolist()
    baseline_acc = [baseline_accuracy] * len(electrodes)
    reduced_acc = loo_results['accuracy_mean'].tolist()
    
    x_pos = np.arange(len(electrodes))
    width = 0.35
    
    ax2.bar(x_pos - width/2, baseline_acc, width, label='Baseline (All electrodes)', 
            color='lightblue', alpha=0.8, edgecolor='black')
    ax2.bar(x_pos + width/2, reduced_acc, width, label='Without electrode', 
            color='lightcoral', alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Excluded Electrode', fontweight='bold')
    ax2.set_ylabel('Classification Accuracy', fontweight='bold')
    ax2.set_title('Accuracy Comparison', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(electrodes, rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Statistical significance visualization
    baseline_std = results_df[results_df['excluded_electrode'] == 'None']['accuracy_std'].iloc[0]
    significance_threshold = 2 * baseline_std
    
    # Color-code by significance
    is_significant = loo_results['accuracy_drop'] > significance_threshold
    sig_colors = ['red' if sig else 'gray' for sig in is_significant]
    
    scatter = ax3.scatter(loo_results['accuracy_drop'], loo_results['accuracy_std'], 
                         c=sig_colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add significance threshold line
    ax3.axvline(x=significance_threshold, color='red', linestyle='--', alpha=0.7, 
                label=f'Significance threshold (2σ = {significance_threshold:.3f})')
    
    # Label points
    for _, row in loo_results.iterrows():
        ax3.annotate(row['excluded_electrode'], 
                    (row['accuracy_drop'], row['accuracy_std']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Accuracy Drop', fontweight='bold')
    ax3.set_ylabel('Standard Deviation', fontweight='bold')
    ax3.set_title('Statistical Significance\n(Red = Significant impact)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary data
    summary_data = []
    baseline_row = results_df[results_df['excluded_electrode'] == 'None'].iloc[0]
    most_important = loo_results.iloc[-1]  # Last in ascending order = highest drop
    least_important = loo_results.iloc[0]   # First in ascending order = lowest drop
    
    summary_data.append(['Metric', 'Value'])
    summary_data.append(['Baseline Accuracy', f"{baseline_row['accuracy_mean']:.3f} ± {baseline_row['accuracy_std']:.3f}"])
    summary_data.append(['Most Important Electrode', f"{most_important['excluded_electrode']} (drop: {most_important['accuracy_drop']:.3f})"])
    summary_data.append(['Least Important Electrode', f"{least_important['excluded_electrode']} (drop: {least_important['accuracy_drop']:.3f})"])
    summary_data.append(['Significant Electrodes', f"{sum(is_significant)} / {len(loo_results)}"])
    summary_data.append(['Mean Accuracy Drop', f"{loo_results['accuracy_drop'].mean():.3f}"])
    summary_data.append(['Max Accuracy Drop', f"{loo_results['accuracy_drop'].max():.3f}"])
    
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data)):
        table[(i, 0)].set_facecolor('#E6E6FA')
        table[(i, 1)].set_facecolor('#F0F8FF')
    
    ax4.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"electrode_importance_plot_{participant}_{session_type}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Electrode importance plot saved to: {plot_path}")
    
    # Show the plot
    plt.show()
    
    return fig

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
    
    # Generate visualization
    print("\n6. Generating electrode importance plot...")
    try:
        baseline_accuracy = baseline_row['accuracy_mean']
        plot_fig = plot_electrode_importance(results_df_sorted, PARTICIPANT, SESSION_TYPE, baseline_accuracy)
        print("✓ Electrode importance visualization created successfully!")
    except Exception as e:
        print(f"⚠ Warning: Could not generate plot - {e}")
        print("Results are still saved in CSV files.")
    
    print(f"\n=== Analysis Complete ===")
    return results_df_sorted

if __name__ == "__main__":
    results = main()
