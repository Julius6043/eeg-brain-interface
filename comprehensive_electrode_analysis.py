# --- Comprehensive Electrode Importance Analysis: All Participants & Sessions ---
"""
This script performs electrode importance analysis for ALL participants and sessions individually,
then creates a voting system to identify the most consistently important electrodes.

Strategy:
1. Run electrode importance analysis for EACH participant-session combination separately
2. For each analysis, rank electrodes by accuracy drop when removed
3. Create a voting system across all analyses
4. Generate comprehensive visualizations and summary statistics

This ensures:
- No data mixing between participants/sessions  
- Individual differences are preserved
- Statistical robustness through multiple independent analyses
- Clear identification of consistently important electrodes
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

# Configuration
ALL_ELECTRODES = ["EEG1", "EEG2", "EEG3", "EEG4", "EEG5", "EEG6", "EEG7", "EEG8"]
INCLUDE_0_BACK = False  # Working memory load analysis only
SESSION_TYPES = ["indoor", "outdoor"]

# Results storage
ALL_RESULTS = []
ELECTRODE_RANKINGS = defaultdict(list)  # electrode -> list of ranks across analyses
ELECTRODE_VOTES = defaultdict(int)      # electrode -> vote count as "most important"

# Auto-detect participants and sessions
def discover_participants_and_sessions():
    """Automatically discover all available participants and sessions"""
    processed_dir = Path("results/processed")
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    
    participants_sessions = []
    
    for participant_dir in processed_dir.iterdir():
        if participant_dir.is_dir() and participant_dir.name.startswith("sub-"):
            participant = participant_dir.name.replace("sub-", "").replace("_", "").lower()
            
            # Look for session files
            for session_file in participant_dir.glob("*_processed-epo.fif"):
                session_type = session_file.stem.replace("_processed-epo", "")
                participants_sessions.append((participant, session_type))
    
    print(f"Discovered {len(participants_sessions)} participant-session combinations:")
    for p, s in participants_sessions:
        print(f"  - {p}: {s}")
    
    return participants_sessions

def load_single_participant_session(participant, session_type):
    """Load epochs for a single participant and session"""
    processed_dir = Path("results/processed")
    
    # Try different naming conventions
    possible_dirs = [
        processed_dir / f"sub-{participant}",
        processed_dir / f"sub-{participant.upper()}",
        processed_dir / f"sub-{participant.capitalize()}",
        processed_dir / participant,
        processed_dir / participant.upper(),
        processed_dir / participant.capitalize()
    ]
    
    epo_file = None
    for participant_dir in possible_dirs:
        if participant_dir.exists():
            potential_file = participant_dir / f"{session_type}_processed-epo.fif"
            if potential_file.exists():
                epo_file = potential_file
                break
    
    if epo_file is None:
        raise FileNotFoundError(f"Could not find epochs file for {participant} {session_type}")

    print(f"Loading data from: {epo_file}")
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)

    # Configure which conditions to analyze
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
    
    # Filter epochs to include only analysis conditions
    epochs_filtered = epochs[analysis_events]
    
    if len(epochs_filtered) == 0:
        raise ValueError(f"No analysis epochs found for {participant} {session_type}")
    
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
        n_estimators=1000,  # Reduced for speed across many participants
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

def analyze_participant_session(participant, session_type):
    """Run electrode importance analysis for a single participant-session"""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {participant.upper()} - {session_type.upper()}")
    print(f"{'='*60}")
    
    try:
        # Load data
        epochs = load_single_participant_session(participant, session_type)
        y = epochs.metadata["difficulty"].astype(int).to_numpy()
        
        results = []
        
        # Baseline: All electrodes
        print(f"\n→ Baseline performance (all electrodes)...")
        X_all, col_names_all, ep_filt_all = extract_features(epochs)
        baseline_mean, baseline_std, baseline_scores = train_and_evaluate_rf(X_all, y)
        
        print(f"   Baseline accuracy: {baseline_mean:.3f} ± {baseline_std:.3f}")
        
        results.append({
            'participant': participant,
            'session_type': session_type,
            'condition': 'All_electrodes',
            'excluded_electrode': 'None',
            'n_channels': len(ep_filt_all.ch_names),
            'accuracy_mean': baseline_mean,
            'accuracy_std': baseline_std,
            'accuracy_drop': 0.0
        })
        
        # Leave-one-out analysis
        print(f"→ Leave-one-out analysis...")
        
        for i, electrode in enumerate(ALL_ELECTRODES):
            print(f"   Excluding {electrode} ({i+1}/{len(ALL_ELECTRODES)})... ", end="")
            
            try:
                # Extract features without this electrode
                X_excl, col_names_excl, ep_filt_excl = extract_features(epochs, exclude_channels=[electrode])
                
                if X_excl.shape[1] == 0:
                    print("No features left, skipping")
                    continue
                    
                # Train and evaluate
                acc_mean, acc_std, cv_scores = train_and_evaluate_rf(X_excl, y)
                accuracy_drop = baseline_mean - acc_mean
                
                print(f"Acc: {acc_mean:.3f}, Drop: {accuracy_drop:.3f}")
                
                results.append({
                    'participant': participant,
                    'session_type': session_type,
                    'condition': f'Without_{electrode}',
                    'excluded_electrode': electrode,
                    'n_channels': len(ep_filt_excl.ch_names),
                    'accuracy_mean': acc_mean,
                    'accuracy_std': acc_std,
                    'accuracy_drop': accuracy_drop
                })
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"ERROR analyzing {participant} {session_type}: {e}")
        return pd.DataFrame()

def create_electrode_importance_tally(all_results_df):
    """Create tally of electrode importance across all participants/sessions"""
    
    print(f"\n{'='*60}")
    print("CREATING ELECTRODE IMPORTANCE TALLY")
    print(f"{'='*60}")
    
    # Initialize tracking structures
    electrode_rankings = defaultdict(list)  # electrode -> list of rankings
    electrode_drops = defaultdict(list)     # electrode -> list of accuracy drops
    session_results = []
    
    # Process each participant-session combination
    for (participant, session), group in all_results_df.groupby(['participant', 'session_type']):
        baseline_row = group[group['excluded_electrode'] == 'None']
        loo_rows = group[group['excluded_electrode'] != 'None']
        
        if len(baseline_row) == 0 or len(loo_rows) == 0:
            continue
            
        baseline_acc = baseline_row['accuracy_mean'].iloc[0]
        baseline_std = baseline_row['accuracy_std'].iloc[0]
        
        # Sort electrodes by accuracy drop (descending)
        loo_sorted = loo_rows.sort_values('accuracy_drop', ascending=False)
        
        # Record rankings (1 = most important, 8 = least important)
        for rank, (_, row) in enumerate(loo_sorted.iterrows(), 1):
            electrode = row['excluded_electrode']
            accuracy_drop = row['accuracy_drop']
            
            electrode_rankings[electrode].append(rank)
            electrode_drops[electrode].append(accuracy_drop)
        
        # Track session-level results
        most_important = loo_sorted.iloc[0]['excluded_electrode']
        session_results.append({
            'participant': participant,
            'session_type': session,
            'baseline_accuracy': baseline_acc,
            'baseline_std': baseline_std,
            'most_important_electrode': most_important,
            'max_accuracy_drop': loo_sorted.iloc[0]['accuracy_drop'],
            'n_electrodes_tested': len(loo_sorted)
        })
    
    # Calculate tally statistics
    tally_results = []
    for electrode in ALL_ELECTRODES:
        if electrode in electrode_rankings:
            rankings = electrode_rankings[electrode]
            drops = electrode_drops[electrode]
            
            # Count how many times this electrode was #1 most important
            times_most_important = sum(1 for rank in rankings if rank == 1)
            # Count how many times in top 3
            times_top3 = sum(1 for rank in rankings if rank <= 3)
            # Count how many times in bottom 3
            times_bottom3 = sum(1 for rank in rankings if rank >= 6)
            
            tally_results.append({
                'electrode': electrode,
                'n_sessions': len(rankings),
                'times_most_important': times_most_important,
                'times_top3': times_top3,
                'times_bottom3': times_bottom3,
                'mean_ranking': np.mean(rankings),
                'std_ranking': np.std(rankings),
                'mean_accuracy_drop': np.mean(drops),
                'std_accuracy_drop': np.std(drops),
                'max_accuracy_drop': np.max(drops),
                'vote_score': times_most_important * 3 + times_top3 * 1  # Weighted voting score
            })
    
    tally_df = pd.DataFrame(tally_results)
    session_df = pd.DataFrame(session_results)
    
    # Sort by vote score (most important first)
    tally_df = tally_df.sort_values('vote_score', ascending=False)
    
    return tally_df, session_df

def plot_comprehensive_tally(tally_df, session_df, all_results_df):
    """Create comprehensive visualization of electrode importance tally"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive EEG Electrode Importance Analysis\nAll Participants & Sessions', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Vote tally (most important metric)
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(tally_df)))
    bars1 = ax1.bar(tally_df['electrode'], tally_df['vote_score'], 
                    color=colors, alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Electrode', fontweight='bold')
    ax1.set_ylabel('Vote Score', fontweight='bold')
    ax1.set_title('Electrode Importance Vote Tally\n(3pts for #1, 1pt for top-3)', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, tally_df['vote_score']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{int(value)}', ha='center', fontweight='bold')
    
    # Plot 2: Times ranked #1 most important
    bars2 = ax2.bar(tally_df['electrode'], tally_df['times_most_important'], 
                    color='lightcoral', alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Electrode', fontweight='bold')
    ax2.set_ylabel('Times Ranked #1 Most Important', fontweight='bold')
    ax2.set_title('Consistency of Top Ranking', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, tally_df['times_most_important']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{int(value)}', ha='center', fontweight='bold')
    
    # Plot 3: Mean accuracy drop across sessions
    bars3 = ax3.bar(tally_df['electrode'], tally_df['mean_accuracy_drop'], 
                    yerr=tally_df['std_accuracy_drop'], capsize=5,
                    color='lightblue', alpha=0.8, edgecolor='black')
    
    ax3.set_xlabel('Electrode', fontweight='bold')
    ax3.set_ylabel('Mean Accuracy Drop', fontweight='bold')
    ax3.set_title('Average Impact Across Sessions', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Top 3 electrodes summary
    top3 = tally_df.head(3)
    total_sessions = len(session_df)
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Participants-Sessions', f"{total_sessions}"],
        ['#1 Most Important Electrode', f"{top3.iloc[0]['electrode']} ({top3.iloc[0]['vote_score']} votes)"],
        ['#2 Most Important Electrode', f"{top3.iloc[1]['electrode']} ({top3.iloc[1]['vote_score']} votes)"],
        ['#3 Most Important Electrode', f"{top3.iloc[2]['electrode']} ({top3.iloc[2]['vote_score']} votes)"],
        ['Most Consistent (low std)', f"{tally_df.loc[tally_df['std_ranking'].idxmin(), 'electrode']}"],
        ['Highest Max Drop', f"{tally_df.loc[tally_df['max_accuracy_drop'].idxmax(), 'electrode']} ({tally_df['max_accuracy_drop'].max():.3f})"]
    ]
    
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0], 
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data)):
        table[(i, 0)].set_facecolor('#E6E6FA')
        table[(i, 1)].set_facecolor('#F0F8FF')
    
    ax4.set_title('Summary Results', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "electrode_importance_tally_all_participants.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Tally plot saved to: {plot_path}")
    
    plt.show()
    return fig

def main():
    """Run comprehensive electrode importance analysis across all participants and sessions"""
    
    print("="*80)
    print("COMPREHENSIVE ELECTRODE IMPORTANCE ANALYSIS")
    print("All Participants & Sessions")
    print("="*80)
    
    # 1. Discover all available participants and sessions
    print("\n1. Discovering participants and sessions...")
    try:
        participants_sessions = discover_participants_and_sessions()
        if not participants_sessions:
            print("No participants/sessions found!")
            return
    except Exception as e:
        print(f"Error discovering data: {e}")
        return
    
    # 2. Run analysis for each participant-session
    print(f"\n2. Running analysis for {len(participants_sessions)} participant-session combinations...")
    all_results = []
    
    for i, (participant, session_type) in enumerate(participants_sessions, 1):
        print(f"\n[{i}/{len(participants_sessions)}] Processing {participant} - {session_type}")
        
        result_df = analyze_participant_session(participant, session_type)
        if not result_df.empty:
            all_results.append(result_df)
        else:
            print(f"   ⚠ Failed to analyze {participant} - {session_type}")
    
    if not all_results:
        print("❌ No successful analyses completed!")
        return
    
    # 3. Combine all results
    print(f"\n3. Combining results from {len(all_results)} successful analyses...")
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    # 4. Create electrode importance tally
    print(f"\n4. Creating electrode importance tally...")
    tally_df, session_df = create_electrode_importance_tally(all_results_df)
    
    # 5. Display results
    print(f"\n{'='*60}")
    print("ELECTRODE IMPORTANCE TALLY RESULTS")
    print(f"{'='*60}")
    print(f"Based on {len(session_df)} participant-session combinations\n")
    
    print("🏆 ELECTRODE RANKING (by vote score):")
    print(tally_df[['electrode', 'vote_score', 'times_most_important', 'times_top3', 'mean_accuracy_drop']].to_string(index=False))
    
    # 6. Save results
    print(f"\n5. Saving comprehensive results...")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Save tally results
    tally_path = output_dir / "electrode_importance_tally_all_participants.csv"
    tally_df.to_csv(tally_path, index=False)
    print(f"Tally results saved to: {tally_path}")
    
    # Save session results
    session_path = output_dir / "electrode_importance_sessions_summary.csv"
    session_df.to_csv(session_path, index=False)
    print(f"Session summary saved to: {session_path}")
    
    # Save complete results
    complete_path = output_dir / "electrode_importance_complete_results.csv"
    all_results_df.to_csv(complete_path, index=False)
    print(f"Complete results saved to: {complete_path}")
    
    # 7. Create comprehensive visualization
    print(f"\n6. Creating comprehensive visualization...")
    try:
        plot_fig = plot_comprehensive_tally(tally_df, session_df, all_results_df)
        print("✅ Comprehensive electrode importance tally visualization created!")
    except Exception as e:
        print(f"⚠ Warning: Could not generate plot - {e}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Analyzed {len(session_df)} participant-session combinations")
    print(f"🏆 Top 3 electrodes by vote score:")
    for i, (_, row) in enumerate(tally_df.head(3).iterrows(), 1):
        print(f"   {i}. {row['electrode']} (score: {row['vote_score']}, #1 rankings: {row['times_most_important']})")
    
    return all_results_df, tally_df, session_df

if __name__ == "__main__":
    all_results, tally_results, session_results = main()
