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

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configuration
ALL_ELECTRODES = ["EEG1", "EEG2", "EEG3", "EEG4", "EEG5", "EEG6", "EEG7", "EEG8"]
INCLUDE_0_BACK = False  # Working memory load analysis only
SESSION_TYPES = ["indoor", "outdoor"]

# Results storage
ALL_RESULTS = []
ELECTRODE_RANKINGS = defaultdict(list)  # electrode -> list of ranks across analyses
ELECTRODE_VOTES = defaultdict(int)      # electrode -> vote count as "most important"

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
    
    # Baseline performance (all electrodes)
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
        
        # Vote for most important electrode
        most_important = sorted_electrodes[0][0]
        ELECTRODE_VOTES[most_important] += 1
        print(f"🏆 Most important: {most_important}")
    
    return results

def create_comprehensive_visualization(all_results_df):
    """Create comprehensive visualization of electrode importance across all analyses"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Comprehensive EEG Electrode Importance Analysis\nAcross All Participants and Sessions', 
                 fontsize=18, fontweight='bold')
    
    # 1. Electrode Vote Counts (Top Priority)
    ax1 = fig.add_subplot(gs[0, 0])
    vote_data = pd.Series(ELECTRODE_VOTES).sort_values(ascending=True)
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(vote_data)))
    bars = ax1.barh(vote_data.index, vote_data.values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Number of "Most Important" Votes', fontweight='bold')
    ax1.set_title('Electrode Voting Results\n(Most Important Across Sessions)', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, vote_data.values):
        ax1.text(value + 0.1, bar.get_y() + bar.get_height()/2, 
                str(int(value)), va='center', fontweight='bold')
    
    # 2. Average Ranking Across All Analyses
    ax2 = fig.add_subplot(gs[0, 1])
    avg_ranks = {electrode: np.mean(ranks) for electrode, ranks in ELECTRODE_RANKINGS.items()}
    rank_data = pd.Series(avg_ranks).sort_values(ascending=True)  # Lower rank = more important
    
    bars2 = ax2.bar(rank_data.index, rank_data.values, color='lightcoral', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Average Ranking (Lower = More Important)', fontweight='bold')
    ax2.set_title('Average Electrode Rankings', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, rank_data.values):
        ax2.text(bar.get_x() + bar.get_width()/2, value + 0.1, 
                f'{value:.1f}', ha='center', fontweight='bold')
    
    # 3. Ranking Consistency (Standard Deviation)
    ax3 = fig.add_subplot(gs[0, 2])
    rank_stds = {electrode: np.std(ranks) for electrode, ranks in ELECTRODE_RANKINGS.items()}
    std_data = pd.Series(rank_stds).sort_values(ascending=True)
    
    bars3 = ax3.bar(std_data.index, std_data.values, color='lightgreen', alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Ranking Standard Deviation\n(Lower = More Consistent)', fontweight='bold')
    ax3.set_title('Electrode Ranking Consistency', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Heatmap of accuracy drops across participants/sessions
    ax4 = fig.add_subplot(gs[1, :])
    
    # Create pivot table for heatmap
    loo_data = all_results_df[all_results_df['condition'] == 'leave_one_out'].copy()
    loo_data['session_id'] = loo_data['participant'] + '_' + loo_data['session_type']
    
    heatmap_data = loo_data.pivot(index='excluded_electrode', 
                                  columns='session_id', 
                                  values='accuracy_drop')
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='Reds', 
                ax=ax4, cbar_kws={'label': 'Accuracy Drop'})
    ax4.set_title('Accuracy Drop Heatmap: All Participants & Sessions', fontweight='bold')
    ax4.set_xlabel('Participant_Session', fontweight='bold')
    ax4.set_ylabel('Excluded Electrode', fontweight='bold')
    
    # 5. Distribution of accuracy drops by electrode
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Box plot of accuracy drops
    electrode_drops_list = []
    electrode_labels = []
    
    for electrode in ALL_ELECTRODES:
        drops = loo_data[loo_data['excluded_electrode'] == electrode]['accuracy_drop'].values
        if len(drops) > 0:
            electrode_drops_list.append(drops)
            electrode_labels.append(electrode)
    
    bp = ax5.boxplot(electrode_drops_list, labels=electrode_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax5.set_ylabel('Accuracy Drop Distribution', fontweight='bold')
    ax5.set_title('Electrode Importance Distributions', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Summary Statistics Table
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Calculate summary statistics
    total_analyses = len(set(loo_data['participant'] + '_' + loo_data['session_type']))
    most_voted = max(ELECTRODE_VOTES.items(), key=lambda x: x[1]) if ELECTRODE_VOTES else ('N/A', 0)
    most_consistent = min(rank_stds.items(), key=lambda x: x[1]) if rank_stds else ('N/A', 0)
    
    summary_stats = [
        ['Metric', 'Value'],
        ['Total Analyses Completed', f"{total_analyses}"],
        ['Total Participants', f"{len(set(loo_data['participant']))}"],
        ['Sessions per Participant', f"{len(SESSION_TYPES)}"],
        ['Most Voted Electrode', f"{most_voted[0]} ({most_voted[1]} votes)"],
        ['Most Consistent Electrode', f"{most_consistent[0]} (std: {most_consistent[1]:.2f})"],
        ['Average Accuracy Drop', f"{loo_data['accuracy_drop'].mean():.3f}"],
        ['Max Accuracy Drop', f"{loo_data['accuracy_drop'].max():.3f}"],
        ['Analyses with Significant Drops', f"{sum(loo_data['accuracy_drop'] > 0.05)} / {len(loo_data)}"]
    ]
    
    table = ax6.table(cellText=summary_stats[1:], colLabels=summary_stats[0], 
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_stats)):
        table[(i, 0)].set_facecolor('#E6E6FA')
        table[(i, 1)].set_facecolor('#F0F8FF')
    
    ax6.set_title('Comprehensive Summary Statistics', fontweight='bold', pad=20)
    
    # Save the comprehensive plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "comprehensive_electrode_importance_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Comprehensive plot saved to: {plot_path}")
    
    plt.show()
    return fig

def main():
    """Run comprehensive electrode importance analysis across all participants and sessions"""
    print("🧠 COMPREHENSIVE ELECTRODE IMPORTANCE ANALYSIS")
    print("=" * 80)
    print("Strategy: Individual analysis + voting across all participants and sessions")
    print(f"Analysis mode: {'Attention + Working Memory' if INCLUDE_0_BACK else 'Working Memory Load Only'}")
    print("=" * 80)
    
    # Get available participants
    participants = get_available_participants()
    print(f"\n📋 Found {len(participants)} participants: {participants}")
    print(f"📋 Session types: {SESSION_TYPES}")
    
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
    
    print(f"\n✅ COMPLETED: {completed_analyses}/{total_planned} analyses")
    
    # Convert to DataFrame
    all_results_df = pd.DataFrame(ALL_RESULTS)
    
    # Create comprehensive voting summary
    print(f"\n🗳️ ELECTRODE VOTING RESULTS (Most Important Across Sessions):")
    print("=" * 60)
    vote_ranking = sorted(ELECTRODE_VOTES.items(), key=lambda x: x[1], reverse=True)
    for rank, (electrode, votes) in enumerate(vote_ranking, 1):
        percentage = (votes / completed_analyses) * 100
        print(f"{rank}. {electrode}: {votes} votes ({percentage:.1f}%)")
    
    # Average ranking analysis
    print(f"\n📊 AVERAGE RANKING ANALYSIS:")
    print("=" * 60)
    avg_rankings = {}
    for electrode, ranks in ELECTRODE_RANKINGS.items():
        avg_rank = np.mean(ranks)
        std_rank = np.std(ranks)
        avg_rankings[electrode] = (avg_rank, std_rank)
    
    rank_sorted = sorted(avg_rankings.items(), key=lambda x: x[1][0])
    for rank, (electrode, (avg_rank, std_rank)) in enumerate(rank_sorted, 1):
        print(f"{rank}. {electrode}: avg rank {avg_rank:.1f} (±{std_rank:.1f})")
    
    # Save comprehensive results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    detailed_path = output_dir / "comprehensive_electrode_analysis_all_participants.csv"
    all_results_df.to_csv(detailed_path, index=False)
    print(f"\n💾 Detailed results saved to: {detailed_path}")
    
    # Save voting summary
    voting_summary = pd.DataFrame([
        {'electrode': electrode, 'votes': votes, 'percentage': (votes/completed_analyses)*100}
        for electrode, votes in vote_ranking
    ])
    voting_path = output_dir / "electrode_voting_summary.csv"
    voting_summary.to_csv(voting_path, index=False)
    print(f"💾 Voting summary saved to: {voting_path}")
    
    # Save ranking summary  
    ranking_summary = pd.DataFrame([
        {'electrode': electrode, 'avg_rank': avg_rank, 'std_rank': std_rank, 'consistency_score': 1/std_rank if std_rank > 0 else float('inf')}
        for electrode, (avg_rank, std_rank) in avg_rankings.items()
    ]).sort_values('avg_rank')
    ranking_path = output_dir / "electrode_ranking_summary.csv"
    ranking_summary.to_csv(ranking_path, index=False)
    print(f"💾 Ranking summary saved to: {ranking_path}")
    
    # Generate comprehensive visualization
    print(f"\n🎨 Generating comprehensive visualization...")
    create_comprehensive_visualization(all_results_df)
    
    # Final recommendations
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    print("=" * 60)
    if vote_ranking:
        top_electrode = vote_ranking[0]
        print(f"🏆 Most consistently important: {top_electrode[0]} ({top_electrode[1]} votes)")
        
        if len(rank_sorted) >= 3:
            top_3 = [electrode for electrode, _ in rank_sorted[:3]]
            print(f"📈 Top 3 by average ranking: {', '.join(top_3)}")
            
            # Find most consistent (lowest std in ranking)
            most_consistent = min(avg_rankings.items(), key=lambda x: x[1][1])
            print(f"🎯 Most consistent performer: {most_consistent[0]} (rank std: {most_consistent[1][1]:.2f})")
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"📊 Results: {completed_analyses} individual analyses → voting → visualization")
    
    return all_results_df, vote_ranking, avg_rankings

if __name__ == "__main__":
    results_df, votes, rankings = main()
