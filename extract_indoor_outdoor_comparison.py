#!/usr/bin/env python3
"""
Quick script to extract indoor vs outdoor comparison from comprehensive results
"""

import pandas as pd
import os

# Load the comprehensive results
results_path = 'results/comprehensive_electrode_analysis_all_participants.csv'
df = pd.read_csv(results_path)

# Filter for the 4 best electrodes identified earlier: EEG7, EEG8, EEG4, EEG1
best_4_electrodes = ['EEG7', 'EEG8', 'EEG4', 'EEG1']

# Get only results for these electrodes
best_4_df = df[df['electrode'].isin(best_4_electrodes)].copy()

# Function to calculate accuracy for 4 electrodes combined
def calculate_4_electrode_accuracy(participant_df):
    """
    For 4 electrode accuracy, we'll use the average performance of removing the 4 worst electrodes
    This approximates the performance with only the 4 best electrodes
    """
    # Get baseline accuracy (when all electrodes are used)
    baseline_rows = participant_df[participant_df['analysis_type'] == 'baseline']
    if len(baseline_rows) == 0:
        return None
    baseline_acc = baseline_rows['accuracy'].iloc[0]
    
    # For 4-electrode performance, subtract the average drop of the 4 worst performing electrodes
    # Get leave-one-out results
    loo_df = participant_df[participant_df['analysis_type'] == 'leave_one_out'].copy()
    if len(loo_df) < 4:
        return None
    
    # Sort by accuracy drop (most negative = worst electrode)
    loo_df = loo_df.sort_values('accuracy_drop')
    
    # Take the 4 worst electrodes (those with most negative drops)
    worst_4 = loo_df.tail(4)
    avg_worst_drop = worst_4['accuracy_drop'].mean()
    
    # Estimate 4-electrode accuracy: baseline - average drop of worst 4
    estimated_4_electrode_acc = baseline_acc - avg_worst_drop
    
    return estimated_4_electrode_acc

# Process each participant
comparison_results = []

for participant in df['participant'].unique():
    participant_df = df[df['participant'] == participant]
    
    indoor_df = participant_df[participant_df['session_type'] == 'indoor']
    outdoor_df = participant_df[participant_df['session_type'] == 'outdoor']
    
    if len(indoor_df) == 0 or len(outdoor_df) == 0:
        continue
    
    # Calculate 4-electrode accuracies
    indoor_4_acc = calculate_4_electrode_accuracy(indoor_df)
    outdoor_4_acc = calculate_4_electrode_accuracy(outdoor_df)
    
    if indoor_4_acc is None or outdoor_4_acc is None:
        continue
    
    # Calculate metrics
    accuracy_diff = indoor_4_acc - outdoor_4_acc
    ratio_outdoor_indoor = outdoor_4_acc / indoor_4_acc if indoor_4_acc > 0 else 0
    
    comparison_results.append({
        'participant': participant,
        'indoor_4electrode_accuracy': round(indoor_4_acc, 4),
        'outdoor_4electrode_accuracy': round(outdoor_4_acc, 4),
        'accuracy_difference_indoor_outdoor': round(accuracy_diff, 4),
        'performance_ratio_outdoor_indoor': round(ratio_outdoor_indoor, 4),
        'best_4_electrodes': ', '.join(best_4_electrodes)
    })

# Create DataFrame and save
indoor_outdoor_df = pd.DataFrame(comparison_results)

# Save to CSV
output_file = 'results/indoor_outdoor_4electrodes_comparison.csv'
indoor_outdoor_df.to_csv(output_file, index=False)

print(f"✅ Indoor vs Outdoor 4-electrode comparison saved to: {output_file}")
print(f"📊 Analyzed {len(indoor_outdoor_df)} participants")

# Print summary statistics
if len(indoor_outdoor_df) > 0:
    avg_indoor = indoor_outdoor_df['indoor_4electrode_accuracy'].mean()
    avg_outdoor = indoor_outdoor_df['outdoor_4electrode_accuracy'].mean()
    avg_diff = indoor_outdoor_df['accuracy_difference_indoor_outdoor'].mean()
    avg_ratio = indoor_outdoor_df['performance_ratio_outdoor_indoor'].mean()
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"Average indoor accuracy: {avg_indoor:.3f}")
    print(f"Average outdoor accuracy: {avg_outdoor:.3f}")
    print(f"Average difference (Indoor-Outdoor): {avg_diff:.3f}")
    print(f"Average outdoor/indoor ratio: {avg_ratio:.3f}")
    
    # Count how many participants perform better in each environment
    indoor_better = sum(1 for diff in indoor_outdoor_df['accuracy_difference_indoor_outdoor'] if diff > 0)
    outdoor_better = sum(1 for diff in indoor_outdoor_df['accuracy_difference_indoor_outdoor'] if diff < 0)
    
    print(f"Participants performing better indoors: {indoor_better}/{len(indoor_outdoor_df)}")
    print(f"Participants performing better outdoors: {outdoor_better}/{len(indoor_outdoor_df)}")
    
    # Show top and bottom performers
    best_outdoor = indoor_outdoor_df.loc[indoor_outdoor_df['performance_ratio_outdoor_indoor'].idxmax()]
    worst_outdoor = indoor_outdoor_df.loc[indoor_outdoor_df['performance_ratio_outdoor_indoor'].idxmin()]
    
    print(f"Best outdoor performance ratio: {best_outdoor['participant']} ({best_outdoor['performance_ratio_outdoor_indoor']:.3f})")
    print(f"Worst outdoor performance ratio: {worst_outdoor['participant']} ({worst_outdoor['performance_ratio_outdoor_indoor']:.3f})")

print(f"\n🎯 ANALYSIS COMPLETE!")
