# --- Electrode Subset Comparison Script ---
"""
This script demonstrates how to use electrode_importance_analysis.py results
to run targeted comparisons with random_forestJW.py

Usage:
1. First run electrode_importance_analysis.py to identify important electrodes
2. Then use this script to compare different electrode subsets
"""

import pandas as pd
import subprocess
import sys
from pathlib import Path

# Configuration
PARTICIPANT = "julian"
SESSION_TYPE = "outdoor"

def run_random_forest_with_electrodes(include_channels=None, exclude_channels=None, description=""):
    """Run random_forestJW.py with specific electrode configuration"""
    
    print(f"\n{'='*50}")
    print(f"RUNNING: {description}")
    print(f"Include: {include_channels}")
    print(f"Exclude: {exclude_channels}")
    print(f"{'='*50}")
    
    # Create a temporary modified version of the script
    # (In practice, you'd modify the configuration variables)
    # For now, we'll just print what would be run
    
    if include_channels:
        print(f"Would set INCLUDE_CHANNELS = {include_channels}")
    if exclude_channels:
        print(f"Would set EXCLUDE_CHANNELS = {exclude_channels}")
    
    # Here you would actually run the script with the modified configuration
    # subprocess.run([sys.executable, "random_forestJW.py"])
    
    return {
        'description': description,
        'include_channels': include_channels,
        'exclude_channels': exclude_channels,
        'accuracy': 0.75,  # Placeholder - would be actual result
        'precision': 0.73,  # Placeholder
        'recall': 0.74,     # Placeholder
        'f1': 0.73         # Placeholder
    }

def load_electrode_importance_results():
    """Load results from electrode importance analysis"""
    results_path = Path("results") / f"electrode_importance_summary_{PARTICIPANT}_{SESSION_TYPE}.csv"
    
    if not results_path.exists():
        print(f"❌ Electrode importance results not found: {results_path}")
        print("Please run electrode_importance_analysis.py first!")
        return None
    
    df = pd.read_csv(results_path)
    return df

def main():
    print("=== ELECTRODE SUBSET COMPARISON ANALYSIS ===")
    
    # 1. Load electrode importance results
    print("\n1. Loading electrode importance results...")
    importance_df = load_electrode_importance_results()
    
    if importance_df is None:
        return
    
    print("Electrode importance ranking:")
    print(importance_df[['excluded_electrode', 'accuracy_drop']].head(8))
    
    # 2. Extract electrode rankings
    loo_results = importance_df[importance_df['excluded_electrode'] != 'None'].copy()
    loo_results = loo_results.sort_values('accuracy_drop', ascending=False)
    
    # Most important electrodes (highest accuracy drop when removed)
    most_important = loo_results['excluded_electrode'].head(3).tolist()
    least_important = loo_results['excluded_electrode'].tail(2).tolist()
    
    print(f"\nMost important electrodes: {most_important}")
    print(f"Least important electrodes: {least_important}")
    
    # 3. Define comparison scenarios
    scenarios = [
        {
            'include_channels': None,
            'exclude_channels': None,
            'description': "Baseline: All 8 electrodes"
        },
        {
            'include_channels': most_important,
            'exclude_channels': None,
            'description': f"Top 3 most important: {most_important}"
        },
        {
            'include_channels': most_important[:2],
            'exclude_channels': None,
            'description': f"Top 2 most important: {most_important[:2]}"
        },
        {
            'include_channels': None,
            'exclude_channels': least_important,
            'description': f"All except least important: exclude {least_important}"
        },
        {
            'include_channels': None,
            'exclude_channels': least_important + [loo_results['excluded_electrode'].iloc[-3]],
            'description': f"All except 3 least important"
        }
    ]
    
    # 4. Run comparisons
    print(f"\n2. Running Random Forest comparisons...")
    results = []
    
    for scenario in scenarios:
        result = run_random_forest_with_electrodes(
            include_channels=scenario['include_channels'],
            exclude_channels=scenario['exclude_channels'],
            description=scenario['description']
        )
        results.append(result)
    
    # 5. Create comparison table
    print(f"\n3. Results Summary...")
    comparison_df = pd.DataFrame(results)
    
    print("\n=== PERFORMANCE COMPARISON ===")
    print(comparison_df[['description', 'accuracy', 'precision', 'recall', 'f1']].to_string(index=False))
    
    # 6. Save comparison results
    output_path = Path("results") / f"electrode_subset_comparison_{PARTICIPANT}_{SESSION_TYPE}.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"\nComparison saved to: {output_path}")
    
    # 7. Analysis insights
    print(f"\n=== ANALYSIS INSIGHTS ===")
    print("💡 Use this comparison to determine:")
    print("   - How many electrodes you actually need")
    print("   - Which specific electrodes to prioritize")
    print("   - Trade-offs between electrode count and performance")
    print("   - Optimal electrode configurations for your hardware constraints")
    
    print(f"\n🔧 TO ACTUALLY RUN THE ANALYSIS:")
    print("   1. Modify the INCLUDE_CHANNELS/EXCLUDE_CHANNELS in random_forestJW.py")
    print("   2. Run the script for each configuration")
    print("   3. Compare the actual accuracy/precision/recall results")

if __name__ == "__main__":
    main()
