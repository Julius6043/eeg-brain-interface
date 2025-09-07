# --- Demo: EEG Electrode Importance Plot ---
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Create sample data based on your previous results
sample_data = {
    'excluded_electrode': ['None', 'EEG7', 'EEG8', 'EEG6', 'EEG3', 'EEG5', 'EEG1', 'EEG4', 'EEG2'],
    'accuracy_mean': [0.751, 0.750, 0.751, 0.745, 0.740, 0.735, 0.725, 0.688, 0.686],
    'accuracy_std': [0.082, 0.085, 0.083, 0.090, 0.088, 0.092, 0.095, 0.098, 0.105],
    'accuracy_drop': [0.000, 0.001, 0.000, 0.006, 0.011, 0.016, 0.026, 0.063, 0.065]
}

df = pd.DataFrame(sample_data)

# Filter out baseline and sort by accuracy drop
loo_results = df[df['excluded_electrode'] != 'None'].copy()
loo_results = loo_results.sort_values('accuracy_drop', ascending=True)
baseline_accuracy = df[df['excluded_electrode'] == 'None']['accuracy_mean'].iloc[0]

# Set up the plot style
plt.style.use('default')
sns.set_palette("husl")

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('EEG Electrode Importance Analysis: Julian (Outdoor)', 
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
    ax1.text(value + 0.002, bar.get_y() + bar.get_height()/2, 
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
baseline_std = df[df['excluded_electrode'] == 'None']['accuracy_std'].iloc[0]
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
most_important = loo_results.iloc[-1]  # Last in ascending order = highest drop
least_important = loo_results.iloc[0]   # First in ascending order = lowest drop

summary_data = [
    ['Metric', 'Value'],
    ['Baseline Accuracy', f"{baseline_accuracy:.3f} ± {baseline_std:.3f}"],
    ['Most Important Electrode', f"{most_important['excluded_electrode']} (drop: {most_important['accuracy_drop']:.3f})"],
    ['Least Important Electrode', f"{least_important['excluded_electrode']} (drop: {least_important['accuracy_drop']:.3f})"],
    ['Significant Electrodes', f"{sum(is_significant)} / {len(loo_results)}"],
    ['Mean Accuracy Drop', f"{loo_results['accuracy_drop'].mean():.3f}"],
    ['Max Accuracy Drop', f"{loo_results['accuracy_drop'].max():.3f}"]
]

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
plot_path = output_dir / "electrode_importance_plot_demo.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Demo plot saved to: {plot_path}")

# Show the plot
plt.show()

print("✅ Demo electrode importance visualization created!")
print("\n📊 This plot will show:")
print("• Top-left: Electrode ranking by importance (accuracy drop)")
print("• Top-right: Baseline vs reduced accuracy comparison") 
print("• Bottom-left: Statistical significance of each electrode")
print("• Bottom-right: Summary statistics table")
