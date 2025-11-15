import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# --- Setup and Theming ---
sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": ".8", "axes.labelcolor": ".3", "xtick.color": ".3",
    "ytick.color": ".3", "text.color": ".3", "font.family": "sans-serif",
})
plots_dir = Path("plots/final_thesis_plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# --- Load Data ---
df_articles = pd.read_csv("data/1_articles_filtered_and_enriched.csv")

# --- THIS IS THE FIX ---
# Correctly define the palette and order to include ALL categories from your data
PALETTE = {
    'Left': '#0077b6', 'Left-Center': '#48cae4', 'Center': '#adb5bd', 
    'Least Biased': '#90be6d', 'Right-Center': '#f77f00', 'Right': '#d62828',
    'Pro-Science': '#55a630', 'Questionable': '#6a040f', 'Uncategorized': '#808080'
}
bias_order = [
    'Right', 'Right-Center', 'Center', 'Left-Center', 'Left', 
    'Least Biased', 'Pro-Science', 'Questionable'
]
FALLBACK_COLOR = '#333333'


# --- Generate Plot 1: Political Bias Distribution ---
print("1. Generating Corrected Political Bias Distribution...")
counts = df_articles['bias_label'].value_counts().reindex(bias_order)

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(counts.index, counts.values, color=[PALETTE.get(k, FALLBACK_COLOR) for k in counts.index], height=0.7)

ax.set_title('Full Distribution of News Articles by Political Bias', fontsize=20, pad=20, weight='bold')
ax.set_xlabel('Number of Articles', fontsize=14)
ax.set_ylabel('')
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
ax.bar_label(bars, fmt='{:,.0f}', padding=5, fontsize=12, color='.2')

ax.spines[['top', 'right', 'left']].set_visible(False)
ax.tick_params(axis='y', length=0, labelsize=12)
ax.tick_params(axis='x', labelsize=12)

plt.tight_layout()
plt.savefig(plots_dir / "1_corrected_bias_distribution.png", dpi=300)
plt.close()

print(f"\n--- Corrected plot saved to '{plots_dir}/1_corrected_bias_distribution.png' ---")