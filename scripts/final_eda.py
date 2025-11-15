import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

def create_all_thesis_plots():
    """
    Generates the complete suite of final, thesis-quality visualizations,
    including EDA and the definitive model performance radar chart.
    """
    print("--- Generating All Final Thesis Plots ---")

    # --- 1. Setup and Theming ---
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": ".8", "axes.labelcolor": ".3", "xtick.color": ".3",
        "ytick.color": ".3", "text.color": ".3", "font.family": "sans-serif",
    })
    PALETTE = {
        'Right': '#d62828', 'Right-Center': '#f77f00', 'Center': '#adb5bd',
        'Left-Center': '#48cae4', 'Left': '#0077b6'
    }
    FALLBACK_COLOR = '#808080'
    plots_dir = Path("plots/final_thesis_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"All plots will be saved to: {plots_dir}")

    # --- 2. Load and Prepare Data ---
    try:
        df_articles = pd.read_csv("data/1_articles_filtered_and_enriched.csv")
        df_daily = pd.read_csv("data/2_daily_features_corrected.csv")
        df_model = pd.read_csv("data/3_final_modeling_dataset.csv")
    except FileNotFoundError as e:
        print(f"Error: Missing data file - {e}. Please ensure all previous scripts have been run.")
        return

    df_articles['date'] = pd.to_datetime(df_articles['published_date'], errors='coerce').dt.date
    df_articles.dropna(subset=['date'], inplace=True)
    df_articles['date'] = pd.to_datetime(df_articles['date'])
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_model['date'] = pd.to_datetime(df_model['date'])
    df_model.set_index('date', inplace=True)

    # --- EDA Plots 1-6 (Unchanged) ---
    print("1. Generating Political Bias Distribution...")
    bias_order = ['Right', 'Right-Center', 'Center', 'Left-Center', 'Left']
    counts = df_articles['bias_label'].value_counts().reindex(bias_order)
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(counts.index, counts.values, color=[PALETTE.get(k, FALLBACK_COLOR) for k in counts.index], height=0.6)
    ax.set_title('Distribution of News Articles by Political Bias', fontsize=18, pad=15, weight='bold')
    ax.set_xlabel('Number of Articles', fontsize=12)
    ax.set_ylabel('Political Bias Category', fontsize=12)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    ax.bar_label(bars, fmt='{:,.0f}', padding=5, fontsize=11, color='.2')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    plt.tight_layout(); plt.savefig(plots_dir / "1_bias_distribution.png", dpi=300); plt.close()

    print("2. Generating Sentiment by Bias Spectrum...")
    df_plot_s = df_articles[df_articles['bias_label'].isin(PALETTE.keys())].copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(x='sentiment', y='bias_label', data=df_plot_s, ax=ax, palette=PALETTE, order=bias_order, hue='bias_label', legend=False, width=0.7, fliersize=2)
    ax.axvline(0, color='black', linestyle='--', linewidth=1); ax.set_title('Article Sentiment Score Distribution by Political Bias', fontsize=18, pad=15, weight='bold')
    ax.set_xlabel('Article Sentiment Score (VADER)', fontsize=12); ax.set_ylabel('Political Bias Category', fontsize=12)
    ax.spines[['top', 'right', 'left']].set_visible(False); ax.tick_params(axis='y', length=0)
    plt.tight_layout(); plt.savefig(plots_dir / "2_sentiment_by_bias.png", dpi=300); plt.close()

    print("3. Generating Weekly News Volume by Bias...")
    df_weekly_counts = df_articles.set_index('date').groupby([pd.Grouper(freq='W-MON'), 'bias_label']).size().unstack(fill_value=0).reindex(columns=bias_order)
    fig, ax = plt.subplots(figsize=(16, 8))
    df_weekly_counts.plot(kind='area', stacked=True, ax=ax, color=[PALETTE[k] for k in bias_order], alpha=0.8)
    ax.set_title('Weekly Volume of News Articles by Political Bias Category', fontsize=18, pad=15, weight='bold')
    ax.set_xlabel('Date', fontsize=12); ax.set_ylabel('Number of Articles per Week', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False); ax.legend(title='Bias Category', loc='upper left')
    plt.tight_layout(); plt.savefig(plots_dir / "3_weekly_volume_by_bias.png", dpi=300); plt.close()

    print("4. Generating Feature Correlation Heatmap...")
    corr_features = ['avg_sentiment', 'avg_bias', 'disp_bias', 'n_articles', 'retail_sentiment', 'n_retail_posts', 'rsi', 'macd', 'target_up']
    corr_df = df_model[corr_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='vlag', center=0, fmt='.2f', linewidths=.5, ax=ax, annot_kws={"size": 11})
    ax.set_title('Correlation Matrix of Key Daily Features', fontsize=18, pad=20, weight='bold')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(plots_dir / "4_feature_correlation_heatmap.png", dpi=300); plt.close()
    
    print("5. Generating Daily Bias vs. Sentiment Time Series...")
    df_daily_agg = df_daily.groupby('date')[['avg_bias', 'avg_sentiment']].mean().rolling(window=14).mean()
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.set_title('14-Day Rolling Average of News Bias and Sentiment', fontsize=18, pad=15, weight='bold')
    ax1.plot(df_daily_agg.index, df_daily_agg['avg_bias'], color=PALETTE['Right'], label='Average Bias')
    ax1.set_xlabel('Date', fontsize=12); ax1.set_ylabel('Average Political Bias (Right > 0)', fontsize=12, color=PALETTE['Right'])
    ax1.tick_params(axis='y', labelcolor=PALETTE['Right']); ax1.spines['top'].set_visible(False)
    ax2 = ax1.twinx()
    ax2.plot(df_daily_agg.index, df_daily_agg['avg_sentiment'], color=PALETTE['Left'], linestyle='--', label='Average Sentiment')
    ax2.set_ylabel('Average Article Sentiment', fontsize=12, color=PALETTE['Left'])
    ax2.tick_params(axis='y', labelcolor=PALETTE['Left']); ax2.spines['top'].set_visible(False)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9)); plt.tight_layout()
    plt.savefig(plots_dir / "6_daily_bias_vs_sentiment.png", dpi=300); plt.close()

    print("6. Generating Bias Dispersion vs. News Volume...")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.regplot(x='n_articles', y='disp_bias', data=df_daily, ax=ax, scatter_kws={'alpha':0.3, 'color': '#0077b6'}, line_kws={'color':'#d62828', 'linestyle':'--'})
    ax.set_title('Political Bias Dispersion Increases with News Volume', fontsize=18, pad=15, weight='bold')
    ax.set_xlabel('Number of Articles in a Day', fontsize=12); ax.set_ylabel('Dispersion of Political Bias (Polarization)', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(plots_dir / "7_dispersion_vs_volume.png", dpi=300); plt.close()

    # --- Plot 7: Definitive Model Performance Radar Chart ---
    print("7. Generating Definitive Model Performance Radar Chart...")
    
    # --- THIS IS YOUR CORRECT, WORKING CODE ---
    df_radar = pd.read_csv('data/model_comparison_results.csv').rename(
            columns={'Unnamed: 0':'model', 'Unnamed: 1':'variant'})

    metrics  = ['accuracy', 'f1', 'auc']
    palette_radar = {'baseline': '#B0B0B0', 'enhanced': '#007ACC'}
    models   = df_radar['model'].unique()
    angles   = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles  += angles[:1]

    fig, axs = plt.subplots(1, len(models), subplot_kw={'polar':True}, figsize=(18, 5))

    if len(models) == 1:
        axs = [axs]

    for ax, model in zip(axs, models):
        sub = df_radar[df_radar['model']==model]

        base = sub[sub['variant']=='baseline'][metrics].values.flatten().tolist()
        enh  = sub[sub['variant']=='enhanced'][metrics].values.flatten().tolist()
        base += base[:1]
        enh  += enh[:1]

        ax.plot(angles, base, color=palette_radar['baseline'], lw=2, label='Baseline')
        ax.fill(angles, base, color=palette_radar['baseline'], alpha=0.2)

        ax.plot(angles, enh,  color=palette_radar['enhanced'], lw=2, label='Enhanced')
        ax.fill(angles, enh,  color=palette_radar['enhanced'], alpha=0.2)

        ax.set_title(model, size=16, pad=25, weight='bold')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics], size=12)
        ax.set_ylim(0.45, 0.60)
        ax.set_yticks([0.45, 0.50, 0.55, 0.60])
        ax.set_yticklabels(["0.45", "0.50", "0.55", "0.60"], color="grey", size=10)

    axs[0].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fontsize=12)
    fig.suptitle("Model Performance: Baseline vs. Enhanced Features",
                 fontsize=22, y=1.15, weight='bold')
    
    plt.tight_layout(pad=3.0)
    plt.savefig(plots_dir / "8_final_model_performance_radar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n--- All definitive plots have been generated successfully. ---")
    print(f"--- Please check the '{plots_dir}' directory. ---")

if __name__ == "__main__":
    create_all_thesis_plots()