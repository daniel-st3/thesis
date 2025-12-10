"""Train a logistic regression on the enhanced features and plot coefficients.

This script mirrors the feature preparation used in `4_run_final_model_comparison.py`
but fits on the full dataset for interpretability. It outputs a coefficient table
and a horizontal bar plot under `plots/final_thesis_plots/9_feature_importance.png`.
"""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


OUTPUT_PATH = pathlib.Path("plots/final_thesis_plots/9_feature_importance.png")


def plot_feature_importance():
    try:
        df = pd.read_csv("data/3_final_modeling_dataset.csv", index_col="date", parse_dates=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Final modeling dataset not found. Please run scripts/3_create_final_modeling_dataset.py first."
        )

    df = df.dropna()

    features = ['rsi', 'macd', 'macd_signal', 'retail_sentiment', 'avg_sentiment', 'avg_bias', 'disp_bias']
    X = df[features]
    y = df['target_up']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_scaled, y)

    coef_df = (
        pd.DataFrame({'Feature': features, 'Coefficient': model.coef_[0]})
        .sort_values(by='Coefficient', key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='vlag')
    plt.title('Feature Importance (Logistic Regression Coefficients)')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)

    print("Saved plot to", OUTPUT_PATH)
    print("\nCoefficient table:\n", coef_df.to_string(index=False))


if __name__ == "__main__":
    plot_feature_importance()
