import pandas as pd
import numpy as np
from pathlib import Path # <-- FIX 1: Import the Path library

def engineer_daily_features():
    """
    Takes the filtered master article dataset and aggregates it to the
    ticker-day level, creating the core features for modeling.
    1. Correctly processes the date column.
    2. Applies a nuanced numerical mapping to the detailed bias labels.
    3. Calculates daily avg_sentiment, avg_bias, disp_bias, and n_articles.
    4. Saves the final, corrected daily features dataset.
    """
    print("--- Step 2: Engineering Corrected Daily Aggregated Features ---")

    # --- Configuration ---
    INPUT_PATH = Path("data/1_articles_filtered_and_enriched.csv")
    OUTPUT_PATH = Path("data/2_daily_features_corrected.csv")

    # --- 1. Load the Master Article Data ---
    if not INPUT_PATH.exists():
        print(f"Error: Master article data file not found at {INPUT_PATH}")
        print("Please run '1_build_master_articles.py' first.")
        return
        
    df = pd.read_csv(INPUT_PATH)
    print("Loaded filtered and enriched article data.")

    # Converts 'published_date' string to a proper date column
    df['date'] = pd.to_datetime(df['published_date'], errors='coerce', utc=True).dt.date
    
    original_rows = len(df)
    df.dropna(subset=['date'], inplace=True)
    if len(df) < original_rows:
        print(f"Dropped {original_rows - len(df)} rows due to invalid dates.")
    print("Processed and cleaned date column.")

    # --- 2. Apply Nuanced Bias Score Mapping ---
    bias_score_map = {
        'Left': -2.0,
        'Left-Center': -1.0,
        'Center': 0.0,
        'Least Biased': 0.0,
        'Pro-Science': 0.0,
        'Right-Center': 1.0,
        'Right': 2.0
    }
    df['bias_score'] = df['bias_label'].map(bias_score_map)
    print("Applied new 5-point numerical bias mapping.")

    # --- 3. Aggregate to Daily Features ---
    print("Aggregating features to the ticker-day level...")
    
    daily_features = df.groupby(['date', 'ticker']).agg(
        avg_bias=('bias_score', 'mean'),
        disp_bias=('bias_score', 'std'),
        avg_sentiment=('sentiment', 'mean'),
        n_articles=('title', 'count')
    ).reset_index()

    daily_features['disp_bias'] = daily_features['disp_bias'].fillna(0)
    daily_features['avg_bias'] = daily_features['avg_bias'].fillna(0)
    print("Re-calculated daily aggregated features.")
    
    # --- 4. Save the Output ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily_features.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

    print(f"\n--- Success! ---")
    print(f"Daily features have been engineered and saved to: {OUTPUT_PATH}")
    print("\nPreview of the corrected daily data:")
    print(daily_features.head())

if __name__ == "__main__":
    engineer_daily_features()