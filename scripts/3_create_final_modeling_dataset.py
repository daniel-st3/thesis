import pandas as pd
import numpy as np
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def create_final_dataset():
    """
    Assembles the final, definitive dataset for modeling.
    1. Starts with the corrected daily news features.
    2. Engineers and merges daily retail sentiment features.
    3. Engineers and merges stock market technical indicators.
    4. Creates lagged and rolling features.
    5. Saves the final dataset ready for modeling.
    """
    print("--- Step 3: Creating Final, Corrected Modeling Dataset ---")

    # --- Configuration ---
    NEWS_FEATURES_PATH = Path("data/2_daily_features_corrected.csv")
    RETAIL_DATA_PATH = Path("data/raw_reddit_data.csv")
    PRICE_DATA_PATH = Path("data/daily_prices.csv")
    OUTPUT_PATH = Path("data/3_final_modeling_dataset.csv")

    # --- 1. Loaded Corrected Daily News Features ---
    if not NEWS_FEATURES_PATH.exists():
        print(f"Error: Core features file not found at {NEWS_FEATURES_PATH}")
        return
    df_base = pd.read_csv(NEWS_FEATURES_PATH)
    df_base['date'] = pd.to_datetime(df_base['date'])
    print("Loaded corrected daily news features.")

    # --- 2. Engineered and Merge Retail Sentiment ---
    if not RETAIL_DATA_PATH.exists():
        print(f"Warning: Raw Reddit data not found at {RETAIL_DATA_PATH}. Skipping retail features.")
        df_merged = df_base
    else:
        print("Processing and merging retail sentiment features...")
        df_retail_raw = pd.read_csv(RETAIL_DATA_PATH)
        analyzer = SentimentIntensityAnalyzer()
        df_retail_raw['full_text'] = df_retail_raw['title'].astype(str) + " " + df_retail_raw['selftext'].fillna('').astype(str)
        df_retail_raw['vader_compound'] = df_retail_raw['full_text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])
        df_retail_raw['date'] = pd.to_datetime(df_retail_raw['created_utc']).dt.date
        df_retail_raw['date'] = pd.to_datetime(df_retail_raw['date'])
        
        daily_retail = df_retail_raw.groupby(['date', 'ticker_mentioned']).agg(
            retail_sentiment=('vader_compound', 'mean'),
            n_retail_posts=('vader_compound', 'count')
        ).reset_index().rename(columns={'ticker_mentioned': 'ticker'})
        
        df_merged = pd.merge(df_base, daily_retail, on=['date', 'ticker'], how='left')
        df_merged[['retail_sentiment', 'n_retail_posts']] = df_merged[['retail_sentiment', 'n_retail_posts']].fillna(0)
    
    # --- 3. Engineered and Merged Technical Indicators ---
    if not PRICE_DATA_PATH.exists():
        print(f"Error: Price data not found at {PRICE_DATA_PATH}. Cannot add technical indicators.")
        return
    
    print("Processing and merging stock price and technical indicators...")
    df_prices = pd.read_csv(PRICE_DATA_PATH)
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    
    # First, merged the prices into main dataframe
    df_final = pd.merge(df_merged, df_prices, on=['date', 'ticker'], how='inner') # Inner join to ensure we have prices for every row
    df_final = df_final.sort_values(by=['date', 'ticker'])

    # Now calculated technical indicators on the merged dataframe
    delta = df_final.groupby('ticker')['adj_close'].transform(lambda x: x.diff(1))
    gain, loss = delta.where(delta > 0, 0), -delta.where(delta < 0, 0)
    avg_gain = gain.groupby(df_final['ticker']).transform(lambda x: x.rolling(window=14).mean())
    avg_loss = loss.groupby(df_final['ticker']).transform(lambda x: x.rolling(window=14).mean())
    rs = avg_gain / avg_loss
    df_final['rsi'] = 100 - (100 / (1 + rs))
    
    exp1 = df_final.groupby('ticker')['adj_close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    exp2 = df_final.groupby('ticker')['adj_close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df_final['macd'] = exp1 - exp2
    df_final['macd_signal'] = df_final.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    
    # --- 4. Creatd Target Variable, Lags, and Rolling Features ---
    print("Creating target variable, lags, and rolling features...")
    df_final['target_up'] = (df_final.groupby('ticker')['adj_close'].shift(-1) > df_final['adj_close']).astype(int)

    feature_cols = [
        'avg_sentiment', 'avg_bias', 'disp_bias', 'n_articles',
        'retail_sentiment', 'n_retail_posts', 'rsi', 'macd', 'macd_signal'
    ]
    for col in feature_cols:
        if col in df_final.columns:
            df_final[f'{col}_roll3'] = df_final.groupby('ticker')[col].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            df_final[f'{col}_lag1'] = df_final.groupby('ticker')[col].transform(lambda x: x.shift(1))

    # Cleaned up NaNs created by a lack of future price, or by indicator/lag calculations
    df_final = df_final.dropna(subset=['target_up'])
    df_final = df_final.fillna(0)
    print("Final feature engineering complete.")

    # --- 5. Saved the Final Dataset ---
    df_final.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

    print(f"\n--- Success! ---")
    print(f"The final, definitive modeling dataset has been saved to: {OUTPUT_PATH}")
    print(f"This file contains {len(df_final.columns)} total columns (features + identifiers + target).")
    print("\nPreview of the final dataset:")
    print(df_final.head())


if __name__ == "__main__":
    create_final_dataset()