import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def create_specialized_daily_features():
    """
    Creates a highly detailed daily feature set by first calculating
    metrics for different bias groupings (Left, Right, Center).
    """
    print("--- Engineering Specialized Daily Features ---")
    
    articles_path = Path("data/1_articles_filtered_and_enriched.csv")
    if not articles_path.exists():
        raise FileNotFoundError(f"Required file not found: {articles_path}")
    df = pd.read_csv(articles_path)
    df['date'] = pd.to_datetime(df['published_date'], errors='coerce').dt.date
    df.dropna(subset=['date'], inplace=True)

    bias_score_map = {
        'Left': -2.0, 'Left-Center': -1.0, 'Center': 0.0,
        'Least Biased': 0.0, 'Pro-Science': 0.0, 'Right-Center': 1.0, 'Right': 2.0
    }
    df['bias_score'] = df['bias_label'].map(bias_score_map)

    df['bias_group'] = 'Other'
    df.loc[df['bias_label'].isin(['Left', 'Left-Center']), 'bias_group'] = 'Left'
    df.loc[df['bias_label'].isin(['Right', 'Right-Center']), 'bias_group'] = 'Right'

    grouped = df.groupby(['date', 'ticker', 'bias_group'])
    agg_by_bias_group = grouped['sentiment'].agg(['mean', 'count']).unstack(level='bias_group', fill_value=0)
    agg_by_bias_group.columns = [f'{stat}_{group}' for stat, group in agg_by_bias_group.columns]
    
    overall_agg = df.groupby(['date', 'ticker']).agg(
        avg_bias=('bias_score', 'mean'),
        disp_bias=('bias_score', 'std'),
        n_articles=('title', 'count')
    ).reset_index()
    
    daily_features = pd.merge(overall_agg, agg_by_bias_group.reset_index(), on=['date', 'ticker'], how='outer')
    daily_features = daily_features.fillna(0)
    
    print("Specialized daily features created successfully.")
    return daily_features

def run_advanced_comparison():
    """
    Runs the full, advanced model comparison using specialized features.
    """
    print("--- Starting Advanced Model Comparison ---")
    
    daily_features = create_specialized_daily_features()
    daily_features['date'] = pd.to_datetime(daily_features['date'])

    df_prices = pd.read_csv(Path("data/daily_prices.csv"))
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    df_retail_raw = pd.read_csv(Path("data/raw_reddit_data.csv"))
    analyzer = SentimentIntensityAnalyzer()
    df_retail_raw['sentiment'] = df_retail_raw['title'].fillna('').apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df_retail_raw['date'] = pd.to_datetime(df_retail_raw['created_utc']).dt.date
    df_retail_raw['date'] = pd.to_datetime(df_retail_raw['date'])
    df_retail = df_retail_raw.groupby(['date', 'ticker_mentioned']).agg(retail_sentiment=('sentiment', 'mean')).reset_index().rename(columns={'ticker_mentioned': 'ticker'})

    df_final = pd.merge(daily_features, df_prices, on=['date', 'ticker'], how='inner')
    df_final = pd.merge(df_final, df_retail, on=['date', 'ticker'], how='left').fillna(0)
    df_final = df_final.sort_values(by='date')

    delta = df_final.groupby('ticker')['adj_close'].transform(lambda x: x.diff(1))
    gain, loss = delta.where(delta > 0, 0), -delta.where(delta < 0, 0)
    avg_gain = gain.groupby(df_final['ticker']).transform(lambda x: x.rolling(window=14).mean())
    avg_loss = loss.groupby(df_final['ticker']).transform(lambda x: x.rolling(window=14).mean())
    df_final['rsi'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    df_final = df_final.fillna(0)
    
    df_final['target_up'] = (df_final.groupby('ticker')['adj_close'].shift(-1) > df_final['adj_close']).astype(int)
    df_final = df_final.dropna(subset=['target_up'])

    
    # Defines the base features first, then adds to them without creating duplicates.
    base_features = ['n_articles', 'rsi', 'retail_sentiment']
    
    # The 'Baseline' model in this context will be the most basic, without any specialized sentiment
    baseline_features = base_features

    # Left-Bias model adds features derived from left-leaning sources
    left_bias_features = base_features + ['mean_Left', 'count_Left']

    # Right-Bias model adds features from right-leaning sources
    right_bias_features = base_features + ['mean_Right', 'count_Right']

    # Polarization model adds the overall bias dispersion
    polarization_features = base_features + ['disp_bias']

    # All-Bias model includes all engineered bias features
    all_features = base_features + ['avg_bias', 'disp_bias', 'mean_Left', 'count_Left', 'mean_Right', 'count_Right', 'mean_Other', 'count_Other']
    
    # --- Splits Data and Evaluate Models ---
    y = df_final['target_up']
    split_index = int(len(df_final) * 0.8)
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    results = {}
    model_configs = {
        "Baseline": baseline_features,
        "Left-Bias Enhanced": left_bias_features,
        "Right-Bias Enhanced": right_bias_features,
        "Polarization Enhanced": polarization_features,
        "All-Bias Enhanced": all_features,
    }

    for name, features in model_configs.items():
        print(f"\n--- Training and Evaluating: {name} ---")
        X = df_final[features]
        if not all(feature in X.columns for feature in features):
            print(f"  Skipping model '{name}' due to missing feature columns.")
            continue
            
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        
        model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, random_state=42)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, (y_pred_proba > 0.5).astype(int))
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        results[name] = {'Accuracy': accuracy, 'ROC_AUC': roc_auc}
        print(f"  Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

    print("\n\n--- Final Advanced Model Comparison Summary ---")
    if results:
        summary_df = pd.DataFrame(results).T
        print(summary_df.sort_values(by='ROC_AUC', ascending=False))
    else:
        print("No models were trained.")
    print("---------------------------------------------")

if __name__ == "__main__":
    run_advanced_comparison()