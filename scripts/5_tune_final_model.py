import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

def tune_final_model():
    """
    Performs hyperparameter tuning on the final Enhanced Model to find the
    optimal settings and evaluates its performance.
    """
    print("--- Step 5: Hyperparameter Tuning for the Final Enhanced Model ---")

    # --- 1. Load the Final Dataset ---
    DATASET_PATH = Path("data/3_final_modeling_dataset.csv")
    if not DATASET_PATH.exists():
        print(f"Error: Final modeling dataset not found at {DATASET_PATH}")
        return
        
    df = pd.read_csv(DATASET_PATH)
    df = df.sort_values(by='date')
    print("Loaded final modeling dataset.")

    # --- 2. Define Feature Set and Target ---
    enhanced_features = [
        'avg_sentiment', 'avg_bias', 'disp_bias', 'n_articles',
        'retail_sentiment', 'n_retail_posts', 'rsi', 'macd', 'macd_signal',
        'avg_sentiment_roll3', 'avg_bias_roll3', 'disp_bias_roll3', 'n_articles_roll3',
        'retail_sentiment_roll3', 'n_retail_posts_roll3', 'rsi_roll3', 'macd_roll3', 'macd_signal_roll3',
        'avg_sentiment_lag1', 'avg_bias_lag1', 'disp_bias_lag1', 'n_articles_lag1',
        'retail_sentiment_lag1', 'n_retail_posts_lag1', 'rsi_lag1', 'macd_lag1', 'macd_signal_lag1'
    ]
    y = df['target_up']
    X = df[enhanced_features]

    # --- 3. Time-based Split ---
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

    # --- 4. Define Hyperparameter Search Space ---
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 7],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    # --- 5. Set up Randomized Search with Time-Series Cross-Validation ---
    print("\nSetting up Randomized Search for hyperparameter tuning...")
    # TimeSeriesSplit is crucial to respect the temporal order of the data during validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=50,  # Number of different parameter combinations to try
        scoring='roc_auc', # We'll optimize for ROC AUC score
        cv=tscv,
        n_jobs=-1, # Use all available CPU cores
        verbose=1,
        random_state=42
    )

    # --- 6. Run the Tuning Process ---
    print("Starting the tuning process... This may take several minutes.")
    random_search.fit(X_train, y_train)

    print("\nTuning process finished.")
    print(f"\nBest parameters found: {random_search.best_params_}")
    print(f"Best cross-validation ROC AUC score: {random_search.best_score_:.4f}")

    # --- 7. Evaluate the Best Model on the Unseen Test Set ---
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, (y_pred_proba > 0.5).astype(int))
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\n--- Tuned Model Performance on Test Set ---")
    print(f"  Accuracy:      {accuracy:.4f}")
    print(f"  ROC AUC Score: {roc_auc:.4f}")
    print("-------------------------------------------")

if __name__ == "__main__":
    tune_final_model()