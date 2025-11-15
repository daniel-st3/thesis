import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def run_model_comparison(X, y, model_name, model):
    """
    Runs time-series cross-validation for a given model.
    Handles class imbalance using SMOTE within each fold.
    """
    print(f"--- Running {model_name} ---")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Defined the pipeline with a scaler and the model
        # SMOTE is applied only to the training data in each fold to prevent data leakage
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        # Fitted the pipeline on the training data
        pipeline.fit(X_train, y_train)
        
        # Made predictions on the test data
        X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)
        y_pred = pipeline.named_steps['classifier'].predict(X_test_scaled)
        y_pred_proba = pipeline.named_steps['classifier'].predict_proba(X_test_scaled)[:, 1]

        # Calculated scores
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        scores['auc'].append(roc_auc_score(y_test, y_pred_proba))
        
    # Returned the average scores across all folds
    return {k: np.mean(v) for k, v in scores.items()}


def main():
    """
    Main function to load data, define models and features, and run the comparison.
    """
    print("Loading final modeling dataset...")
    try:
        df = pd.read_csv('data/3_final_modeling_dataset.csv', index_col='date', parse_dates=True)
    except FileNotFoundError:
        print("Final modeling dataset not found. Please run script 3 first.")
        return
    
    df.dropna(inplace=True)
    
    # These names now EXACTLY match the columns in your CSV file.
    technical_indicators = ['rsi', 'macd', 'macd_signal'] # 'macd_hist' is not in the file
    sentiment_features = ['retail_sentiment', 'avg_sentiment'] # Corrected names
    bias_features = ['avg_bias', 'disp_bias']

    # Defined the feature sets based on the corrected names
    baseline_feature_set = technical_indicators + ['avg_sentiment']
    enhanced_feature_set = technical_indicators + sentiment_features + bias_features
    
    # Defined features and target variable
    X_baseline = df[baseline_feature_set]
    X_enhanced = df[enhanced_feature_set]
    y = df['target_up'] # Corrected target variable name

    # Defined models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    all_results = {}
    
    # Runned comparison for all models
    for name, model in models.items():
        print(f"\n===== Evaluating Model: {name} =====")
        
        # Runned Baseline Model
        baseline_results = run_model_comparison(X_baseline, y, f"{name} (Baseline)", model)
        print(f"Average Baseline Scores: {baseline_results}")
        
        # Runned Enhanced Model
        enhanced_results = run_model_comparison(X_enhanced, y, f"{name} (Enhanced)", model)
        print(f"Average Enhanced Scores: {enhanced_results}")
        
        all_results[name] = {'baseline': baseline_results, 'enhanced': enhanced_results}

    # Saved the results
    with open('data/model_comparison_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
        
    print("\n\nComparison complete. Results saved to data/model_comparison_results.pkl")
    
    # For easy viewing, also saved as CSV
    results_df = pd.DataFrame.from_dict({(i, j): all_results[i][j] for i in all_results.keys() for j in all_results[i].keys()}, orient='index')
    results_df.to_csv('data/model_comparison_results.csv')
    print("CSV results saved to data/model_comparison_results.csv")


if __name__ == "__main__":
    main()