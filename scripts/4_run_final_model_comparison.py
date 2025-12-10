import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def run_model_comparison(X, y, label, model):
    """Run time-series cross-validation for a given model and feature set.

    Handles class imbalance using SMOTE within each fold. Returns the mean of
    each metric across folds.
    """
    print(f"--- Running {label} ---")

    tscv = TimeSeriesSplit(n_splits=5)
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("classifier", model),
        ])

        pipeline.fit(X_train, y_train)

        # Use the full pipeline for inference to ensure consistent preprocessing
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        scores["accuracy"].append(accuracy_score(y_test, y_pred))
        scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        scores["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        scores["auc"].append(roc_auc_score(y_test, y_pred_proba))

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

    technical_indicators = ['rsi', 'macd', 'macd_signal']
    sentiment_features = ['retail_sentiment', 'avg_sentiment']
    bias_features = ['avg_bias', 'disp_bias']
    required_columns = set(technical_indicators + sentiment_features + bias_features + ['target_up'])

    missing_cols = required_columns.difference(df.columns)
    if missing_cols:
        print(f"The following required columns are missing from the dataset: {sorted(missing_cols)}")
        print("Please regenerate the modeling dataset to include these fields.")
        return

    feature_sets = {
        'baseline': technical_indicators + ['avg_sentiment'],
        'bias_only': technical_indicators + ['avg_sentiment'] + bias_features,
        'retail_only': technical_indicators + ['avg_sentiment', 'retail_sentiment'],
        'enhanced': technical_indicators + sentiment_features + bias_features,
    }

    y = df['target_up']

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = []

    # Run comparison for all models and feature sets
    for name, model in models.items():
        print(f"\n===== Evaluating Model: {name} =====")

        for feature_label, columns in feature_sets.items():
            X = df[columns]
            label = f"{name} ({feature_label})"
            scores = run_model_comparison(X, y, label, model)
            print(f"Average {feature_label.title()} Scores: {scores}")
            results.append({'model': name, 'feature_set': feature_label, **scores})

    results_df = pd.DataFrame(results)

    with open('data/model_comparison_results.pkl', 'wb') as f:
        pickle.dump(results_df, f)

    print("\n\nComparison complete. Results saved to data/model_comparison_results.pkl")

    results_df.to_csv('data/model_comparison_results.csv', index=False)
    print("CSV results saved to data/model_comparison_results.csv")


if __name__ == "__main__":
    main()
