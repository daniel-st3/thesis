import pandas as pd
import yaml
import os

def merge_features_with_prices():
    """
    Merges the engineered news features with the daily stock prices.
    """
    # Loads configuration from config.yaml
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        prices_path = config['paths']['daily_prices']
        news_features_path = config['paths']['news_bias_features']
        output_path = config['paths']['merged_news_prices']
        
        print("Configuration loaded successfully.")
        print(f"Prices file: {prices_path}")
        print(f"News features file: {news_features_path}")

    except FileNotFoundError:
        print("Error: config.yaml not found. Please ensure the file exists.")
        return
    except KeyError as e:
        print(f"Error: Missing key in config.yaml: {e}")
        return

    # Checks if input files exist
    if not os.path.exists(prices_path):
        print(f"Error: Price data file not found at {prices_path}")
        return
    if not os.path.exists(news_features_path):
        print(f"Error: News features file not found at {news_features_path}")
        return

    # Loads the datasets
    print("\nLoading datasets...")
    prices_df = pd.read_csv(prices_path)
    news_df = pd.read_csv(news_features_path)

    # --- Data Preparation ---
    # Ensures date columns are in a consistent datetime format for merging
    prices_df['date'] = pd.to_datetime(prices_df['date']).dt.strftime('%Y-%m-%d')
    news_df['date'] = pd.to_datetime(news_df['date']).dt.strftime('%Y-%m-%d')

    print(f"Loaded {len(prices_df)} rows from price data.")
    print(f"Loaded {len(news_df)} rows from news features.")

    # --- Merging ---
    # Performs an inner merge to keep only the days where we have BOTH
    # price data and news features. This is crucial for modeling.
    print("\nMerging dataframes on 'date' and 'ticker'...")
    merged_df = pd.merge(prices_df, news_df, on=['date', 'ticker'], how='inner')

    if merged_df.empty:
        print("\nWarning: The merged dataframe is empty.")
        print("This means there were no matching dates and tickers between the two files.")
        print("Please check the date ranges and ticker names in both CSV files.")
        return

    # Creates the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Saves the merged dataframe
    merged_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully merged data.")
    print(f"Merged data saved to: {output_path}")
    print(f"Total rows in merged file: {len(merged_df)}")
    print("\nPreview of the first 5 rows of the merged data:")
    print(merged_df.head())
    print("\nColumns in the new merged file:")
    print(merged_df.columns.tolist())

if __name__ == "__main__":
    merge_features_with_prices()