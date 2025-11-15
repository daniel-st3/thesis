import yaml
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_and_format_prices():
    """
    Fetches historical stock prices for tickers listed in config.yaml,
    formats them into a long format, and saves the data to a CSV file.
    """
    # Load configuration from config.yaml
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        tickers = config['tickers']
        start_date = config['start_date']
        # Set end_date to today for the most recent data
        end_date = datetime.now().strftime('%Y-%m-%d')
        output_path = config['paths']['daily_prices']
        
        print(f"Configuration loaded. Tickers: {tickers}")
        print(f"Fetching data from {start_date} to {end_date}")

    except FileNotFoundError:
        print("Error: config.yaml not found. Please ensure the file exists.")
        return
    except KeyError as e:
        print(f"Error: Missing key in config.yaml: {e}")
        return

    # Created the data directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Fetched data using yfinance
    try:
        # Download historical data for all tickers at once
        # yfinance returns a DataFrame with multi-level columns
        prices_df = yf.download(tickers, start=start_date, end=end_date)
        
        if prices_df.empty:
            print("No data downloaded. Please check tickers and date range.")
            return

        # We only need the 'Adj Close' prices
        adj_close_df = prices_df['Adj Close']

        # The data is in "wide" format (tickers as columns).
        # We need to convert it to "long" format.
        # stack() pivots the columns into a multi-level index
        # reset_index() converts the index levels into columns
        long_format_df = adj_close_df.stack().reset_index()

        # Rename the columns to match the project's requirements
        long_format_df.rename(columns={
            'level_0': 'date',
            'level_1': 'ticker',
            0: 'adj_close'
        }, inplace=True)
        
        # Ensure the 'date' column is in the correct format (YYYY-MM-DD)
        long_format_df['date'] = pd.to_datetime(long_format_df['date']).dt.strftime('%Y-%m-%d')

        # Save the formatted data to the specified path
        long_format_df.to_csv(output_path, index=False)
        
        print(f"\nSuccessfully fetched and formatted price data.")
        print(f"Data saved to: {output_path}")
        print(f"Total rows created: {len(long_format_df)}")
        print("\nPreview of the first 5 rows:")
        print(long_format_df.head())

    except Exception as e:
        print(f"\nAn error occurred during the process: {e}")

if __name__ == "__main__":
    fetch_and_format_prices()