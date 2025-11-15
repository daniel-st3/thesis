import praw
import yaml
import pandas as pd
import os
from datetime import datetime

def fetch_reddit_data():
    """
    Fetches data from specified subreddits for a list of stock tickers.
    """
    # --- Load Config and Credentials ---
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        tickers = config['tickers']
        
        with open('credentials.yaml', 'r') as file:
            creds = yaml.safe_load(file)['reddit']

        print("Configuration and credentials loaded.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a necessary file. {e}")
        return

    # --- Initialize PRAW (Reddit instance) ---
    try:
        reddit = praw.Reddit(
            client_id=creds['client_id'],
            client_secret=creds['client_secret'],
            user_agent=creds['user_agent'],
            username=creds['username'],
            password=creds['password'],
        )
        # Checks if the connection is read-only or authenticated
        print(f"Reddit instance created. Read-only: {reddit.read_only}")
        if reddit.read_only:
             print("Warning: Connection is read-only. Check credentials if you need to post/comment.")
    except Exception as e:
        print(f"Error connecting to Reddit: {e}")
        return

    # --- Defines Search Parameters ---
    subreddits_to_search = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
    limit_per_subreddit = 200 # Number of posts to fetch from each subreddit
    
    fetched_posts = []

    print(f"\nStarting to fetch data from subreddits: {subreddits_to_search}")

    # --- Fetchs Data ---
    for sub_name in subreddits_to_search:
        print(f"--- Searching in r/{sub_name} ---")
        subreddit = reddit.subreddit(sub_name)
        
        # Fetching the 'hot' posts
        for submission in subreddit.hot(limit=limit_per_subreddit):
            # Checks for any ticker mention in title or body
            for ticker in tickers:
                if ticker.lower() in submission.title.lower() or ticker.lower() in submission.selftext.lower():
                    post_data = {
                        'ticker_mentioned': ticker,
                        'subreddit': sub_name,
                        'title': submission.title,
                        'selftext': submission.selftext,
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'created_utc': datetime.fromtimestamp(submission.created_utc)
                    }
                    fetched_posts.append(post_data)
                    break 

    if not fetched_posts:
        print("\nNo posts found matching the tickers. Try increasing the limit or changing subreddits.")
        return

    # --- Saves Data ---
    print(f"\nFinished fetching. Found {len(fetched_posts)} relevant posts.")
    
    # Converts list of dicts to a DataFrame
    df = pd.DataFrame(fetched_posts)
    
    # Defines output path
    output_path = "data/raw_reddit_data.csv"
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Saves to CSV
    df.to_csv(output_path, index=False)
    print(f"Data saved successfully to {output_path}")
    print("\nPreview of the first 5 rows:")
    print(df.head())


if __name__ == "__main__":
    fetch_reddit_data()