# src/fetch_news_api.py

import os
import yaml
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

from gnews import GNews


root = Path(__file__).parent.parent
cfg = yaml.safe_load(open(root / "config.yaml"))

# ISO strings from config.yaml
start_iso = cfg["date_range"]["start"]  # e.g. "2024-05-31"
end_iso   = cfg["date_range"]["end"]    # e.g. "2025-05-31"

# Converts to datetime objects
start_dt = datetime.fromisoformat(start_iso)
end_dt   = datetime.fromisoformat(end_iso)

TICKERS = cfg["tickers"]
RAW_DIR = root / cfg["paths"]["raw_news"]
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Pause 5 seconds between each Google‐News request
SLEEP_BETWEEN_QUERIES = 5


# 2) INITIALIZE GNEWS CLIENT

# We set an initial full‐range here. 
google_news = GNews(
    language="en",
    country="US",
    # These initial dates will be overwritten in fetch_all_for_ticker()
    start_date=(start_dt.year, start_dt.month, start_dt.day),
    end_date=(end_dt.year,   end_dt.month,   end_dt.day),
    max_results=100,
    exclude_websites=None
)


# 3) WINDOWED FETCH (ALL ARTICLES) 

def fetch_all_for_ticker(ticker: str):
    """
    Fetch *all* Google News articles for `ticker` between start_dt and end_dt
    by slicing into 30-day windows (≤100 results each). Assign directly to
    private attributes _start_date and _end_date to avoid setter warnings.
    """
    all_articles = []
    window_start = start_dt

    while window_start <= end_dt:
        window_end = min(window_start + timedelta(days=30), end_dt)

        # Directly override the private attributes:
        google_news._start_date = window_start
        google_news._end_date   = window_end

        print(f"   (window {window_start.date()} → {window_end.date()}, sleeping {SLEEP_BETWEEN_QUERIES}s …)")
        time.sleep(SLEEP_BETWEEN_QUERIES)

        try:
            subset = google_news.get_news(ticker)
        except Exception as e:
            print(f"⚠️  Error fetching {window_start.date()}–{window_end.date()} for {ticker}: {e}")
            subset = []

        all_articles.extend(subset)
        window_start = window_end + timedelta(days=1)

    # Deduplicate by URL
    unique = { art["url"]: art for art in all_articles }
    return list(unique.values())


# 4) MAIN: LOOP TICKERS AND SAVE JSON

def main():
    for ticker in TICKERS:
        print(f"➡️  Fetching ALL GNews articles for {ticker} between {start_iso} and {end_iso} …")
        print(f"   (initial sleep {SLEEP_BETWEEN_QUERIES}s to avoid blocking…)")
        time.sleep(SLEEP_BETWEEN_QUERIES)

        articles = fetch_all_for_ticker(ticker)
        count = len(articles)

        out_file = RAW_DIR / f"{ticker}_{start_iso}_{end_iso}.json"
        with open(out_file, "w") as f:
            json.dump(articles, f, indent=2)
        print(f"   ✓ Saved {count} articles → {out_file.name}\n")

    print("✅ Done fetching all tickers.")


if __name__ == "__main__":
    main()
