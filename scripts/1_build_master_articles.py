import pandas as pd
import json
from pathlib import Path
from textblob import TextBlob

def build_master_articles():
    """
    Builds the master article-level dataset from the ground up.
    1. Loads all raw news JSONs.
    2. Loads the definitive, user-created publisher bias & category file.
    3. Filters publishers to keep only Categories A, B, and C.
    4. Merges bias and sentiment onto the filtered articles.
    5. Saves the final, clean, enriched article dataset.
    """
    print("--- Step 1: Building Filtered & Enriched Master Article Dataset ---")

    # --- Configuration ---
    RAW_DIR = Path("data/news_raw")
    BIAS_FILE_PATH = Path("data/publisher_biases_complete.csv")
    OUTPUT_PATH = Path("data/1_articles_filtered_and_enriched.csv")
    
    # --- 1. Load All Raw News Articles ---
    records = []
    print(f"Loading raw news from: {RAW_DIR}")
    for f in RAW_DIR.glob("*.json"):
        ticker = f.stem.split("_")[0]
        with open(f, 'r', encoding='utf-8') as file:
            articles = json.load(file)
            for article in articles:
                publisher_info = article.get("publisher", {})
                records.append({
                    "ticker": ticker,
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "published_date": article.get("published date", ""),
                    "url": article.get("url", ""),
                    "publisher_name": publisher_info.get("title") if isinstance(publisher_info, dict) else None
                })
    df_news = pd.DataFrame(records)
    print(f"Loaded {len(df_news)} total raw articles.")

    # --- 2. Load and Filter Publisher Biases ---
    print(f"Loading publisher classifications from: {BIAS_FILE_PATH}")
    df_bias = pd.read_csv(BIAS_FILE_PATH)
    
    # Filtering heuristic
    categories_to_keep = ['A', 'B', 'C']
    df_filtered_publishers = df_bias[df_bias['Category'].isin(categories_to_keep)].copy()
    print(f"Filtered down to {len(df_filtered_publishers)} publishers in Categories A, B, and C.")

    # --- 3. Merge and Filter Articles ---
    df_filtered_publishers = df_filtered_publishers[['Publisher', 'Category', 'MBFC_Bias']]
    df_filtered_publishers.rename(columns={'MBFC_Bias': 'bias_label'}, inplace=True)

    # Kept ONLY articles from the filtered publishers
    df_master = pd.merge(df_news, df_filtered_publishers, left_on='publisher_name', right_on='Publisher', how='inner')
    print(f"After merging and filtering, we have {len(df_master)} relevant articles.")

    # --- 4. Compute Sentiment ---
    print("Computing sentiment for relevant articles...")
    def calculate_sentiment(row):
        text = str(row["title"]) + ". " + str(row["description"])
        return TextBlob(text).sentiment.polarity
    df_master['sentiment'] = df_master.apply(calculate_sentiment, axis=1)

    # --- 5. Save the Output ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

    print(f"\n--- Success! ---")
    print(f"Master article dataset has been built and saved to: {OUTPUT_PATH}")
    print("This file now contains only relevant articles, enriched with your detailed bias labels and sentiment.")

if __name__ == "__main__":
    build_master_articles()