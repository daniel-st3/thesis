# src/build_features.py
import json
from pathlib import Path
import pandas as pd
from textblob import TextBlob

RAW_DIR      = Path("data") / "news_raw"
BIAS_CSV     = Path("data") / "publisher_biases_complete.csv"
OUT_ART_CSV  = Path("data") / "features" / "news_bias_features.csv"
OUT_DAILY_PQ = Path("data") / "features" / "news_features.parquet"

def load_raw_articles():
    records = []
    for f in RAW_DIR.glob("*.json"):
        # ticker is the first part of the filename (e.g. "AAPL_…")
        ticker = f.stem.split("_")[0]
        try:
            arr = json.load(open(f, encoding="utf8"))
        except Exception:
            print(f"⚠️  Skipping invalid JSON: {f.name}")
            continue
        for rec in arr:
            pub = rec.get("publisher", {})
            pub_name = pub.get("title") if isinstance(pub, dict) else None
            records.append({
                "ticker": ticker,
                "title": rec.get("title", ""),
                "description": rec.get("description", ""),
                "published_date": rec.get("published date", ""),
                "url": rec.get("url", ""),
                "publisher": pub_name or "",
            })
    if not records:
        raise RuntimeError(f"No valid JSON files found in {RAW_DIR}")
    return pd.DataFrame(records)

def compute_sentiment(df):
    # combine title+description, compute TextBlob.polarity
    def polarity(row):
        text = (row["title"] or "") + ". " + (row["description"] or "")
        return TextBlob(text).sentiment.polarity
    df["sentiment"] = df.apply(polarity, axis=1)
    return df

def load_bias_lookup():
    df = pd.read_csv(BIAS_CSV, dtype=str)
    # numeric mapping
    numeric = {
        "Left": -1.0,
        "Left-Center": -0.5,
        "Least Biased": 0.0,
        "Right-Center": 0.5,
        "Right": 1.0,
    }
    # keep only those with a mapped label
    df = df[df["MBFC_Bias"].isin(numeric)].copy()
    df["Num_Bias"] = df["MBFC_Bias"].map(numeric).astype(float)
    # build map publisher name -> numeric
    return df.set_index("Publisher")["Num_Bias"].to_dict()

def main():
    # 1) load raw + extract ticker/publisher
    df = load_raw_articles()
    print(f"🔍 Loaded {len(df)} raw articles")

    # 2) compute sentiment
    df = compute_sentiment(df)
    print("✏️  Computed sentiment polarity")

    # 3) load bias lookup and map
    bias_map = load_bias_lookup()
    print(f"⚙️  Loaded {len(bias_map)} numeric publisher biases")
    df["bias"] = df["publisher"].map(bias_map).fillna(0.0)

    # 4) parse dates
    df["date"] = pd.to_datetime(
        df["published_date"], utc=True, errors="coerce"
    ).dt.date
    n_bad_dates = df["date"].isna().sum()
    if n_bad_dates:
        print(f"⚠️  Dropped {n_bad_dates} rows with invalid dates")
    df = df.dropna(subset=["date"])

    # 5) save per‐article features
    OUT_ART_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ART_CSV, index=False, encoding="utf8")
    print(f"💾 Saved {len(df)} rows to {OUT_ART_CSV}")

    # 6) aggregate to daily features
    feat = (
        df.groupby(["ticker", "date"])
          .agg(
             avg_sentiment = ("sentiment", "mean"),
             avg_bias      = ("bias",      "mean"),
             disp_bias     = ("bias",      "std"),
             n_articles    = ("bias",      "count"),
          )
          .reset_index()
    )
    # fill missing dispersion with 0
    feat["disp_bias"] = feat["disp_bias"].fillna(0.0)

    # 7) write daily parquet
    OUT_DAILY_PQ.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(OUT_DAILY_PQ, index=False)
    print(f"✅ Saved {len(feat)} daily rows to {OUT_DAILY_PQ}")

if __name__ == "__main__":
    main()
