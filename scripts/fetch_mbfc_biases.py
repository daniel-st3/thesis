# scripts/fetch_mbfc_biases.py

import os
import json
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse


# 1) Configuration: adjust these paths if your folder structure is different

ROOT = Path(__file__).parent.parent         
CSV_INPUT = ROOT / "data" / "final_publisher_categories.csv"
MBFC_JSON  = ROOT / "data" / "mbfc_ratings.json"
CSV_OUTPUT = ROOT / "data" / "publisher_biases_mbfc.csv"


# 2) RapidAPI credentials 

RAPIDAPI_HOST = "media-bias-fact-check-ratings-api2.p.rapidapi.com"
RAPIDAPI_KEY  = "b74fd346f7msh0a60e077acbab79p195718jsnf3e87f419693"


# 3) Step 1: pull down the entire MBFC database (1 API call)

def fetch_all_mbfc():
    """Fetches all MBFC ratings via RapidAPI and writes to mbfc_ratings.json."""
    print("↪ Fetching all MBFC ratings from RapidAPI…")
    url = f"https://{RAPIDAPI_HOST}/fetch-data"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    # Response may contain a UTF-8 BOM at the very start.
    raw_text = response.text
    # If there's a leading BOM (u'\ufeff'), removes it:
    if raw_text.startswith("\ufeff"):
        raw_text = raw_text.lstrip("\ufeff")

    # Load JSON from the cleaned text
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse MBFC JSON response: {e}")

    # Save raw MBFC JSON to disk
    MBFC_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MBFC_JSON, "w", encoding="utf-8") as fout:
        json.dump(data, fout, indent=2)

    print(f"✓ Saved {len(data):,} MBFC entries to {MBFC_JSON.resolve()}")
    return data


# 4) Step 2: build lookup tables for fast matching

def build_mbfc_lookups(mbfc_data):
    """
    Given the list of MBFC entries, build two dicts:
      1) name_lookup[source_name_lower] = bias
      2) url_lookup[domain] = bias
    """
    name_lookup = {}
    url_lookup  = {}

    for entry in mbfc_data:
        source_name = entry.get("Source", "").strip()
        source_url  = entry.get("Source URL", "").strip().lower()
        bias_label  = entry.get("Bias", "").strip()

        if source_name:
            name_lookup[source_name.lower()] = bias_label

        # Extract just the domain (e.g., "nytimes.com", "washingtonpost.com")
        if source_url:
            # If MBFC provided only a bare domain, use that; else parse out the domain from a full URL.
            candidate = source_url if source_url.startswith("http") else f"https://{source_url}"
            parsed = urlparse(candidate)
            domain = parsed.netloc.lower().lstrip("www.")
            if domain:
                url_lookup[domain] = bias_label

    print(f"→ Built name_lookup (entries): {len(name_lookup):,}")
    print(f"→ Built url_lookup (domains): {len(url_lookup):,}")
    return name_lookup, url_lookup


# 5) Step 3: loads final_publisher_categories.csv and attempts to match

def match_publishers_with_mbfc(name_lookup, url_lookup):
    """
    Loads final_publisher_categories.csv, filters to A/B/E (you said you only care about those),
    and then tries to match each publisher to an MBFC bias. Writes out publisher_biases_mbfc.csv
    with a new column 'MBFC_Bias'.  Any unmatched rows remain blank.
    """
    df = pd.read_csv(CSV_INPUT, dtype=str)
    print(f"Loaded {len(df):,} total publishers from {CSV_INPUT.name}")

    # Filter only Category A, B, or E:
    df_filtered = df[df["Category"].isin(["A","B","E"])].copy()
    print(f"Filtering to Categories A/B/E → {len(df_filtered):,} publishers to attempt matching")

    # Prepare a column to hold the matched bias label (if found)
    df_filtered["MBFC_Bias"] = ""

    matched_by_name = 0
    matched_by_url  = 0
    unmatched       = 0

    for idx, row in df_filtered.iterrows():
        pub = row["Publisher"].strip()
        pub_lower = pub.lower()

        # 1) Exact name match?
        if pub_lower in name_lookup:
            df_filtered.at[idx, "MBFC_Bias"] = name_lookup[pub_lower]
            matched_by_name += 1
            continue

        # 2) Try matching via domain substring
        found = False
        for domain, bias in url_lookup.items():
            # If the domain appears inside the publisher name OR vice versa
            if domain in pub_lower or pub_lower.replace(" ", "") in domain:
                df_filtered.at[idx, "MBFC_Bias"] = bias
                matched_by_url += 1
                found = True
                break

        if found:
            continue

        # 3) If neither name nor domain matched:
        unmatched += 1
        df_filtered.at[idx, "MBFC_Bias"] = ""  # leave blank for manual fill later

    print(f"✓ Matched by exact name: {matched_by_name}")
    print(f"✓ Matched by domain heuristic: {matched_by_url}")
    print(f"✗ Left unmatched: {unmatched}")

    # Save out the filtered + bias‐filled CSV
    df_filtered.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")
    print(f"\n→ Saved matched publishers to {CSV_OUTPUT.resolve()}")

    return df_filtered


# 6) Orchestrate everything

def main():
    # (a) Fetch MBFC JSON if it doesn’t already exist, else load from disk
    if not MBFC_JSON.exists():
        mbfc_data = fetch_all_mbfc()
    else:
        print(f"Loading existing MBFC JSON from {MBFC_JSON.name}")
        with open(MBFC_JSON, "r", encoding="utf-8") as fin:
            mbfc_data = json.load(fin)

    # (b) Build lookup tables
    name_lookup, url_lookup = build_mbfc_lookups(mbfc_data)

    # (c) Match your A/B/E publishers to MBFC biases
    df_result = match_publishers_with_mbfc(name_lookup, url_lookup)

    print("✅ Done. Inspect `publisher_biases_mbfc.csv` in the data/ folder.\n"
          "   Any MBFC_Bias that remains blank will need a manual lookup on MBFC’s site or AllSides.")

if __name__ == "__main__":
    main()
