# scripts/find_unmatched_publishers.py

import pandas as pd
from pathlib import Path

# 1) CSV files:
article_counts_path = Path("data") / "publisher_article_counts.csv"
bias_complete_path  = Path("data") / "publisher_biases_complete.csv"

# 2) Loads article_counts and “complete” bias lookup:
df_counts   = pd.read_csv(article_counts_path, dtype=str)
df_bias_all = pd.read_csv(bias_complete_path,  dtype=str)

# 3) Builds two sets of “Publisher” strings (strip whitespace):
counts_pubs = set(df_counts["Publisher"].str.strip())
bias_pubs   = set(df_bias_all["Publisher"].str.strip())

# 4) Finds publishers in article_counts.csv → NOT IN → publisher_biases_complete.csv:
still_missing = sorted(counts_pubs - bias_pubs)
print(f"\n❗ Publishers in article_counts.csv but MISSING from publisher_biases_complete.csv ({len(still_missing)}):\n")
for name in still_missing:
    print("  -", name)

# 5) Finds publishers in publisher_biases_complete.csv → NOT IN → article_counts.csv:
extra_in_bias = sorted(bias_pubs - counts_pubs)
print(f"\n❗ Publishers in publisher_biases_complete.csv but NOT FOUND in article_counts.csv ({len(extra_in_bias)}):\n")
for name in extra_in_bias:
    print("  -", name)

# 6) Checks for any blank MBFC_Bias cells in publisher_biases_complete.csv:
blank_mask = df_bias_all["MBFC_Bias"].isna() | (df_bias_all["MBFC_Bias"].str.strip() == "")
num_blanks = blank_mask.sum()
print(f"\nℹ️  Number of publishers still missing any MBFC_Bias label: {num_blanks}\n")
if num_blanks > 0:
    print("Publishers with blank MBFC_Bias:\n")
    for pub in df_bias_all.loc[blank_mask, "Publisher"]:
        print("  -", pub)
print()
