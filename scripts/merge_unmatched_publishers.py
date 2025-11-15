# scripts/merge_unmatched_publishers.py

import pandas as pd
from pathlib import Path

# 1) Paths 
article_counts_path = Path("data") / "publisher_article_counts.csv"
bias_lookup_path    = Path("data") / "publisher_biases_mbfc.csv"
output_path         = Path("data") / "publisher_biases_complete.csv"

# 2) Loads both CSVs
df_counts = pd.read_csv(article_counts_path, dtype=str)
df_bias   = pd.read_csv(bias_lookup_path,  dtype=str)

# 3) Builds sets of publisher names (strip whitespace)
counts_publishers = set(df_counts["Publisher"].str.strip())
bias_publishers   = set(df_bias["Publisher"].str.strip())

# 4) Finds publishers present in counts but missing from bias_lookup
unmatched = sorted(counts_publishers - bias_publishers)
print(f"ℹ️  Found {len(unmatched)} publishers in article_counts.csv that are missing from publisher_biases_mbfc.csv\n")
for name in unmatched:
    print("  -", name)
print()

# 5) Creates a DataFrame of those unmatched names, with empty MBFC_Bias
df_unmatched = pd.DataFrame({
    "Publisher": unmatched,
    "MBFC_Bias": ""   # leave blank so you can fill by hand later
})

# 6) If df_bias has extra columns beyond ["Publisher","MBFC_Bias"], add those
#    columns (empty) to df_unmatched so both DataFrames share the same columns.
for col in df_bias.columns:
    if col not in df_unmatched.columns:
        df_unmatched[col] = ""

# Re‐orders df_unmatched to match df_bias’s column order exactly
df_unmatched = df_unmatched[df_bias.columns]

# 7) Concatenates existing bias_lookup + these new blank‐bias rows
df_complete = pd.concat([df_bias, df_unmatched], ignore_index=True)

# 8) Saves out the combined file
df_complete.to_csv(output_path, index=False)
print(f"✅  Combined bias lookup saved to:\n   {output_path.resolve()}\n")
print("▶️  Now open that CSV and fill in each blank MBFC_Bias by hand.")
