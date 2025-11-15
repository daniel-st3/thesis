# scripts/fill_missing_mbfc.py

import pandas as pd
from pathlib import Path

# 1) Loads the existing CSV with MBFC_Bias blanks
ROOT_DIR = Path(__file__).parent.parent
input_csv = ROOT_DIR / "data" / "publisher_biases_mbfc.csv"

print(f"📂 Loading publishers from: {input_csv.resolve()}")
df = pd.read_csv(input_csv, dtype=str)

# Ensures the MBFC_Bias column exists:
if "MBFC_Bias" not in df.columns:
    raise KeyError("CSV does not have an 'MBFC_Bias' column.")

# 2) Finds rows where MBFC_Bias is blank or NaN
is_blank = df["MBFC_Bias"].isna() | (df["MBFC_Bias"].str.strip() == "")
missing_df = df[is_blank].copy()

if missing_df.empty:
    print("✅ No missing MBFC_Bias entries—nothing to fill.")
    exit(0)

print(f"⚠️  Found {len(missing_df)} publishers with missing MBFC_Bias.\n")

# Allowed set of MBFC labels:
allowed_labels = {
    "Left", "Left-Center", "Least Biased", "Right-Center", "Right", 
    "Questionable", "Pro-Science"
}

# 3) Loops through each missing‐bias row and prompt the user to enter one
for idx, row in missing_df.iterrows():
    pub = row["Publisher"]
    cat = row["Category"]
    while True:
        raw = input(
            f"Enter MBFC_Bias for “{pub}” (Category: {cat})\n"
            f"  Options: {', '.join(sorted(allowed_labels))}\n"
            f"  → "
        ).strip()
        if raw == "":
            print("   ✖️  You must type one of the allowed labels (cannot be blank).")
            continue
        if raw not in allowed_labels:
            print(f"   ✖️  “{raw}” is not in the allowed list. Please choose exactly one of:")
            print("      ", ", ".join(sorted(allowed_labels)))
            continue

        # If we get here, `raw` is valid:
        df.at[idx, "MBFC_Bias"] = raw
        break

# 4) Shows how many remain blank (should be zero now)
still_blank = (df["MBFC_Bias"].isna()) | (df["MBFC_Bias"].str.strip() == "")
if still_blank.any():
    print("\n⚠️  WARNING: Some MBFC_Bias entries are still blank:")
    print(df.loc[still_blank, ["Publisher","Category"]])
    print("\nPlease re-run the script to fill the remaining blanks.")
else:
    print("\n✅ All missing MBFC_Bias entries have been filled by the user.")

# 5) Saves the newly‐filled CSV (overwriting the original)
df.to_csv(input_csv, index=False, encoding="utf-8")
print(f"\n✅  Updated CSV saved to:\n   {input_csv.resolve()}")
