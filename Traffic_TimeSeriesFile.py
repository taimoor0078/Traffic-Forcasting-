import os
import glob
import pandas as pd

# =====================================================
# CONFIGURATION
# =====================================================
TIMESERIES_FOLDER = "timeseries"
OUTPUT_FILE = "traffic_timeseries.csv"

# =====================================================
# LOAD ALL TIMESERIES FILES
# =====================================================
files = sorted(glob.glob(os.path.join(TIMESERIES_FOLDER, "*_timeseries.csv")))

if len(files) == 0:
    raise Exception("No time-series files found!")

print("=" * 70)
print(f"Found {len(files)} time-series files")
print("=" * 70)

merged_data = []
time_offset = 0

for file in files:

    df = pd.read_csv(file)

    if df.empty:
        print(f"Skipped Empty File : {os.path.basename(file)}")
        continue

    # Ensure sorting
    df = df.sort_values("Time").reset_index(drop=True)

    # Make timeline continuous
    df["Time"] = df["Time"] + time_offset

    merged_data.append(df)

    print(
        f"{os.path.basename(file):35}"
        f" Rows={len(df):5d}"
        f" Time={df['Time'].min()} -> {df['Time'].max()}"
    )

    # Update offset for next file
    time_offset = df["Time"].max() + 1

# =====================================================
# MERGE
# =====================================================
final_df = pd.concat(merged_data, ignore_index=True)

# Remove duplicates if any
final_df = final_df.drop_duplicates(subset=["Time"])

# Sort again
final_df = final_df.sort_values("Time").reset_index(drop=True)

# Save
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("MERGE COMPLETED SUCCESSFULLY")
print("=" * 70)
print("Total Files :", len(files))
print("Total Rows  :", len(final_df))
print("Time Range  :", final_df["Time"].min(), "->", final_df["Time"].max())
print("Saved File  :", OUTPUT_FILE)
print("=" * 70)

print("\nPreview:")
print(final_df.head())

print("\nLast Rows:")
print(final_df.tail())