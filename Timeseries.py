import os
import glob
import pandas as pd

# ========= SETTINGS =========
DETECTIONS_FOLDER = "detections"      # apna detections folder
OUTPUT_FILE = "traffic_timeseries.csv"
AGGREGATION_SECONDS = 1               # 1 second window
# ============================

all_files = glob.glob(os.path.join(DETECTIONS_FOLDER, "*_detections.csv"))

if len(all_files) == 0:
    raise Exception("No detection CSV files found!")

print(f"Found {len(all_files)} detection files")

dfs = []

for file in all_files:
    df = pd.read_csv(file)

    # agar timestamp column ka naam different ho to yahan change kar lena
    if "timestamp_sec" not in df.columns:
        print(f"Skipping {os.path.basename(file)} (timestamp_sec missing)")
        continue

    df["Time"] = df["timestamp_sec"].astype(int)

    dfs.append(df)

detections = pd.concat(dfs, ignore_index=True)

print("Total detections:", len(detections))

vehicle_classes = [
    "Car",
    "Bus",
    "Truck",
    "Van",
    "Pickup",
    "Taxi",
    "Motorcycle",
    "Bicycle"
]

rows = []

for sec, group in detections.groupby("Time"):

    row = {"Time": sec}

    total = 0

    for cls in vehicle_classes:
        count = (group["class_name"] == cls).sum()
        row[cls] = count
        total += count

    row["Persons"] = (group["class_name"] == "Person").sum()

    row["EmergencyVehicles"] = (
        group["class_name"].isin(
            ["Ambulance", "PoliceVehicle", "FireTruck"]
        ).sum()
    )

    row["TotalVehicles"] = total + row["EmergencyVehicles"]

    rows.append(row)

timeseries = pd.DataFrame(rows).sort_values("Time")

timeseries.to_csv(OUTPUT_FILE, index=False)

print("-------------------------------------")
print("Time Series Created Successfully")
print("Rows :", len(timeseries))
print("Saved:", OUTPUT_FILE)
print("-------------------------------------")